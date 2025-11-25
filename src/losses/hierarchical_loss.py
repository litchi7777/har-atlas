"""
Hierarchical SSL Loss for Motion Primitive Discovery

3階層のContrastive Learningを実装:
- L_activity: 同じActivity同士をpositive（データセット内）
- L_atomic: Body Part別のPrototype学習（PiCO）

Usage:
    loss_fn = HierarchicalSSLLoss(
        atlas_path="docs/atlas/activity_mapping.json",
        embed_dim=512,
        temperature=0.1
    )

    # Forward
    total_loss, loss_dict = loss_fn(
        embeddings=embeddings,          # (batch, embed_dim)
        dataset_ids=dataset_ids,        # (batch,) データセット名
        activity_ids=activity_ids,      # (batch,) Activity名
        body_parts=body_parts           # (batch,) Body Part名
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import sys
from pathlib import Path

# Atlas Loaderをインポート
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.atlas_loader import AtlasLoader


class BodyPartPrototypes(nn.Module):
    """
    Body Part別のPrototype管理

    各Body Partに対して、Atomic Motion数分のPrototypeベクトルを管理
    """

    def __init__(
        self,
        body_parts: List[str],
        num_prototypes_per_part: Dict[str, int],
        embed_dim: int = 512,
        prototype_dim: int = 128,
    ):
        """
        Args:
            body_parts: Body Partのリスト ["wrist", "hip", "chest", "leg", "head"]
            num_prototypes_per_part: 各Body PartのPrototype数 {"wrist": 27, ...}
            embed_dim: 入力埋め込み次元
            prototype_dim: Prototype空間の次元
        """
        super().__init__()

        self.body_parts = body_parts
        self.num_prototypes = num_prototypes_per_part
        self.embed_dim = embed_dim
        self.prototype_dim = prototype_dim

        # Body Part別のProjection Head
        self.projections = nn.ModuleDict({
            bp: nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, prototype_dim),
            )
            for bp in body_parts
        })

        # Body Part別のPrototypeベクトル
        self.prototypes = nn.ParameterDict({
            bp: nn.Parameter(torch.randn(num_prototypes_per_part[bp], prototype_dim))
            for bp in body_parts
        })

        # Prototypeを正規化
        self._normalize_prototypes()

    def _normalize_prototypes(self):
        """全Prototypeを単位ベクトルに正規化"""
        with torch.no_grad():
            for bp in self.body_parts:
                self.prototypes[bp].data = F.normalize(
                    self.prototypes[bp].data, dim=1
                )

    def project(self, embeddings: torch.Tensor, body_part: str) -> torch.Tensor:
        """
        埋め込みをBody Part別のPrototype空間に射影

        Args:
            embeddings: (batch, embed_dim)
            body_part: Body Part名

        Returns:
            (batch, prototype_dim) 正規化された射影ベクトル
        """
        if body_part not in self.projections:
            raise ValueError(f"Unknown body part: {body_part}")

        projected = self.projections[body_part](embeddings)
        return F.normalize(projected, dim=1)

    def get_prototype_scores(
        self,
        embeddings: torch.Tensor,
        body_part: str,
        temperature: float = 0.1
    ) -> torch.Tensor:
        """
        各Prototypeとの類似度スコアを計算

        Args:
            embeddings: (batch, embed_dim)
            body_part: Body Part名
            temperature: 温度パラメータ

        Returns:
            (batch, num_prototypes) 各Prototypeとの類似度
        """
        projected = self.project(embeddings, body_part)  # (batch, prototype_dim)
        prototypes = F.normalize(self.prototypes[body_part], dim=1)  # (num_proto, prototype_dim)

        # コサイン類似度
        scores = torch.mm(projected, prototypes.t()) / temperature  # (batch, num_proto)
        return scores

    def get_soft_assignments(
        self,
        embeddings: torch.Tensor,
        body_part: str,
        candidate_ids: Optional[torch.Tensor] = None,
        temperature: float = 0.1
    ) -> torch.Tensor:
        """
        Soft Prototype割り当てを計算（PiCO用）

        Args:
            embeddings: (batch, embed_dim)
            body_part: Body Part名
            candidate_ids: (batch, max_candidates) 各サンプルの候補Prototype ID
                          Noneの場合は全Prototypeが候補
            temperature: 温度パラメータ

        Returns:
            (batch, num_prototypes) Soft割り当て確率
        """
        scores = self.get_prototype_scores(embeddings, body_part, temperature)

        if candidate_ids is not None:
            # 候補以外のPrototypeをマスク
            mask = torch.zeros_like(scores, dtype=torch.bool)
            batch_indices = torch.arange(scores.size(0), device=scores.device)

            for i in range(candidate_ids.size(1)):
                valid = candidate_ids[:, i] >= 0  # -1はパディング
                mask[batch_indices[valid], candidate_ids[valid, i]] = True

            scores = scores.masked_fill(~mask, float("-inf"))

        return F.softmax(scores, dim=1)


class ActivityContrastiveLoss(nn.Module):
    """
    Activity-levelのContrastive Loss

    同じActivity同士をpositive、違うActivityをnegativeとして学習
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        activity_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, embed_dim) 正規化済み埋め込み
            activity_labels: (batch,) Activity ID

        Returns:
            Contrastive Loss
        """
        batch_size = embeddings.size(0)
        device = embeddings.device

        if batch_size < 2:
            # embeddingsの0倍を返して計算グラフを維持
            return (embeddings.sum() * 0.0)

        # 正規化
        embeddings = F.normalize(embeddings, dim=1)

        # 類似度行列
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature

        # 同じActivityかどうかのマスク
        labels = activity_labels.view(-1, 1)
        positive_mask = (labels == labels.t()).float()

        # 自分自身を除外
        self_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask * (1 - self_mask)

        # 正例がない場合はスキップ
        num_positives = positive_mask.sum(dim=1)
        valid_samples = num_positives > 0

        if not valid_samples.any():
            # embeddingsの0倍を返して計算グラフを維持
            return (embeddings.sum() * 0.0)

        # 対角成分を大きな負値でマスク（-infではなく）
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()  # 数値安定化
        sim_matrix = sim_matrix.masked_fill(self_mask.bool(), -1e9)

        # InfoNCE Loss: -log(exp(pos) / sum(exp(all)))
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # 正例の平均
        loss = -(positive_mask * log_prob).sum(dim=1) / (num_positives + 1e-8)
        loss = loss[valid_samples].mean()

        return loss


class PrototypeContrastiveLoss(nn.Module):
    """
    Prototype-levelのContrastive Loss（PiCO inspired）

    Body Part別にPrototype割り当てを学習
    同じPrototypeに割り当てられたサンプル同士がpositive
    """

    def __init__(self, temperature: float = 0.1, momentum: float = 0.99):
        super().__init__()
        self.temperature = temperature
        self.momentum = momentum

    def forward(
        self,
        embeddings: torch.Tensor,
        soft_assignments: torch.Tensor,
        activity_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, embed_dim) 正規化済み埋め込み
            soft_assignments: (batch, num_prototypes) Soft Prototype割り当て
            activity_labels: (batch,) Activity ID（同じActivityも確実にpositive）

        Returns:
            Prototype Contrastive Loss
        """
        batch_size = embeddings.size(0)
        device = embeddings.device

        if batch_size < 2:
            # embeddingsの0倍を返して計算グラフを維持
            return (embeddings.sum() * 0.0)

        # 正規化
        embeddings = F.normalize(embeddings, dim=1)

        # 類似度行列
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature

        # Soft positive重み（Prototype割り当ての類似度）
        # soft_assignments: (batch, num_proto)
        # positive_weights[i,j] = sum_k(q_i[k] * q_j[k]) = Prototype割り当ての内積
        positive_weights = torch.mm(soft_assignments, soft_assignments.t())

        # 同じActivityは確実にpositive（重み1.0）
        if activity_labels is not None:
            labels = activity_labels.view(-1, 1)
            same_activity = (labels == labels.t()).float()
            positive_weights = torch.max(positive_weights, same_activity)

        # 自分自身を除外
        self_mask = torch.eye(batch_size, device=device)
        positive_weights = positive_weights * (1 - self_mask)

        # 重みがない場合はスキップ
        weight_sum = positive_weights.sum(dim=1)
        valid_samples = weight_sum > 0

        if not valid_samples.any():
            # embeddingsの0倍を返して計算グラフを維持
            return (embeddings.sum() * 0.0)

        # 対角成分を大きな負値でマスク（数値安定化）
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()
        sim_matrix = sim_matrix.masked_fill(self_mask.bool(), -1e9)

        # Weighted InfoNCE Loss
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # 重み付き正例の和
        weighted_log_prob = positive_weights * log_prob
        loss = -weighted_log_prob.sum(dim=1) / (weight_sum + 1e-8)
        loss = loss[valid_samples].mean()

        return loss


class HierarchicalSSLLoss(nn.Module):
    """
    階層的SSL損失

    L_total = λ_activity * L_activity + λ_prototype * L_prototype

    - L_activity: 同じActivity同士をpositive
    - L_prototype: Body Part別のPrototype学習（同じPrototype = positive）
    """

    def __init__(
        self,
        atlas_path: str,
        embed_dim: int = 512,
        prototype_dim: int = 128,
        temperature: float = 0.1,
        lambda_activity: float = 0.5,
        lambda_prototype: float = 0.5,
    ):
        """
        Args:
            atlas_path: activity_mapping.json へのパス
            embed_dim: エンコーダー出力次元
            prototype_dim: Prototype空間の次元
            temperature: Contrastive Loss温度
            lambda_activity: Activity Loss重み
            lambda_prototype: Prototype Loss重み
        """
        super().__init__()

        # Atlas読み込み
        self.atlas = AtlasLoader(atlas_path)

        # Body Part設定
        self.body_parts = AtlasLoader.NORMALIZED_BODY_PARTS

        # 各Body PartのPrototype数（= Atomic Motion数）をatomic_motions.jsonから取得
        num_prototypes = self.atlas.get_prototype_counts()
        # 最低1つは必要
        for bp in self.body_parts:
            if bp not in num_prototypes or num_prototypes[bp] == 0:
                num_prototypes[bp] = 1

        # Prototype管理
        self.prototypes = BodyPartPrototypes(
            body_parts=self.body_parts,
            num_prototypes_per_part=num_prototypes,
            embed_dim=embed_dim,
            prototype_dim=prototype_dim,
        )

        # Atomic Motion → ID マッピング
        self.atomic_to_id = self.atlas.get_atomic_motion_to_id()

        # Loss関数
        self.activity_loss = ActivityContrastiveLoss(temperature=temperature)
        self.prototype_loss = PrototypeContrastiveLoss(temperature=temperature)

        # 重み
        self.lambda_activity = lambda_activity
        self.lambda_prototype = lambda_prototype
        self.temperature = temperature

    def get_candidate_prototype_ids(
        self,
        dataset: str,
        activity: str,
        body_part: str,
        max_candidates: int = 10,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Activity + Body Partに対する候補Prototype IDを取得

        Returns:
            (max_candidates,) 候補ID（不足分は-1でパディング）
        """
        try:
            candidates = self.atlas.get_candidate_atomic_ids(
                dataset, activity, body_part, self.atomic_to_id
            )
        except KeyError:
            # Atlasに登録されていないActivityの場合、空リストを返す
            # → 全Prototypeを候補とする（制約なし）
            candidates = []

        # パディング
        if len(candidates) < max_candidates:
            candidates = candidates + [-1] * (max_candidates - len(candidates))
        else:
            candidates = candidates[:max_candidates]

        tensor = torch.tensor(candidates, dtype=torch.long)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    def forward(
        self,
        embeddings: torch.Tensor,
        dataset_ids: List[str],
        activity_ids: List[str],
        body_parts: List[str],
        activity_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            embeddings: (batch, embed_dim) エンコーダー出力
            dataset_ids: (batch,) データセット名のリスト
            activity_ids: (batch,) Activity名のリスト
            body_parts: (batch,) Body Part名のリスト
            activity_labels: (batch,) Activity ID（整数、省略時はactivity_idsから生成）

        Returns:
            (total_loss, {"activity": loss, "prototype": loss, ...})
        """
        batch_size = embeddings.size(0)
        device = embeddings.device

        # Activity labelsが与えられていない場合、activity_idsから生成
        if activity_labels is None:
            unique_activities = list(set(zip(dataset_ids, activity_ids)))
            activity_to_idx = {a: i for i, a in enumerate(unique_activities)}
            activity_labels = torch.tensor(
                [activity_to_idx[(d, a)] for d, a in zip(dataset_ids, activity_ids)],
                dtype=torch.long,
                device=device,
            )

        loss_dict = {}

        # 1. Activity-level Loss
        loss_activity = self.activity_loss(embeddings, activity_labels)
        loss_dict["activity"] = loss_activity

        # 2. Body Part別のPrototype Loss
        # Body Partごとにグループ化
        bp_groups = defaultdict(list)
        for i, bp in enumerate(body_parts):
            normalized_bp = self.atlas._normalize_body_part(bp)
            bp_groups[normalized_bp].append(i)

        prototype_losses = []
        for bp, indices in bp_groups.items():
            if bp not in self.body_parts:
                continue

            indices_tensor = torch.tensor(indices, device=device)
            bp_embeddings = embeddings[indices_tensor]
            bp_activity_labels = activity_labels[indices_tensor]

            # 候補Prototype IDを取得
            candidate_ids_list = []
            for idx in indices:
                candidates = self.get_candidate_prototype_ids(
                    dataset_ids[idx],
                    activity_ids[idx],
                    body_parts[idx],
                    device=device,
                )
                candidate_ids_list.append(candidates)

            candidate_ids = torch.stack(candidate_ids_list)  # (bp_batch, max_candidates)

            # Soft割り当て
            soft_assignments = self.prototypes.get_soft_assignments(
                bp_embeddings,
                bp,
                candidate_ids=candidate_ids,
                temperature=self.temperature,
            )

            # Prototype Loss
            loss = self.prototype_loss(
                bp_embeddings,
                soft_assignments,
                bp_activity_labels,
            )
            prototype_losses.append(loss)
            loss_dict[f"prototype_{bp}"] = loss

        # Prototype Lossの平均
        if prototype_losses:
            loss_prototype = torch.stack(prototype_losses).mean()
        else:
            # embeddingsの0倍を返して計算グラフを維持
            loss_prototype = (embeddings.sum() * 0.0)

        loss_dict["prototype"] = loss_prototype

        # Total Loss
        total_loss = (
            self.lambda_activity * loss_activity +
            self.lambda_prototype * loss_prototype
        )
        loss_dict["total"] = total_loss

        return total_loss, loss_dict


if __name__ == "__main__":
    # テスト
    print("Testing HierarchicalSSLLoss...")

    # ダミーデータ
    batch_size = 8
    embed_dim = 512

    embeddings = torch.randn(batch_size, embed_dim)
    dataset_ids = ["dsads"] * 4 + ["pamap2"] * 4
    activity_ids = ["sitting", "standing", "sitting", "standing"] * 2
    body_parts = ["wrist", "wrist", "chest", "chest"] * 2

    # Loss初期化
    loss_fn = HierarchicalSSLLoss(
        atlas_path="docs/atlas/activity_mapping.json",
        embed_dim=embed_dim,
    )

    # Forward
    total_loss, loss_dict = loss_fn(
        embeddings=embeddings,
        dataset_ids=dataset_ids,
        activity_ids=activity_ids,
        body_parts=body_parts,
    )

    print(f"Total Loss: {total_loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")

    print("\n✅ Test passed!")
