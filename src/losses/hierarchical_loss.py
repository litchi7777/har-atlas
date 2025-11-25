"""
Hierarchical SSL Loss for Motion Primitive Discovery

3階層のContrastive Learningを実装:
- L_complex: Complex Activity (level=0) 内のContrastive Loss（データセット内）
- L_simple: Simple Activity (level=1) 内のContrastive Loss（データセット内）
- L_atomic: Body Part別Prototype学習（クロスデータセット、Atomic共有でsoft positive）

L_total = λ0 * L_complex + λ1 * L_simple + λ2 * L_atomic
λ0=0.1, λ1=0.3, λ2=0.6（階層が深いほど重視）

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
    データセット内のサンプルのみを対象とする
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        activity_labels: torch.Tensor,
        dataset_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, embed_dim) 正規化済み埋め込み
            activity_labels: (batch,) Activity ID
            dataset_labels: (batch,) Dataset ID（Noneなら全てデータセット内として扱う）

        Returns:
            Contrastive Loss
        """
        batch_size = embeddings.size(0)
        device = embeddings.device

        if batch_size < 2:
            return (embeddings.sum() * 0.0)

        # 正規化
        embeddings = F.normalize(embeddings, dim=1)

        # 類似度行列
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature

        # 同じActivityかどうかのマスク
        labels = activity_labels.view(-1, 1)
        positive_mask = (labels == labels.t()).float()

        # データセット内のみを対象（cross-datasetは除外）
        if dataset_labels is not None:
            ds_labels = dataset_labels.view(-1, 1)
            same_dataset = (ds_labels == ds_labels.t()).float()
            positive_mask = positive_mask * same_dataset

        # 自分自身を除外
        self_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask * (1 - self_mask)

        # 正例がない場合はスキップ
        num_positives = positive_mask.sum(dim=1)
        valid_samples = num_positives > 0

        if not valid_samples.any():
            return (embeddings.sum() * 0.0)

        # 対角成分を大きな負値でマスク
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()
        sim_matrix = sim_matrix.masked_fill(self_mask.bool(), -1e9)

        # InfoNCE Loss
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # 正例の平均
        loss = -(positive_mask * log_prob).sum(dim=1) / (num_positives + 1e-8)
        loss = loss[valid_samples].mean()

        return loss


class ComplexActivityLoss(nn.Module):
    """
    Complex Activity (level=0) のContrastive Loss

    同じComplex Activity内のサンプルをpositive
    データセット内のみを対象
    """

    def __init__(self, atlas: "AtlasLoader", temperature: float = 0.1):
        super().__init__()
        self.atlas = atlas
        self.temperature = temperature
        self.contrastive = ActivityContrastiveLoss(temperature)

    def forward(
        self,
        embeddings: torch.Tensor,
        activity_ids: List[str],
        dataset_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, embed_dim)
            activity_ids: Activity名のリスト
            dataset_labels: (batch,) Dataset ID

        Returns:
            Complex Activity Contrastive Loss
        """
        device = embeddings.device
        batch_size = len(activity_ids)

        # Complex Activityのみを抽出
        complex_indices = []
        complex_activities = []

        for i, act in enumerate(activity_ids):
            level = self.atlas.get_activity_level(act)
            if level == 0:  # Complex
                complex_indices.append(i)
                complex_activities.append(act)

        if len(complex_indices) < 2:
            return embeddings.sum() * 0.0

        # Complex Activityのみのサブセット
        indices_tensor = torch.tensor(complex_indices, device=device)
        complex_embeddings = embeddings[indices_tensor]
        complex_ds_labels = dataset_labels[indices_tensor]

        # Activity名をIDに変換
        unique_acts = list(set(complex_activities))
        act_to_id = {a: i for i, a in enumerate(unique_acts)}
        complex_act_labels = torch.tensor(
            [act_to_id[a] for a in complex_activities],
            dtype=torch.long,
            device=device,
        )

        return self.contrastive(complex_embeddings, complex_act_labels, complex_ds_labels)


class SimpleActivityLoss(nn.Module):
    """
    Simple Activity (level=1) のContrastive Loss

    同じSimple Activity内のサンプルをpositive
    データセット内のみを対象
    """

    def __init__(self, atlas: "AtlasLoader", temperature: float = 0.1):
        super().__init__()
        self.atlas = atlas
        self.temperature = temperature
        self.contrastive = ActivityContrastiveLoss(temperature)

    def forward(
        self,
        embeddings: torch.Tensor,
        activity_ids: List[str],
        dataset_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, embed_dim)
            activity_ids: Activity名のリスト
            dataset_labels: (batch,) Dataset ID

        Returns:
            Simple Activity Contrastive Loss
        """
        device = embeddings.device
        batch_size = len(activity_ids)

        # Simple Activityのみを抽出
        simple_indices = []
        simple_activities = []

        for i, act in enumerate(activity_ids):
            level = self.atlas.get_activity_level(act)
            if level == 1:  # Simple
                simple_indices.append(i)
                simple_activities.append(act)

        if len(simple_indices) < 2:
            return embeddings.sum() * 0.0

        # Simple Activityのみのサブセット
        indices_tensor = torch.tensor(simple_indices, device=device)
        simple_embeddings = embeddings[indices_tensor]
        simple_ds_labels = dataset_labels[indices_tensor]

        # Activity名をIDに変換
        unique_acts = list(set(simple_activities))
        act_to_id = {a: i for i, a in enumerate(unique_acts)}
        simple_act_labels = torch.tensor(
            [act_to_id[a] for a in simple_activities],
            dtype=torch.long,
            device=device,
        )

        return self.contrastive(simple_embeddings, simple_act_labels, simple_ds_labels)


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
            return (embeddings.sum() * 0.0)

        # 正規化
        embeddings = F.normalize(embeddings, dim=1)

        # 類似度行列
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature

        # Soft positive重み（Prototype割り当ての類似度）
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
            return (embeddings.sum() * 0.0)

        # 対角成分を大きな負値でマスク
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


class AtomicMotionLoss(nn.Module):
    """
    Atomic Motion (Level 2) のContrastive Loss

    クロスデータセットで学習。Atomic Motion共有度によるsoft positive重みを使用。
    Body Part別にPrototypeを学習（PiCO）。

    核心的アイデア:
    - 同じAtomic Motionを持つActivity同士をsoft positiveとして扱う
    - walkingとwalking_treadmillは同じAtomic → 強いpositive
    - walkingとrunningは一部共有 → 弱いpositive
    """

    def __init__(
        self,
        atlas: "AtlasLoader",
        prototypes: "BodyPartPrototypes",
        temperature: float = 0.1,
    ):
        super().__init__()
        self.atlas = atlas
        self.prototypes = prototypes
        self.temperature = temperature
        self.prototype_loss = PrototypeContrastiveLoss(temperature)

        # Atomic Motion → ID マッピング
        self.atomic_to_id = atlas.get_atomic_motion_to_id()

    def _compute_atomic_sharing_weights(
        self,
        activity_ids: List[str],
        body_part: str,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Activity間のAtomic共有度行列を計算

        Returns:
            (batch, batch) のsoft positive重み行列
        """
        batch_size = len(activity_ids)
        weights = torch.zeros(batch_size, batch_size, device=device)

        for i in range(batch_size):
            for j in range(batch_size):
                if i == j:
                    weights[i, j] = 1.0
                else:
                    weights[i, j] = self.atlas.get_atomic_sharing_weight(
                        activity_ids[i], activity_ids[j], body_part
                    )

        return weights

    def forward(
        self,
        embeddings: torch.Tensor,
        activity_ids: List[str],
        body_parts: List[str],
        dataset_ids: List[str],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            embeddings: (batch, embed_dim)
            activity_ids: Activity名のリスト
            body_parts: Body Part名のリスト
            dataset_ids: データセット名のリスト

        Returns:
            (total_loss, {"atomic_wrist": loss, "atomic_hip": loss, ...})
        """
        device = embeddings.device
        loss_dict = {}
        losses = []

        # Body Partごとにグループ化
        bp_groups = defaultdict(list)
        for i, bp in enumerate(body_parts):
            normalized_bp = self.atlas._normalize_body_part(bp)
            if normalized_bp in self.prototypes.body_parts:
                bp_groups[normalized_bp].append(i)

        for bp, indices in bp_groups.items():
            if len(indices) < 2:
                continue

            indices_tensor = torch.tensor(indices, device=device)
            bp_embeddings = embeddings[indices_tensor]
            bp_activities = [activity_ids[i] for i in indices]
            bp_datasets = [dataset_ids[i] for i in indices]

            # Atomic共有度に基づくsoft positive重み
            atomic_weights = self._compute_atomic_sharing_weights(
                bp_activities, bp, device
            )

            # 候補Prototype IDを取得（各サンプル）
            candidate_ids_list = []
            has_any_candidates = False
            for idx in indices:
                candidates = self._get_candidate_prototype_ids(
                    dataset_ids[idx],
                    activity_ids[idx],
                    body_parts[idx],
                    device=device,
                )
                candidate_ids_list.append(candidates)
                if candidates is not None:
                    has_any_candidates = True

            # 全サンプルの候補がNoneの場合、candidate_ids=Noneで全Prototypeを候補に
            if has_any_candidates:
                # 有効な候補がある場合のみスタック
                # None要素は最大候補数で埋める
                max_cands = 10
                filled_list = []
                for cand in candidate_ids_list:
                    if cand is None:
                        # 全Prototypeを候補とするため-1で埋める
                        # ただし実際には全Prototypeが候補になるよう処理が必要
                        filled_list.append(torch.tensor([-1] * max_cands, dtype=torch.long, device=device))
                    else:
                        filled_list.append(cand)
                candidate_ids = torch.stack(filled_list)
            else:
                candidate_ids = None

            # Soft割り当て
            soft_assignments = self.prototypes.get_soft_assignments(
                bp_embeddings,
                bp,
                candidate_ids=candidate_ids,
                temperature=self.temperature,
            )

            # Prototype類似度とAtomic共有度を組み合わせたsoft positive
            prototype_weights = torch.mm(soft_assignments, soft_assignments.t())
            combined_weights = torch.max(prototype_weights, atomic_weights)

            # 自分自身を除外
            batch_size = len(indices)
            self_mask = torch.eye(batch_size, device=device)
            combined_weights = combined_weights * (1 - self_mask)

            # Weighted Contrastive Loss
            bp_embeddings_norm = F.normalize(bp_embeddings, dim=1)
            sim_matrix = torch.mm(bp_embeddings_norm, bp_embeddings_norm.t()) / self.temperature

            # 重みがない場合はスキップ
            weight_sum = combined_weights.sum(dim=1)
            valid_samples = weight_sum > 0

            if not valid_samples.any():
                loss = bp_embeddings.sum() * 0.0
            else:
                # 数値安定化
                logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
                sim_matrix = sim_matrix - logits_max.detach()
                sim_matrix = sim_matrix.masked_fill(self_mask.bool(), -1e9)

                # Weighted InfoNCE Loss
                exp_sim = torch.exp(sim_matrix)
                log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

                weighted_log_prob = combined_weights * log_prob
                loss = -weighted_log_prob.sum(dim=1) / (weight_sum + 1e-8)
                loss = loss[valid_samples].mean()

            losses.append(loss)
            loss_dict[f"atomic_{bp}"] = loss

        if losses:
            total_loss = torch.stack(losses).mean()
        else:
            total_loss = embeddings.sum() * 0.0

        loss_dict["atomic_total"] = total_loss
        return total_loss, loss_dict

    def _get_candidate_prototype_ids(
        self,
        dataset: str,
        activity: str,
        body_part: str,
        max_candidates: int = 10,
        device: torch.device = None,
    ) -> Optional[torch.Tensor]:
        """
        候補Prototype IDを取得

        atomic_motions.jsonから直接取得（データセット非依存）。
        候補が見つからない場合はNoneを返す（全Prototypeを候補とする）。
        """
        # Body Part正規化
        normalized_bp = self.atlas._normalize_body_part(body_part)

        # atomic_motions.jsonから直接取得
        atomics = self.atlas.get_activity_atomic_signature(activity, normalized_bp)

        if not atomics:
            # Atlasに登録されていないActivityの場合、Noneを返す
            # → get_soft_assignmentsでcandidate_ids=Noneなら全Prototypeが候補
            return None

        # Atomic Motion名をIDに変換
        if normalized_bp not in self.atomic_to_id:
            return None

        bp_mapping = self.atomic_to_id[normalized_bp]
        candidates = [bp_mapping[a] for a in atomics if a in bp_mapping]

        if not candidates:
            return None

        if len(candidates) < max_candidates:
            candidates = candidates + [-1] * (max_candidates - len(candidates))
        else:
            candidates = candidates[:max_candidates]

        tensor = torch.tensor(candidates, dtype=torch.long)
        if device is not None:
            tensor = tensor.to(device)
        return tensor


class HierarchicalSSLLoss(nn.Module):
    """
    3階層SSL損失

    L_total = λ0 * L_complex + λ1 * L_simple + λ2 * L_atomic

    - L_complex: Complex Activity (level=0) 内のContrastive Loss（データセット内）
    - L_simple: Simple Activity (level=1) 内のContrastive Loss（データセット内）
    - L_atomic: Body Part別Prototype学習（クロスデータセット、Atomic共有でsoft positive）

    デフォルト重み: λ0=0.1, λ1=0.3, λ2=0.6（階層が深いほど重視）
    """

    def __init__(
        self,
        atlas_path: str,
        embed_dim: int = 512,
        prototype_dim: int = 128,
        temperature: float = 0.1,
        lambda_complex: float = 0.1,
        lambda_simple: float = 0.3,
        lambda_atomic: float = 0.6,
        # 後方互換性のため旧パラメータも受け付ける
        lambda_activity: float = None,
        lambda_prototype: float = None,
    ):
        """
        Args:
            atlas_path: activity_mapping.json へのパス
            embed_dim: エンコーダー出力次元
            prototype_dim: Prototype空間の次元
            temperature: Contrastive Loss温度
            lambda_complex: Complex Activity Loss重み (λ0)
            lambda_simple: Simple Activity Loss重み (λ1)
            lambda_atomic: Atomic Motion Loss重み (λ2)
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

        # 3階層Loss関数
        self.complex_loss = ComplexActivityLoss(self.atlas, temperature=temperature)
        self.simple_loss = SimpleActivityLoss(self.atlas, temperature=temperature)
        self.atomic_loss = AtomicMotionLoss(self.atlas, self.prototypes, temperature=temperature)

        # 重み（後方互換性: 旧パラメータが指定されていたら変換）
        if lambda_activity is not None and lambda_prototype is not None:
            # 旧形式: activity -> complex+simple, prototype -> atomic
            self.lambda_complex = lambda_activity * 0.25
            self.lambda_simple = lambda_activity * 0.75
            self.lambda_atomic = lambda_prototype
        else:
            self.lambda_complex = lambda_complex
            self.lambda_simple = lambda_simple
            self.lambda_atomic = lambda_atomic

        self.temperature = temperature

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
            (total_loss, {"complex": loss, "simple": loss, "atomic": loss, ...})
        """
        batch_size = embeddings.size(0)
        device = embeddings.device

        # Dataset IDを整数に変換
        unique_datasets = list(set(dataset_ids))
        dataset_to_idx = {d: i for i, d in enumerate(unique_datasets)}
        dataset_labels = torch.tensor(
            [dataset_to_idx[d] for d in dataset_ids],
            dtype=torch.long,
            device=device,
        )

        loss_dict = {}

        # 1. Complex Activity Loss (L_complex) - データセット内
        loss_complex = self.complex_loss(embeddings, activity_ids, dataset_labels)
        loss_dict["complex"] = loss_complex

        # 2. Simple Activity Loss (L_simple) - データセット内
        loss_simple = self.simple_loss(embeddings, activity_ids, dataset_labels)
        loss_dict["simple"] = loss_simple

        # 3. Atomic Motion Loss (L_atomic) - クロスデータセット
        loss_atomic, atomic_details = self.atomic_loss(
            embeddings, activity_ids, body_parts, dataset_ids
        )
        loss_dict["atomic"] = loss_atomic
        loss_dict.update(atomic_details)

        # Total Loss: L_total = λ0 * L_complex + λ1 * L_simple + λ2 * L_atomic
        total_loss = (
            self.lambda_complex * loss_complex +
            self.lambda_simple * loss_simple +
            self.lambda_atomic * loss_atomic
        )
        loss_dict["total"] = total_loss

        return total_loss, loss_dict

    # 後方互換性のためのメソッド（旧API）
    def get_candidate_prototype_ids(
        self,
        dataset: str,
        activity: str,
        body_part: str,
        max_candidates: int = 10,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Activity + Body Partに対する候補Prototype IDを取得（後方互換性）"""
        try:
            candidates = self.atlas.get_candidate_atomic_ids(
                dataset, activity, body_part, self.atomic_to_id
            )
        except KeyError:
            candidates = []

        if len(candidates) < max_candidates:
            candidates = candidates + [-1] * (max_candidates - len(candidates))
        else:
            candidates = candidates[:max_candidates]

        tensor = torch.tensor(candidates, dtype=torch.long)
        if device is not None:
            tensor = tensor.to(device)
        return tensor


if __name__ == "__main__":
    # テスト
    print("Testing HierarchicalSSLLoss with 3-level hierarchy...")

    # ダミーデータ（Complex + Simple Activityを混ぜる）
    # 同じActivity、同じBody Partのサンプルを複数含める
    batch_size = 24
    embed_dim = 512

    embeddings = torch.randn(batch_size, embed_dim, requires_grad=True)

    # Complex Activities: vacuum_cleaning, cooking（同じActivityを複数含める）
    # Simple Activities: walking, running（同じActivityを複数含める）
    dataset_ids = ["dsads"] * 12 + ["pamap2"] * 12
    activity_ids = [
        # DSADS: Complex（vacuum_cleaning x3, cooking x3）+ Simple（walking x3, running x3）
        "vacuum_cleaning", "vacuum_cleaning", "vacuum_cleaning",
        "cooking", "cooking", "cooking",
        "walking", "walking", "walking",
        "running", "running", "running",
        # PAMAP2: 同様
        "vacuum_cleaning", "vacuum_cleaning", "vacuum_cleaning",
        "ironing", "ironing", "ironing",
        "walking", "walking", "walking",
        "cycling", "cycling", "cycling",
    ]
    # 同じBody Partのサンプルを複数含める
    body_parts = ["wrist"] * 6 + ["hip"] * 6 + ["wrist"] * 6 + ["hip"] * 6

    # Loss初期化
    loss_fn = HierarchicalSSLLoss(
        atlas_path="docs/atlas/activity_mapping.json",
        embed_dim=embed_dim,
    )

    print(f"Lambda values: complex={loss_fn.lambda_complex}, simple={loss_fn.lambda_simple}, atomic={loss_fn.lambda_atomic}")
    print(f"Body parts in batch: {set(body_parts)}")

    # Forward
    total_loss, loss_dict = loss_fn(
        embeddings=embeddings,
        dataset_ids=dataset_ids,
        activity_ids=activity_ids,
        body_parts=body_parts,
    )

    print(f"\nTotal Loss: {total_loss.item():.4f}")
    print("\n3-Level Loss Breakdown:")
    print(f"  L_complex (λ0={loss_fn.lambda_complex}): {loss_dict['complex'].item():.4f}")
    print(f"  L_simple (λ1={loss_fn.lambda_simple}): {loss_dict['simple'].item():.4f}")
    print(f"  L_atomic (λ2={loss_fn.lambda_atomic}): {loss_dict['atomic'].item():.4f}")

    print("\nAtomic Loss Details (per body part):")
    for k, v in sorted(loss_dict.items()):
        if k.startswith("atomic_") and k != "atomic_total":
            print(f"  {k}: {v.item():.4f}")

    # 勾配計算テスト
    total_loss.backward()
    grad_norm = embeddings.grad.norm().item()
    print(f"\n✅ Gradient computation passed! (grad norm: {grad_norm:.4f})")
    print("✅ 3-level hierarchical loss test passed!")
