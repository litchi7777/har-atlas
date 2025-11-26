"""
Hierarchical SSL Loss for Motion Primitive Discovery

3階層のContrastive Learningを実装:
- L_atomic (重み大): 同じAtomic Motion（Prototype）→ positive（PiCOで推定、クロスデータセット）
- L_activity (重み中): 同じActivity + 同じデータセット → positive
- L_complex (重み小): 同じComplex Activity + 同じデータセット → positive

L_total = λ0 * L_complex + λ1 * L_activity + λ2 * L_atomic
λ0=0.1, λ1=0.3, λ2=0.6

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

            scores = scores.masked_fill(~mask, -1e4)  # Half型対応

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
        sim_matrix = sim_matrix.masked_fill(self_mask.bool(), -1e4)

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
    重み: 小（Complex Activityは内部に多様なSimple Activityを含むため）
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

        # Complex Activity (level=0) のみを抽出
        complex_indices = []
        complex_activities = []

        for i, act in enumerate(activity_ids):
            level = self.atlas.get_activity_level(act)
            if level == 0:  # Complex Activity
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
    Activity-level Contrastive Loss

    同じActivity内のサンプルをpositive（Level関係なく全Activity対象）
    データセット内のみを対象
    重み: 中
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
            Activity Contrastive Loss
        """
        device = embeddings.device

        # 全Activity対象（Levelフィルタなし）
        # Activity名をIDに変換
        unique_acts = list(set(activity_ids))
        act_to_id = {a: i for i, a in enumerate(unique_acts)}
        activity_labels = torch.tensor(
            [act_to_id[a] for a in activity_ids],
            dtype=torch.long,
            device=device,
        )

        return self.contrastive(embeddings, activity_labels, dataset_labels)


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
        sim_matrix = sim_matrix.masked_fill(self_mask.bool(), -1e4)

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
    Atomic Motion (Level 2) のContrastive Loss（PiCO）

    クロスデータセットで学習。Body Part別にPrototype（=Atomic Motion）を学習。

    核心的アイデア:
    - PiCOでサンプル → Atomic Motion（Prototype）への割り当てを推定
    - 同じAtomic Motionに割り当てられたサンプル同士をpositive
    - Activity間の類似度ではなく、サンプル単位の割り当て結果でpositive判定
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

    def forward(
        self,
        embeddings: torch.Tensor,
        body_parts: List[str],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        PiCOによるAtomic Motion Loss

        サンプル → Prototype（Atomic Motion）への割り当てを推定し、
        同じPrototypeに割り当てられたサンプル同士をpositiveとしてContrastive Learning

        Args:
            embeddings: (batch, embed_dim)
            body_parts: Body Part名のリスト

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

            # PiCO: サンプル → Prototype（Atomic Motion）へのSoft割り当て
            # candidate_ids=None で全Prototypeを候補とする
            soft_assignments = self.prototypes.get_soft_assignments(
                bp_embeddings,
                bp,
                candidate_ids=None,  # 全Prototypeが候補
                temperature=self.temperature,
            )

            # 同じPrototypeに割り当てられたサンプル同士がpositive
            # soft_assignments: (batch, num_prototypes)
            # positive_weights[i,j] = Σ_k soft_assignments[i,k] * soft_assignments[j,k]
            positive_weights = torch.mm(soft_assignments, soft_assignments.t())

            # 自分自身を除外
            batch_size = len(indices)
            self_mask = torch.eye(batch_size, device=device)
            positive_weights = positive_weights * (1 - self_mask)

            # Weighted Contrastive Loss
            bp_embeddings_norm = F.normalize(bp_embeddings, dim=1)
            sim_matrix = torch.mm(bp_embeddings_norm, bp_embeddings_norm.t()) / self.temperature

            # 重みがない場合はスキップ
            weight_sum = positive_weights.sum(dim=1)
            valid_samples = weight_sum > 0

            if not valid_samples.any():
                loss = bp_embeddings.sum() * 0.0
            else:
                # 数値安定化
                logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
                sim_matrix = sim_matrix - logits_max.detach()
                sim_matrix = sim_matrix.masked_fill(self_mask.bool(), -1e4)

                # Weighted InfoNCE Loss
                exp_sim = torch.exp(sim_matrix)
                log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

                weighted_log_prob = positive_weights * log_prob
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


class HierarchicalSSLLoss(nn.Module):
    """
    3階層SSL損失

    L_total = λ0 * L_complex + λ1 * L_activity + λ2 * L_atomic

    - L_atomic (重み大): 同じAtomic Motion（Prototype）→ positive（PiCOで推定、クロスデータセット）
    - L_activity (重み中): 同じActivity + 同じデータセット → positive
    - L_complex (重み小): 同じComplex Activity + 同じデータセット → positive

    デフォルト重み: λ0=0.1, λ1=0.3, λ2=0.6
    """

    def __init__(
        self,
        atlas_path: str,
        embed_dim: int = 512,
        prototype_dim: int = 128,
        temperature: float = 0.1,
        lambda_complex: float = 0.1,
        lambda_activity: float = 0.3,
        lambda_atomic: float = 0.6,
        # 後方互換性のため旧パラメータも受け付ける
        lambda_simple: float = None,
        lambda_prototype: float = None,
    ):
        """
        Args:
            atlas_path: activity_mapping.json へのパス
            embed_dim: エンコーダー出力次元
            prototype_dim: Prototype空間の次元
            temperature: Contrastive Loss温度
            lambda_complex: Complex Activity Loss重み (λ0) - 小
            lambda_activity: Activity Loss重み (λ1) - 中
            lambda_atomic: Atomic Motion Loss重み (λ2) - 大
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
        self.complex_loss_fn = ComplexActivityLoss(self.atlas, temperature=temperature)
        self.activity_loss_fn = SimpleActivityLoss(self.atlas, temperature=temperature)
        self.atomic_loss_fn = AtomicMotionLoss(self.atlas, self.prototypes, temperature=temperature)

        # 重み（後方互換性: 旧パラメータが指定されていたら変換）
        if lambda_simple is not None:
            # 旧形式: lambda_simple -> lambda_activity
            self.lambda_activity = lambda_simple
        else:
            self.lambda_activity = lambda_activity

        self.lambda_complex = lambda_complex
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

        # 1. Complex Activity Loss (L_complex) - 同じComplex Activity + 同じデータセット
        loss_complex = self.complex_loss_fn(embeddings, activity_ids, dataset_labels)
        loss_dict["complex"] = loss_complex

        # 2. Activity Loss (L_activity) - 同じActivity + 同じデータセット
        loss_activity = self.activity_loss_fn(embeddings, activity_ids, dataset_labels)
        loss_dict["activity"] = loss_activity

        # 3. Atomic Motion Loss (L_atomic) - PiCOで同じPrototype → positive
        loss_atomic, atomic_details = self.atomic_loss_fn(embeddings, body_parts)
        loss_dict["atomic"] = loss_atomic
        loss_dict.update(atomic_details)

        # Total Loss: L_total = λ0 * L_complex + λ1 * L_activity + λ2 * L_atomic
        total_loss = (
            self.lambda_complex * loss_complex +
            self.lambda_activity * loss_activity +
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

    print(f"Lambda values: complex={loss_fn.lambda_complex}, activity={loss_fn.lambda_activity}, atomic={loss_fn.lambda_atomic}")
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
    print(f"  L_activity (λ1={loss_fn.lambda_activity}): {loss_dict['activity'].item():.4f}")
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
