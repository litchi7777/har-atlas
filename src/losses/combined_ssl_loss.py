"""
Combined SSL Loss: MTL + Hierarchical Loss

MTLタスク（binary_*, invariant_*）と階層的Loss（L_complex, L_activity, L_atomic）を
同時に学習するための統合Loss関数

Usage:
    loss_fn = CombinedSSLLoss(
        ssl_tasks=["binary_permute", "binary_reverse", "binary_timewarp"],
        atlas_path="docs/atlas/activity_mapping.json",
        embed_dim=512,
    )

    # Forward
    total_loss, loss_dict = loss_fn(
        predictions=predictions,       # MTLタスクの予測
        labels=labels,                 # MTLタスクのラベル
        embeddings=embeddings,         # エンコーダー出力
        dataset_ids=dataset_ids,       # データセット名
        activity_ids=activity_ids,     # Activity名
        body_parts=body_parts,         # Body Part名
    )
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any

from .ssl_losses import IntegratedSSLLoss
from .hierarchical_loss import HierarchicalSSLLoss


class CombinedSSLLoss(nn.Module):
    """
    MTL + 階層的SSL Lossの統合

    L_total = λ_mtl * L_mtl + λ_hier * L_hierarchical

    L_mtl = Σ w_task * L_task  (binary_permute, binary_reverse, etc.)
    L_hierarchical = λ0 * L_complex + λ1 * L_activity + λ2 * L_atomic
    """

    def __init__(
        self,
        # MTL設定
        ssl_tasks: List[str],
        task_weights: Optional[Dict[str, float]] = None,
        # 階層的Loss設定
        atlas_path: str = "docs/atlas/activity_mapping.json",
        embed_dim: int = 512,
        prototype_dim: int = 128,
        temperature: float = 0.1,
        lambda_complex: float = 0.1,
        lambda_activity: float = 0.3,
        lambda_atomic: float = 0.6,
        # 統合重み
        lambda_mtl: float = 1.0,
        lambda_hierarchical: float = 1.0,
    ):
        """
        Args:
            ssl_tasks: SSLタスクのリスト (binary_*, invariant_*)
            task_weights: 各タスクの重み
            atlas_path: Activity Atlasへのパス
            embed_dim: エンコーダー出力次元
            prototype_dim: Prototype空間の次元
            temperature: Contrastive Loss温度
            lambda_complex: Complex Activity Loss重み
            lambda_activity: Activity Loss重み
            lambda_atomic: Atomic Motion Loss重み
            lambda_mtl: MTL全体の重み
            lambda_hierarchical: 階層的Loss全体の重み
        """
        super().__init__()

        # MTL Loss
        self.mtl_loss = IntegratedSSLLoss(
            ssl_tasks=ssl_tasks,
            task_weights=task_weights,
        )
        self.ssl_tasks = ssl_tasks

        # Hierarchical Loss
        self.hierarchical_loss = HierarchicalSSLLoss(
            atlas_path=atlas_path,
            embed_dim=embed_dim,
            prototype_dim=prototype_dim,
            temperature=temperature,
            lambda_complex=lambda_complex,
            lambda_activity=lambda_activity,
            lambda_atomic=lambda_atomic,
        )

        # 統合重み
        self.lambda_mtl = lambda_mtl
        self.lambda_hierarchical = lambda_hierarchical

    def forward(
        self,
        # MTL用
        predictions: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        # 階層的Loss用
        embeddings: Optional[torch.Tensor] = None,
        dataset_ids: Optional[List[str]] = None,
        activity_ids: Optional[List[str]] = None,
        body_parts: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            predictions: MTLタスクの予測 {task_name: prediction}
            labels: MTLタスクのラベル {task_name: label}
            embeddings: (batch, embed_dim) エンコーダー出力
            dataset_ids: データセット名のリスト
            activity_ids: Activity名のリスト
            body_parts: Body Part名のリスト

        Returns:
            (total_loss, loss_dict)
        """
        loss_dict = {}
        total_loss = 0.0
        device = None

        # MTL Loss計算
        if predictions is not None and labels is not None:
            mtl_loss, mtl_task_losses = self.mtl_loss(predictions, labels)
            loss_dict["mtl_total"] = mtl_loss
            for task, task_loss in mtl_task_losses.items():
                loss_dict[f"mtl_{task}"] = task_loss
            total_loss = total_loss + self.lambda_mtl * mtl_loss
            device = mtl_loss.device

        # Hierarchical Loss計算
        if (embeddings is not None and dataset_ids is not None and
            activity_ids is not None and body_parts is not None):
            hier_loss, hier_loss_dict = self.hierarchical_loss(
                embeddings=embeddings,
                dataset_ids=dataset_ids,
                activity_ids=activity_ids,
                body_parts=body_parts,
            )
            loss_dict["hier_total"] = hier_loss
            loss_dict["hier_complex"] = hier_loss_dict["complex"]
            loss_dict["hier_activity"] = hier_loss_dict["activity"]
            loss_dict["hier_atomic"] = hier_loss_dict["atomic"]
            total_loss = total_loss + self.lambda_hierarchical * hier_loss
            device = hier_loss.device

        # どちらも計算されていない場合
        if device is None:
            raise ValueError("Either MTL or Hierarchical inputs must be provided")

        # Tensorでない場合（両方0.0のとき）に対応
        if not isinstance(total_loss, torch.Tensor):
            total_loss = torch.tensor(total_loss, device=device, requires_grad=True)

        loss_dict["total"] = total_loss
        return total_loss, loss_dict

    def get_prototype_state_dict(self) -> Dict[str, Any]:
        """Prototypeの状態を取得（チェックポイント用）"""
        return self.hierarchical_loss.prototypes.state_dict()

    def load_prototype_state_dict(self, state_dict: Dict[str, Any]):
        """Prototypeの状態をロード"""
        self.hierarchical_loss.prototypes.load_state_dict(state_dict)


if __name__ == "__main__":
    # テスト
    print("Testing CombinedSSLLoss...")

    batch_size = 16
    embed_dim = 512

    # MTL用ダミーデータ
    ssl_tasks = ["binary_permute", "binary_reverse", "binary_timewarp"]
    predictions = {
        "binary_permute": torch.randn(batch_size, 2),
        "binary_reverse": torch.randn(batch_size, 2),
        "binary_timewarp": torch.randn(batch_size, 2),
    }
    labels = {
        "binary_permute": torch.randint(0, 2, (batch_size,)),
        "binary_reverse": torch.randint(0, 2, (batch_size,)),
        "binary_timewarp": torch.randint(0, 2, (batch_size,)),
    }

    # 階層的Loss用ダミーデータ
    embeddings = torch.randn(batch_size, embed_dim, requires_grad=True)
    dataset_ids = ["dsads"] * 8 + ["pamap2"] * 8
    activity_ids = ["walking"] * 4 + ["running"] * 4 + ["walking"] * 4 + ["cycling"] * 4
    body_parts = ["wrist"] * 8 + ["hip"] * 8

    # Loss初期化
    loss_fn = CombinedSSLLoss(
        ssl_tasks=ssl_tasks,
        embed_dim=embed_dim,
        lambda_mtl=1.0,
        lambda_hierarchical=0.5,
    )

    # Forward
    total_loss, loss_dict = loss_fn(
        predictions=predictions,
        labels=labels,
        embeddings=embeddings,
        dataset_ids=dataset_ids,
        activity_ids=activity_ids,
        body_parts=body_parts,
    )

    print(f"\nTotal Loss: {total_loss.item():.4f}")
    print("\nLoss Breakdown:")
    for k, v in sorted(loss_dict.items()):
        if not k.startswith("hier_atomic_") or k == "hier_atomic":
            print(f"  {k}: {v.item():.4f}")

    # 勾配計算テスト
    total_loss.backward()
    print(f"\n✅ Gradient computation passed!")
    print("✅ CombinedSSLLoss test passed!")
