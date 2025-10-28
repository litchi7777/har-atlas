"""
画像ベースのHARモデル

Self-Supervised Learning用のSSLモデルと、分類用のClassificationModelを提供します。
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .backbones import get_backbone, freeze_backbone


class SSLModel(nn.Module):
    """
    Self-Supervised Learning用のモデル

    エンコーダー（バックボーン）+ プロジェクションヘッド
    """

    def __init__(self, config: dict):
        """
        Args:
            config: モデル設定辞書
                - backbone: バックボーン名 (e.g., "resnet18")
                - feature_dim: 中間特徴次元
                - projection_dim: プロジェクション後の次元
                - pretrained: ImageNet事前学習済み重みを使用するか (default: False)
        """
        super().__init__()

        backbone_name = config["backbone"]
        pretrained = config.get("pretrained", False)

        # バックボーン（エンコーダー）
        self.encoder, encoder_dim = get_backbone(backbone_name, pretrained=pretrained)

        # プロジェクションヘッド
        feature_dim = config.get("feature_dim", 128)
        projection_dim = config.get("projection_dim", 64)

        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
        )

    def forward(
        self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        順伝播

        Args:
            x1: 第1ビュー (batch_size, 3, H, W)
            x2: 第2ビュー (optional)

        Returns:
            (z1, z2): プロジェクション後の特徴表現
        """
        # エンコード
        h1 = self.encoder(x1)

        # グローバルプーリング（必要な場合）
        if len(h1.shape) > 2:
            h1 = torch.mean(h1, dim=[-1, -2])

        # プロジェクション
        z1 = self.projection(h1)

        if x2 is not None:
            h2 = self.encoder(x2)
            if len(h2.shape) > 2:
                h2 = torch.mean(h2, dim=[-1, -2])
            z2 = self.projection(h2)
            return z1, z2

        return z1, None


class ClassificationModel(nn.Module):
    """
    分類用のモデル

    事前学習済みエンコーダー + 分類ヘッド
    """

    def __init__(self, config: dict):
        """
        Args:
            config: モデル設定辞書
                - backbone: バックボーン名
                - num_classes: クラス数
                - pretrained: ImageNet事前学習済み重みを使用するか (default: False)
                - pretrained_path: SSL事前学習済みモデルのパス (optional)
                - freeze_backbone: バックボーンを凍結するか (default: False)
                - dropout: ドロップアウト率 (default: 0.0)
        """
        super().__init__()

        backbone_name = config["backbone"]
        num_classes = config["num_classes"]
        pretrained = config.get("pretrained", False)

        # バックボーン（エンコーダー）
        self.encoder, encoder_dim = get_backbone(backbone_name, pretrained=pretrained)

        # SSL事前学習済み重みをロード（指定されている場合）
        pretrained_path = config.get("pretrained_path")
        if pretrained_path:
            self.load_pretrained_encoder(pretrained_path)

        # バックボーンを凍結（指定されている場合）
        if config.get("freeze_backbone", False):
            freeze_backbone(self.encoder)

        # ドロップアウト
        dropout_rate = config.get("dropout", 0.0)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        # 分類ヘッド
        self.fc = nn.Linear(encoder_dim, num_classes)

    def load_pretrained_encoder(
        self, pretrained_path: str, device: Optional[torch.device] = None
    ) -> None:
        """
        SSL事前学習済みのエンコーダー重みをロード

        Args:
            pretrained_path: チェックポイントファイルのパス
            device: デバイス (optional)

        Raises:
            FileNotFoundError: チェックポイントファイルが存在しない場合
            RuntimeError: 重みのロードに失敗した場合
        """
        pretrained_path = Path(pretrained_path)
        if not pretrained_path.exists():
            raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")

        # チェックポイントをロード
        checkpoint = torch.load(pretrained_path, map_location=device or "cpu")

        # state_dictを取得
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # encoder.* の重みのみを抽出
        encoder_state = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                new_key = key.replace("encoder.", "")
                encoder_state[new_key] = value

        # エンコーダーにロード
        missing_keys, unexpected_keys = self.encoder.load_state_dict(encoder_state, strict=False)

        if missing_keys:
            print(f"Warning: Missing keys in encoder: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in encoder: {unexpected_keys}")

        print(f"Loaded pretrained encoder from {pretrained_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播

        Args:
            x: 入力画像 (batch_size, 3, H, W)

        Returns:
            logits: クラスロジット (batch_size, num_classes)
        """
        # エンコード
        h = self.encoder(x)

        # グローバルプーリング（必要な場合）
        if len(h.shape) > 2:
            h = torch.mean(h, dim=[-1, -2])

        # ドロップアウト
        if self.dropout is not None:
            h = self.dropout(h)

        # 分類
        logits = self.fc(h)

        return logits
