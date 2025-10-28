"""
バックボーンモデル（エンコーダー）

画像ベースのHAR用のバックボーンアーキテクチャを提供します。
"""

from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models


def get_backbone(name: str, pretrained: bool = False) -> Tuple[nn.Module, int]:
    """
    バックボーンモデルを取得

    Args:
        name: バックボーン名 ("resnet18", "resnet50", "mobilenet_v2", etc.)
        pretrained: ImageNet事前学習済み重みを使用するか

    Returns:
        (backbone, feature_dim): バックボーンモデルと特徴次元のタプル

    Raises:
        ValueError: 無効なバックボーン名の場合
    """
    if name == "resnet18":
        backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()  # Remove classification head
        return backbone, feature_dim

    elif name == "resnet50":
        backbone = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feature_dim

    elif name == "resnet34":
        backbone = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feature_dim

    elif name == "resnet101":
        backbone = models.resnet101(weights="IMAGENET1K_V1" if pretrained else None)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feature_dim

    elif name == "mobilenet_v2":
        backbone = models.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        feature_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        return backbone, feature_dim

    elif name == "efficientnet_b0":
        backbone = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        feature_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        return backbone, feature_dim

    elif name == "vit_b_16":
        backbone = models.vit_b_16(weights="IMAGENET1K_V1" if pretrained else None)
        feature_dim = backbone.heads.head.in_features
        backbone.heads = nn.Identity()
        return backbone, feature_dim

    else:
        raise ValueError(
            f"Unknown backbone: {name}. "
            f"Available: resnet18, resnet34, resnet50, resnet101, "
            f"mobilenet_v2, efficientnet_b0, vit_b_16"
        )


def freeze_backbone(backbone: nn.Module) -> None:
    """
    バックボーンの重みを凍結

    Args:
        backbone: 凍結するバックボーンモデル
    """
    for param in backbone.parameters():
        param.requires_grad = False


def unfreeze_backbone(backbone: nn.Module) -> None:
    """
    バックボーンの重みを凍結解除

    Args:
        backbone: 凍結解除するバックボーンモデル
    """
    for param in backbone.parameters():
        param.requires_grad = True


# ============================================================================
# Tests
# ============================================================================

__test__ = {
    "get_backbone": """
    >>> import torch
    >>> backbone, feature_dim = get_backbone('resnet18')
    >>> backbone is not None
    True
    >>> feature_dim
    512
    >>> # Test forward pass
    >>> x = torch.randn(2, 3, 224, 224)
    >>> out = backbone(x)
    >>> out.shape[0]
    2
    >>> out.shape[1]
    512
    """,
    "get_backbone_resnet50": """
    >>> backbone, feature_dim = get_backbone('resnet50')
    >>> feature_dim
    2048
    """,
    "get_backbone_invalid": """
    >>> try:
    ...     get_backbone('invalid_backbone')
    ... except ValueError as e:
    ...     'Unknown backbone' in str(e)
    True
    """,
    "freeze_backbone": """
    >>> import torch
    >>> backbone, _ = get_backbone('resnet18')
    >>> freeze_backbone(backbone)
    >>> all(not p.requires_grad for p in backbone.parameters())
    True
    """,
    "unfreeze_backbone": """
    >>> import torch
    >>> backbone, _ = get_backbone('resnet18')
    >>> freeze_backbone(backbone)
    >>> unfreeze_backbone(backbone)
    >>> all(p.requires_grad for p in backbone.parameters())
    True
    """,
}
