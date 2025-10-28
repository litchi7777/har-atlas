"""
トレーニング用ユーティリティ関数

オプティマイザー、スケジューラー、ロギングの設定等
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


logger = logging.getLogger(__name__)


def get_optimizer(
    model: nn.Module, optimizer_name: str, learning_rate: float, weight_decay: float = 0.0, **kwargs
) -> torch.optim.Optimizer:
    """
    オプティマイザーを作成

    Args:
        model: 最適化するモデル
        optimizer_name: オプティマイザー名 ('adam', 'sgd', 'adamw')
        learning_rate: 学習率
        weight_decay: 重み減衰
        **kwargs: オプティマイザー固有のパラメータ

    Returns:
        オプティマイザーインスタンス

    Raises:
        ValueError: サポートされていないオプティマイザーの場合
    """
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=kwargs.get("betas", (0.9, 0.999)),
        )

    elif optimizer_name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=kwargs.get("momentum", 0.9),
            weight_decay=weight_decay,
            nesterov=kwargs.get("nesterov", False),
        )

    elif optimizer_name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=kwargs.get("betas", (0.9, 0.999)),
        )

    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}. " f"Supported: adam, sgd, adamw"
        )


def get_scheduler(
    optimizer: torch.optim.Optimizer, scheduler_name: str, **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    学習率スケジューラーを作成

    Args:
        optimizer: オプティマイザー
        scheduler_name: スケジューラー名 ('step', 'cosine', 'plateau', 'exponential')
        **kwargs: スケジューラー固有のパラメータ

    Returns:
        スケジューラーインスタンス（Noneの場合はスケジューリングなし）

    Raises:
        ValueError: サポートされていないスケジューラーの場合
    """
    if scheduler_name is None or scheduler_name.lower() == "none":
        return None

    scheduler_name = scheduler_name.lower()

    if scheduler_name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=kwargs.get("step_size", 30), gamma=kwargs.get("gamma", 0.1)
        )

    elif scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=kwargs.get("T_max", 100), eta_min=kwargs.get("eta_min", 0)
        )

    elif scheduler_name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get("mode", "min"),
            factor=kwargs.get("factor", 0.1),
            patience=kwargs.get("patience", 10),
        )

    elif scheduler_name == "exponential":
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=kwargs.get("gamma", 0.95))

    else:
        raise ValueError(
            f"Unsupported scheduler: {scheduler_name}. "
            f"Supported: step, cosine, plateau, exponential"
        )


def init_wandb(config: Dict[str, Any], model: nn.Module) -> bool:
    """
    Weights & Biases を初期化

    Args:
        config: 設定辞書
        model: 追跡するモデル

    Returns:
        W&Bが有効化されたかどうか
    """
    wandb_config = config.get("wandb", {})

    if not wandb_config.get("enabled", False):
        return False

    if not WANDB_AVAILABLE:
        logger.warning(
            "W&B is enabled in config but wandb is not installed. "
            "Install with: pip install wandb"
        )
        return False

    try:
        wandb.init(
            project=wandb_config.get("project", "har-foundation"),
            entity=wandb_config.get("entity"),
            name=wandb_config.get("name"),
            tags=wandb_config.get("tags", []),
            notes=wandb_config.get("notes"),
            config=config,
        )

        # モデルを監視
        wandb.watch(model, log="all", log_freq=100)

        logger.info("Weights & Biases initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize W&B: {e}")
        return False


class AverageMeter:
    """
    メトリクスの移動平均を計算

    学習中の損失やメトリクスを効率的に追跡します。
    """

    def __init__(self, name: str = ""):
        """
        Args:
            name: メーター名
        """
        self.name = name
        self.reset()

    def reset(self) -> None:
        """統計をリセット"""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        値を更新

        Args:
            val: 新しい値
            n: サンプル数
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


class EarlyStopping:
    """
    Early Stopping ハンドラー

    検証損失が改善しない場合にトレーニングを停止します。
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        """
        Args:
            patience: 改善がない場合の許容エポック数
            min_delta: 改善とみなす最小変化量
            mode: 'min' (損失等) または 'max' (精度等)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        スコアを評価してearly stoppingするか判定

        Args:
            score: 現在のスコア

        Returns:
            early stoppingするべきかどうか
        """
        if self.best_score is None:
            self.best_score = score
            return False

        # スコアが改善したか判定
        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:  # mode == 'max'
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False

    def reset(self) -> None:
        """状態をリセット"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
