"""
Supervised Fine-tuning スクリプト

事前学習済みエンコーダーを用いた分類モデルのファインチューニングを実行します。
"""

import argparse
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# har-unified-datasetサブモジュールをパスに追加
har_dataset_path = project_root / "har-unified-dataset"
sys.path.insert(0, str(har_dataset_path))

from src.data.augmentations import get_augmentation_pipeline
from src.data.dataset import FinetuneDataset
from src.data.sensor_dataset import SensorDataset
from src.dataset_info import get_dataset_info, select_sensors
from src.models.model import ClassificationModel
from src.models.sensor_models import SensorClassificationModel
from src.utils.common import count_parameters, get_device, set_seed
from src.utils.config import load_config, validate_config
from src.utils.logger import setup_logger
from src.utils.metrics import calculate_metrics
from src.utils.training import (
    AverageMeter,
    EarlyStopping,
    get_optimizer,
    get_scheduler,
    init_wandb,
)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# 定数
LOG_INTERVAL = 10
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 4
DEFAULT_EVAL_INTERVAL = 1
DEFAULT_EARLY_STOPPING_PATIENCE = 10
DEFAULT_EARLY_STOPPING_MIN_DELTA = 0.001


@dataclass
class ExperimentDirs:
    """実験ディレクトリ構造を保持するデータクラス"""

    root: Path
    log: Path

    @classmethod
    def create(cls, base_dir: Path, run_id: str) -> "ExperimentDirs":
        """実験ディレクトリを作成"""
        root = base_dir / "finetune" / f"run_{run_id}"
        log = root / "logs"

        root.mkdir(parents=True, exist_ok=True)
        log.mkdir(exist_ok=True)

        return cls(root=root, log=log)


@dataclass
class DataLoaderParams:
    """DataLoader共通パラメータを保持するデータクラス"""

    batch_size: int
    num_workers: int
    pin_memory: bool

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DataLoaderParams":
        """設定から DataLoader パラメータを作成"""
        data_config = config.get("data", {})
        return cls(
            batch_size=data_config.get("batch_size", DEFAULT_BATCH_SIZE),
            num_workers=data_config.get("num_workers", DEFAULT_NUM_WORKERS),
            pin_memory=data_config.get("pin_memory", True),
        )


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_wandb: bool = False,
) -> Tuple[float, float]:
    """1エポック分の学習を実行

    Args:
        model: 学習するモデル
        dataloader: データローダー
        criterion: 損失関数
        optimizer: オプティマイザー
        device: デバイス
        epoch: 現在のエポック
        use_wandb: W&Bへのログを有効化するか

    Returns:
        (平均損失, 精度)のタプル
    """
    model.train()
    loss_meter = AverageMeter("Loss")
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}")

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # 順伝播
        output = model(data)
        loss = criterion(output, target)

        # 逆伝播
        loss.backward()
        optimizer.step()

        # 精度計算
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # 統計を更新
        loss_meter.update(loss.item(), data.size(0))
        acc = 100.0 * correct / total
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc:.2f}%"})

        # W&Bにログ
        if use_wandb and batch_idx % LOG_INTERVAL == 0:
            wandb.log(
                {
                    "train/batch_loss": loss.item(),
                    "train/batch_accuracy": acc,
                    "train/step": epoch * len(dataloader) + batch_idx,
                }
            )

    accuracy = 100.0 * correct / total
    return loss_meter.avg, accuracy


def evaluate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> Dict[str, float]:
    """モデルを評価

    Args:
        model: 評価するモデル
        dataloader: データローダー
        criterion: 損失関数
        device: デバイス

    Returns:
        評価メトリクス辞書
    """
    model.eval()
    loss_meter = AverageMeter("Loss")
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            loss_meter.update(loss.item(), data.size(0))

            _, predicted = output.max(1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_targets.extend(target.cpu().numpy().tolist())

    # メトリクスを計算
    metrics = calculate_metrics(all_targets, all_preds)
    metrics["loss"] = loss_meter.avg

    return metrics


def create_sensor_datasets(
    config: Dict[str, Any], logger
) -> Tuple[SensorDataset, SensorDataset, int, int, int]:
    """センサーデータセットを作成

    Args:
        config: 設定辞書
        logger: ロガー

    Returns:
        (train_dataset, val_dataset, num_classes, in_channels, sequence_length)
    """
    sensor_config = config["sensor_data"]
    dataset_name = sensor_config["dataset_name"]
    data_root = sensor_config["data_root"]
    mode = sensor_config["mode"]

    # データセット情報を取得
    dataset_info = get_dataset_info(dataset_name, data_root)
    num_classes = dataset_info["n_classes"]

    # センサーを選択
    sensors = select_sensors(
        dataset_name, data_root, mode, sensor_config.get("specific_sensors")
    )

    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Sensors: {sensors}")
    logger.info(f"Classes: {num_classes}")

    # データ拡張
    train_transform = get_augmentation_pipeline(
        config.get("augmentation", {}).get("mode", "light")
    )

    # データセット作成
    train_dataset = SensorDataset(
        data_path=data_root,
        sensor_locations=sensors,
        user_ids=sensor_config["train_users"],
        mode="train",
        transform=train_transform,
    )

    val_dataset = SensorDataset(
        data_path=data_root,
        sensor_locations=sensors,
        user_ids=sensor_config["val_users"],
        mode="val",
        transform=None,
    )

    # 入力チャンネル数を取得
    in_channels = train_dataset.get_num_channels()
    sequence_length = train_dataset.get_sequence_length()

    logger.info(f"Input shape: ({in_channels}, {sequence_length})")

    return train_dataset, val_dataset, num_classes, in_channels, sequence_length


def create_image_datasets(
    config: Dict[str, Any], logger
) -> Tuple[FinetuneDataset, FinetuneDataset, int]:
    """画像データセットを作成

    Args:
        config: 設定辞書
        logger: ロガー

    Returns:
        (train_dataset, val_dataset, num_classes)
    """
    train_dataset = FinetuneDataset(
        data_path=config["data"]["train_path"],
        augmentation_config=config.get("augmentation"),
    )

    val_dataset = FinetuneDataset(
        data_path=config["data"]["val_path"],
        augmentation_config=None,  # 検証時は拡張なし
    )

    num_classes = len(train_dataset.class_to_idx)

    return train_dataset, val_dataset, num_classes


def create_data_loaders(
    train_dataset,
    val_dataset,
    loader_params: DataLoaderParams,
    logger,
) -> Tuple[DataLoader, DataLoader]:
    """データローダーを作成

    Args:
        train_dataset: トレーニングデータセット
        val_dataset: 検証データセット
        loader_params: DataLoaderパラメータ
        logger: ロガー

    Returns:
        (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_params.batch_size,
        shuffle=True,
        num_workers=loader_params.num_workers,
        pin_memory=loader_params.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=loader_params.batch_size,
        shuffle=False,
        num_workers=loader_params.num_workers,
        pin_memory=loader_params.pin_memory,
    )

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Validation dataset: {len(val_dataset)} samples")

    return train_loader, val_loader


def create_model(
    config: Dict[str, Any],
    num_classes: int,
    dataset_type: str,
    in_channels: Optional[int] = None,
    device: torch.device = torch.device("cuda"),
    logger=None,
) -> nn.Module:
    """モデルを作成

    Args:
        config: 設定辞書
        num_classes: クラス数
        dataset_type: データセットタイプ ("sensor" or "image")
        in_channels: 入力チャネル数（センサーデータの場合）
        device: デバイス
        logger: ロガー

    Returns:
        作成されたモデル
    """
    if dataset_type == "sensor":
        if in_channels is None:
            raise ValueError("in_channels is required for sensor models")

        model = SensorClassificationModel(
            in_channels=in_channels,
            num_classes=num_classes,
            backbone=config["model"].get("backbone", "simple_cnn"),
            pretrained_path=config["model"].get("pretrained_path"),
            freeze_backbone=config["model"].get("freeze_backbone", False),
        ).to(device)
    else:
        # 画像モデル
        model = ClassificationModel(config["model"]).to(device)

        # 事前学習済み重みをロード（指定されている場合）
        pretrained_path = config["model"].get("pretrained_path")
        if pretrained_path:
            try:
                model.load_pretrained_encoder(pretrained_path, device=device)
                logger.info(f"Loaded pretrained weights from {pretrained_path}")
            except Exception as e:
                logger.warning(f"Failed to load pretrained weights: {e}")

    param_info = count_parameters(model)
    logger.info(
        f"Model created: {config['model']['backbone']}, "
        f"Total params: {param_info['total']:,}, "
        f"Trainable: {param_info['trainable']:,}"
    )

    return model


def log_validation_metrics(
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_metrics: Dict[str, float],
    optimizer: torch.optim.Optimizer,
    use_wandb: bool,
    logger,
) -> None:
    """検証メトリクスをログ

    Args:
        epoch: エポック数
        train_loss: トレーニング損失
        train_acc: トレーニング精度
        val_metrics: 検証メトリクス
        optimizer: オプティマイザー
        use_wandb: W&Bを使用するか
        logger: ロガー
    """
    logger.info(
        f"Val Loss: {val_metrics['loss']:.4f}, "
        f"Val Acc: {val_metrics['accuracy']:.2f}%, "
        f"Val F1: {val_metrics['f1']:.4f}"
    )

    # W&Bにログ
    if use_wandb:
        wandb.log(
            {
                "train/epoch_loss": train_loss,
                "train/epoch_accuracy": train_acc,
                "val/loss": val_metrics["loss"],
                "val/accuracy": val_metrics["accuracy"],
                "val/precision": val_metrics["precision"],
                "val/recall": val_metrics["recall"],
                "val/f1": val_metrics["f1"],
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/epoch": epoch,
            }
        )


def run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    early_stopping: EarlyStopping,
    config: Dict[str, Any],
    device: torch.device,
    use_wandb: bool,
    logger,
) -> float:
    """トレーニングループを実行

    Args:
        model: モデル
        train_loader: トレーニングデータローダー
        val_loader: 検証データローダー
        criterion: 損失関数
        optimizer: オプティマイザー
        scheduler: スケジューラー
        early_stopping: Early stopping オブジェクト
        config: 設定辞書
        device: デバイス
        use_wandb: W&Bを使用するか
        logger: ロガー

    Returns:
        ベスト精度
    """
    best_metric = 0.0
    num_epochs = config["training"]["epochs"]
    eval_interval = config.get("evaluation", {}).get(
        "eval_interval", DEFAULT_EVAL_INTERVAL
    )

    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)

    for epoch in range(1, num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{num_epochs}")

        # 学習
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, use_wandb
        )

        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # 評価
        if epoch % eval_interval == 0:
            val_metrics = evaluate(model, val_loader, criterion, device)

            log_validation_metrics(
                epoch, train_loss, train_acc, val_metrics, optimizer, use_wandb, logger
            )

            # ベストメトリクスの更新
            current_metric = val_metrics["accuracy"]
            if current_metric > best_metric:
                best_metric = current_metric
                logger.info(f"New best accuracy: {best_metric:.4f}")

            # Early stoppingチェック
            if early_stopping(current_metric):
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

        # 学習率を更新
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if epoch % eval_interval == 0:
                    scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(f"Learning rate: {current_lr:.6f}")

    return best_metric


def main(args: argparse.Namespace) -> None:
    """メイン関数

    Args:
        args: コマンドライン引数
    """
    # 設定をロード
    config = load_config(args.config)
    validate_config(config, mode="finetune")

    # 実験ディレクトリを作成
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dirs = ExperimentDirs.create(Path("experiments"), run_id)

    # 設定ファイルを実験ディレクトリにコピー
    shutil.copy(args.config, experiment_dirs.root / "config.yaml")

    # ロガーをセットアップ
    logger = setup_logger("finetune", str(experiment_dirs.log))
    logger.info(f"Experiment directory: {experiment_dirs.root}")
    logger.info(f"Configuration loaded from {args.config}")
    logger.info(f"Starting fine-tuning with config: {config['model']['name']}")

    # シード設定
    set_seed(config["seed"])
    logger.info(f"Random seed set to {config['seed']}")

    # デバイス設定
    device = get_device(config["device"])
    logger.info(f"Using device: {device}")

    # データセットとデータローダーを作成
    dataset_type = config.get("dataset_type", "image")

    try:
        if dataset_type == "sensor":
            # センサーデータの場合
            train_dataset, val_dataset, num_classes, in_channels, sequence_length = (
                create_sensor_datasets(config, logger)
            )
        else:
            # 画像データの場合
            train_dataset, val_dataset, num_classes = create_image_datasets(
                config, logger
            )
            in_channels = None

        # DataLoaderパラメータを作成
        loader_params = DataLoaderParams.from_config(config)

        # DataLoaderを作成
        train_loader, val_loader = create_data_loaders(
            train_dataset, val_dataset, loader_params, logger
        )

        logger.info(f"Number of classes: {num_classes}")

    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        raise

    # モデルを作成
    model = create_model(config, num_classes, dataset_type, in_channels, device, logger)

    # 損失関数を定義
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config.get("loss", {}).get("label_smoothing", 0.0)
    )

    # オプティマイザーを作成
    optimizer = get_optimizer(
        model=model,
        optimizer_name=config["training"].get("optimizer", "adam"),
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.0),
    )

    # スケジューラーを作成
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_name=config["training"].get("scheduler", "step"),
        step_size=config["training"].get("step_size", 20),
        gamma=config["training"].get("gamma", 0.1),
        T_max=config["training"]["epochs"],
    )

    # W&Bを初期化
    use_wandb = init_wandb(config, model)

    # Early stoppingを初期化
    early_stopping = EarlyStopping(
        patience=config.get("early_stopping", {}).get(
            "patience", DEFAULT_EARLY_STOPPING_PATIENCE
        ),
        min_delta=config.get("early_stopping", {}).get(
            "min_delta", DEFAULT_EARLY_STOPPING_MIN_DELTA
        ),
        mode="max",  # 精度を最大化
    )

    # トレーニングループを実行
    best_metric = run_training_loop(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        early_stopping,
        config,
        device,
        use_wandb,
        logger,
    )

    # 完了
    logger.info("=" * 80)
    logger.info("Fine-tuning completed!")
    logger.info(f"Best accuracy: {best_metric:.4f}")
    logger.info("=" * 80)

    # クリーンアップ
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Supervised Fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/finetune.yaml",
        help="Path to configuration file",
    )

    args = parser.parse_args()
    main(args)
