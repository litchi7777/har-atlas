"""
Supervised Fine-tuning スクリプト

事前学習済みエンコーダーを用いた分類モデルのファインチューニングを実行します。
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import shutil

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models.model import ClassificationModel
from src.models.sensor_models import SensorClassificationModel
from src.data.dataset import FinetuneDataset
from src.data.sensor_dataset import SensorDataset
from src.data.augmentations import get_augmentation_pipeline

# Import from har-unified-dataset submodule
import sys

har_dataset_path = project_root / "har-unified-dataset"
sys.path.insert(0, str(har_dataset_path))
from src.dataset_info import get_dataset_info, select_sensors
from src.utils.config import load_config, validate_config
from src.utils.common import set_seed, get_device, save_checkpoint, count_parameters
from src.utils.logger import setup_logger
from src.utils.metrics import calculate_metrics
from src.utils.training import get_optimizer, get_scheduler, init_wandb, AverageMeter, EarlyStopping

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_wandb: bool = False,
) -> Tuple[float, float]:
    """
    1エポック分の学習を実行

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
        if use_wandb and batch_idx % 10 == 0:
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
    """
    モデルを評価

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


def main(args: argparse.Namespace) -> None:
    """
    メイン関数

    Args:
        args: コマンドライン引数
    """
    # 設定をロード
    config = load_config(args.config)
    validate_config(config, mode="finetune")

    # 実験ディレクトリを作成
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path("experiments") / "finetune" / f"run_{run_id}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # ログディレクトリ
    log_dir = experiment_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # 設定ファイルを実験ディレクトリにコピー
    shutil.copy(args.config, experiment_dir / "config.yaml")

    # ロガーをセットアップ（実験ディレクトリ内に）
    logger = setup_logger("finetune", str(log_dir))
    logger.info(f"Experiment directory: {experiment_dir}")
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

        else:
            # 画像データの場合
            train_dataset = FinetuneDataset(
                data_path=config["data"]["train_path"],
                augmentation_config=config.get("augmentation"),
            )

            val_dataset = FinetuneDataset(
                data_path=config["data"]["val_path"], augmentation_config=None  # 検証時は拡張なし
            )

            num_classes = len(train_dataset.class_to_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.get("data", {}).get("batch_size", 64),
            shuffle=True,
            num_workers=config.get("data", {}).get("num_workers", 4),
            pin_memory=config.get("data", {}).get("pin_memory", True),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get("data", {}).get("batch_size", 64),
            shuffle=False,
            num_workers=config.get("data", {}).get("num_workers", 4),
            pin_memory=config.get("data", {}).get("pin_memory", True),
        )

        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Validation dataset: {len(val_dataset)} samples")
        logger.info(f"Number of classes: {num_classes}")

    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        raise

    # モデルを作成
    if dataset_type == "sensor":
        # センサーモデル
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
        patience=config.get("early_stopping", {}).get("patience", 10),
        min_delta=config.get("early_stopping", {}).get("min_delta", 0.001),
        mode="max",  # 精度を最大化
    )

    # トレーニングループ
    best_metric = 0.0

    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)

    for epoch in range(1, config["training"]["epochs"] + 1):
        logger.info(f"\nEpoch {epoch}/{config['training']['epochs']}")

        # 学習
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, use_wandb
        )

        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # 評価
        eval_interval = config.get("evaluation", {}).get("eval_interval", 1)
        if epoch % eval_interval == 0:
            val_metrics = evaluate(model, val_loader, criterion, device)

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

            # ベストメトリクスの更新（checkpoint保存なし）
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
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(f"Learning rate: {current_lr:.6f}")

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
        description="Supervised Fine-tuning", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", type=str, default="configs/finetune.yaml", help="Path to configuration file"
    )

    args = parser.parse_args()
    main(args)
