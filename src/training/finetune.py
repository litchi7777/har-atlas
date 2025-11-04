"""
Supervised Fine-tuning スクリプト

事前学習済みエンコーダーを用いたセンサーデータ分類モデルのファインチューニングを実行します。
"""

import argparse
import atexit
import glob
import shutil
import signal
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data.augmentations import get_augmentation_pipeline
from src.data.batch_dataset import InMemoryDataset
from src.models.sensor_models import SensorClassificationModel

# har-unified-datasetサブモジュールをパスに追加
har_dataset_path = project_root / "har-unified-dataset"
sys.path.insert(0, str(har_dataset_path))

# データセットメタデータ（dataset_infoから抜粋）
DATASETS = {
    "DSADS": {
        "n_classes": 19,
        "labels": {
            0: 'Sitting', 1: 'Standing', 2: 'Lying(Back)', 3: 'Lying(Right)',
            4: 'StairsUp', 5: 'StairsDown', 6: 'Standing(Elevator, still)',
            7: 'Moving(elevator)', 8: 'Walking(parking)',
            9: 'Walking(Treadmill, Flat)', 10: 'Walking(Treadmill, Slope)',
            11: 'Running(treadmill)', 12: 'Exercising(Stepper)',
            13: 'Exercising(Cross trainer)', 14: 'Cycling(Exercise bike, Vertical)',
            15: 'Cycling(Exercise bike, Horizontal)', 16: 'Rowing',
            17: 'Jumping', 18: 'PlayingBasketball'
        },
    },
    "MHEALTH": {
        "n_classes": 12,
        "labels": {
            -1: 'Undefined',
            0: 'Standing', 1: 'Sitting', 2: 'LyingDown', 3: 'Walking',
            4: 'StairsUp', 5: 'WaistBendsForward', 6: 'FrontalElevationArms',
            7: 'KneesBending', 8: 'Cycling', 9: 'Jogging',
            10: 'Running', 11: 'JumpFrontBack'
        },
    },
}
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
DEFAULT_DATA_ROOT = "har-unified-dataset/data/processed"
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 4
DEFAULT_EVAL_INTERVAL = 1
DEFAULT_EARLY_STOPPING_PATIENCE = 10
DEFAULT_EARLY_STOPPING_MIN_DELTA = 0.001


@dataclass
class DataLoaderConfig:
    """データローダー設定を保持するデータクラス"""

    data_root: str
    datasets: List[str]
    exclude_patterns: List[str]
    sample_threshold: int
    train_num_samples: int
    val_num_samples: int
    test_num_samples: int
    batch_size: int
    test_users: List[str]
    val_users: List[str]


@dataclass
class ExperimentDirs:
    """実験ディレクトリ構造を保持するデータクラス"""

    root: Path

    @classmethod
    def create(cls, base_dir: Path, run_id: str) -> "ExperimentDirs":
        """実験ディレクトリを作成"""
        root = base_dir / "finetune" / f"run_{run_id}"

        root.mkdir(parents=True, exist_ok=True)

        return cls(root=root)


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


def collect_labeled_data_paths(
    data_root: str,
    datasets: List[str],
    exclude_patterns: List[str],
    test_users: List[str],
    val_users: List[str],
    logger,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]], Dict[str, int]]:
    """ラベル付きデータのパスを収集してtrain/val/testに分割

    Args:
        data_root: データルートディレクトリ
        datasets: データセット名のリスト
        exclude_patterns: 除外パターン
        test_users: テスト用ユーザーID
        val_users: 検証用ユーザーID
        logger: ロガー

    Returns:
        (train_paths, val_paths, test_paths, dataset_labels):
            - train/val/test_paths: [(X.npy path, y.npy path), ...]
            - dataset_labels: {dataset_name: {label_id: label_name}}
    """
    train_paths = []
    val_paths = []
    test_paths = []
    dataset_labels = {}

    for dataset_name in datasets:
        dataset_name_lower = dataset_name.lower()
        dataset_name_upper = dataset_name.upper()

        # データセット情報からラベルを取得
        if dataset_name_upper in DATASETS:
            dataset_labels[dataset_name_upper] = DATASETS[dataset_name_upper]["labels"]
        else:
            logger.warning(f"Dataset {dataset_name_upper} not found in DATASETS metadata")

        # パターン: data_root/dataset/USER*/SENSOR*/ACC/X.npy
        pattern = f"{data_root}/{dataset_name_lower}/*/*/ACC/X.npy"
        x_paths = sorted(glob.glob(pattern))

        # 除外パターンを適用
        for exclude in exclude_patterns:
            x_paths = [p for p in x_paths if exclude not in p]

        # 各X.npyに対応するY.npyを探す
        for x_path in x_paths:
            y_path = x_path.replace("/X.npy", "/Y.npy")
            if not Path(y_path).exists():
                continue

            # ユーザーIDを抽出（例: USER00001 -> 1）
            user_match = Path(x_path).parts
            user_dir = [p for p in user_match if p.startswith("USER")]
            if not user_dir:
                continue

            user_id = user_dir[0].replace("USER", "").lstrip("0") or "0"

            # train/val/test に分割
            if user_id in test_users:
                test_paths.append((x_path, y_path))
            elif user_id in val_users:
                val_paths.append((x_path, y_path))
            else:
                train_paths.append((x_path, y_path))

        logger.info(
            f"Dataset '{dataset_name}' (ACC only): "
            f"{data_root}/{dataset_name_lower}/*/*/ACC/X.npy -> "
            f"{len([p for p in x_paths if Path(p.replace('/X.npy', '/Y.npy')).exists()])} files"
        )

    logger.info(f"Train: {len(train_paths)} files")
    logger.info(f"Val: {len(val_paths)} files")
    logger.info(f"Test: {len(test_paths)} files")

    return train_paths, val_paths, test_paths, dataset_labels


def get_num_classes_from_labels(dataset_labels: Dict[str, Dict[int, str]]) -> int:
    """ラベル辞書からクラス数を計算

    Args:
        dataset_labels: {dataset_name: {label_id: label_name}}

    Returns:
        クラス数（負のラベルは除外）
    """
    all_labels = set()
    for labels in dataset_labels.values():
        all_labels.update([k for k in labels.keys() if k >= 0])
    return len(all_labels)


def get_input_shape(dataset: InMemoryDataset) -> Tuple[int, int]:
    """データセットから入力形状を取得

    Args:
        dataset: データセット

    Returns:
        (channels, sequence_length)
    """
    sample_x, _ = dataset[0]
    in_channels = sample_x.shape[0]
    sequence_length = sample_x.shape[1]
    return in_channels, sequence_length


def setup_batch_dataloaders(
    config: Dict[str, Any], logger
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int, int]:
    """バッチデータローダーのセットアップ

    Args:
        config: 設定辞書
        logger: ロガー

    Returns:
        (train_loader, val_loader, test_loader, num_classes, in_channels, sequence_length)
    """
    sensor_config = config["sensor_data"]
    batch_loader_config = sensor_config["batch_loader"]
    user_split = sensor_config["user_split"]

    logger.info("Using in-memory data loader")

    # データパスを収集してtrain/val/testに分割
    train_paths, val_paths, test_paths, dataset_labels = collect_labeled_data_paths(
        sensor_config.get("data_root", DEFAULT_DATA_ROOT),
        sensor_config.get("datasets", []),
        batch_loader_config.get("exclude_patterns", []),
        user_split.get("test_users", []),
        user_split.get("val_users", []),
        logger,
    )

    # クラス数を計算
    num_classes = get_num_classes_from_labels(dataset_labels)
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Dataset labels: {dataset_labels}")

    # データセットを作成（全データをメモリに読み込む）
    logger.info("Loading data into memory...")
    train_dataset = InMemoryDataset(train_paths, filter_negative_labels=True)
    val_dataset = InMemoryDataset(val_paths, filter_negative_labels=True)
    test_dataset = InMemoryDataset(test_paths, filter_negative_labels=True)

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # DataLoaderを作成（通常のシャッフル）
    batch_size = batch_loader_config.get("batch_size", 64)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # 入力形状を取得
    in_channels, sequence_length = get_input_shape(train_dataset)

    logger.info(f"Input shape: ({in_channels}, {sequence_length})")
    logger.info(f"Training batches per epoch: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, num_classes, in_channels, sequence_length


def create_model(
    config: Dict[str, Any],
    num_classes: int,
    in_channels: int,
    device: torch.device,
    logger,
) -> nn.Module:
    """モデルを作成

    Args:
        config: 設定辞書
        num_classes: クラス数
        in_channels: 入力チャネル数
        device: デバイス
        logger: ロガー

    Returns:
        作成されたモデル
    """
    model = SensorClassificationModel(
        in_channels=in_channels,
        num_classes=num_classes,
        backbone=config["model"].get("backbone", "simple_cnn"),
        pretrained_path=config["model"].get("pretrained_path"),
        freeze_backbone=config["model"].get("freeze_backbone", False),
    ).to(device)

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
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    early_stopping: EarlyStopping,
    config: Dict[str, Any],
    device: torch.device,
    use_wandb: bool,
    experiment_dirs: "ExperimentDirs",
    logger,
) -> Dict[str, float]:
    """トレーニングループを実行

    Args:
        model: モデル
        train_loader: トレーニングデータローダー
        val_loader: 検証データローダー
        test_loader: テストデータローダー
        criterion: 損失関数
        optimizer: オプティマイザー
        scheduler: スケジューラー
        early_stopping: Early stopping オブジェクト
        config: 設定辞書
        device: デバイス
        use_wandb: W&Bを使用するか
        experiment_dirs: 実験ディレクトリ
        logger: ロガー

    Returns:
        結果辞書 {'best_val_accuracy': float, 'test_accuracy': float, ...}
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

    # トレーニング完了後、テストセットで評価（現在のモデルを使用）
    logger.info("=" * 80)
    logger.info("Evaluating on test set...")
    logger.info("=" * 80)

    # テストセットで最終評価
    test_metrics = evaluate(model, test_loader, criterion, device)

    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
    logger.info(f"Test F1: {test_metrics['f1']:.4f}")

    # W&Bにログ
    if use_wandb:
        wandb.log({
            "test/loss": test_metrics["loss"],
            "test/accuracy": test_metrics["accuracy"],
            "test/precision": test_metrics["precision"],
            "test/recall": test_metrics["recall"],
            "test/f1": test_metrics["f1"],
        })

    # 結果を辞書にまとめて返す
    results = {
        "best_val_accuracy": best_metric,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
    }

    return results


def main(args: argparse.Namespace) -> None:
    """メイン関数

    Args:
        args: コマンドライン引数
    """
    # GPU メモリクリーンアップ関数
    def cleanup_gpu():
        """Ctrl+C時やプログラム終了時にGPUメモリを解放"""
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("\n[Cleanup] GPU memory cleared")
        except Exception as e:
            print(f"\n[Cleanup] Error during cleanup: {e}")

    # シグナルハンドラーを設定（Ctrl+C対応）
    def signal_handler(signum, frame):
        print("\n[Signal] Received interrupt signal. Cleaning up...")
        cleanup_gpu()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # プログラム終了時にも実行
    atexit.register(cleanup_gpu)

    # 設定をロード
    config = load_config(args.config)
    validate_config(config, mode="finetune")

    # 実験ディレクトリを作成
    # グリッドサーチ時は run_experiments.py が実験ディレクトリ名を設定済み
    # 通常はタイムスタンプベースのディレクトリを新規作成
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dirs = ExperimentDirs.create(Path("experiments"), run_id)

    # 設定ファイルを実験ディレクトリにコピー（まだコピーされていない場合のみ）
    config_copy_path = experiment_dirs.root / "config.yaml"
    if not config_copy_path.exists():
        shutil.copy(args.config, config_copy_path)

    # ロガーをセットアップ（コンソール出力のみ、run_experiments.pyがexperiment.logに保存）
    logger = setup_logger("finetune", console_only=True)
    logger.info(f"Experiment directory: {experiment_dirs.root}")
    logger.info(f"Configuration loaded from {args.config}")
    logger.info(f"Starting fine-tuning with config: {config['model']['name']}")

    # シード設定
    set_seed(config["seed"])
    logger.info(f"Random seed set to {config['seed']}")

    # デバイス設定
    device = get_device(config["device"])
    logger.info(f"Using device: {device}")

    # データローダーを作成
    try:
        train_loader, val_loader, test_loader, num_classes, in_channels, sequence_length = (
            setup_batch_dataloaders(config, logger)
        )

        logger.info(f"Number of classes: {num_classes}")

    except Exception as e:
        logger.error(f"Failed to create dataloaders: {e}")
        raise

    # モデルを作成
    model = create_model(config, num_classes, in_channels, device, logger)

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
        total_epochs=config["training"]["epochs"],
        warmup_epochs=config["training"].get("warmup_epochs", 0),
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
    results = run_training_loop(
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        early_stopping,
        config,
        device,
        use_wandb,
        experiment_dirs,
        logger,
    )

    # 結果をJSONファイルに保存
    results_file = experiment_dirs.root / "results.json"
    import json
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    # 完了
    logger.info("=" * 80)
    logger.info("Fine-tuning completed!")
    logger.info(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
    logger.info(f"Test accuracy: {results['test_accuracy']:.4f}")
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
