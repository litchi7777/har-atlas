"""
Supervised Fine-tuning スクリプト

事前学習済みエンコーダーを用いたセンサーデータ分類モデルのファインチューニングを実行します。
"""

import argparse
import atexit
import glob
import importlib.util
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

# プロジェクトのモジュールをインポート
from src.data.augmentations import get_augmentation_pipeline
from src.data.batch_dataset import InMemoryDataset, MultiDeviceInMemoryDataset
from src.models.backbones import SensorClassificationModel, MultiDeviceSensorClassificationModel
from src.models.limu_bert import LIMUBertClassifier

# har-unified-datasetからdataset_infoを直接ロード（名前空間の衝突を避けるため）
har_dataset_info_path = project_root / "har-unified-dataset" / "src" / "dataset_info.py"
spec = importlib.util.spec_from_file_location("har_dataset_info", har_dataset_info_path)
har_dataset_info = importlib.util.module_from_spec(spec)
spec.loader.exec_module(har_dataset_info)
DATASETS = har_dataset_info.DATASETS

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
    is_multi_device: bool = False,
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
        is_multi_device: マルチデバイスモードか（データがリスト形式）

    Returns:
        (平均損失, 精度)のタプル
    """
    model.train()
    loss_meter = AverageMeter("Loss")
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}")

    for batch_idx, (data, target) in enumerate(pbar):
        if is_multi_device:
            # data is a list of tensors, one per device
            data = [d.to(device) for d in data]
            target = target.to(device)
        else:
            # data is a single tensor
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

        # 統計を更新（マルチデバイスの場合は最初のデバイスからバッチサイズを取得）
        batch_size = data[0].size(0) if is_multi_device else data.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc = 100.0 * correct / total
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{acc:.2f}%"})

    accuracy = 100.0 * correct / total
    return loss_meter.avg, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_multi_device: bool = False,
) -> Dict[str, float]:
    """モデルを評価

    Args:
        model: 評価するモデル
        dataloader: データローダー
        criterion: 損失関数
        device: デバイス
        is_multi_device: マルチデバイスモードか（データがリスト形式）

    Returns:
        評価メトリクス辞書
    """
    model.eval()
    loss_meter = AverageMeter("Loss")
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Evaluating"):
            if is_multi_device:
                # data is a list of tensors, one per device
                data = [d.to(device) for d in data]
                target = target.to(device)
            else:
                # data is a single tensor
                data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            # バッチサイズの取得（マルチデバイスの場合は最初のデバイスから）
            batch_size = data[0].size(0) if is_multi_device else data.size(0)
            loss_meter.update(loss.item(), batch_size)

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


def extract_datasets_and_locations_from_pairs(
    dataset_location_pairs: List[List],
) -> tuple:
    """dataset_location_pairsからdatasetsとlocationsを抽出

    Args:
        dataset_location_pairs: [[dataset, location], ...] or [[dataset, [loc1, loc2]], ...]
            - ["dsads", "Torso"]: シングルデバイス
            - ["dsads", ["Torso", "RightArm"]]: マルチデバイス

    Returns:
        (datasets, locations, is_multi_device):
            - datasets: ユニークなデータセット名のリスト
            - locations: ロケーションのリスト（マルチの場合はリストのリスト）
            - is_multi_device: マルチデバイスモードかどうか
    """
    if not dataset_location_pairs:
        raise ValueError("dataset_location_pairs is empty")

    datasets = []
    locations = []
    is_multi_device = False

    for pair in dataset_location_pairs:
        dataset, location = pair
        datasets.append(dataset)

        if isinstance(location, list):
            # マルチデバイス
            locations.extend(location)
            is_multi_device = True
        else:
            # シングルデバイス
            locations.append(location)

    # ユニークなデータセットとロケーション
    datasets = sorted(set(datasets))
    locations = sorted(set(locations))

    return datasets, locations, is_multi_device


def get_device_locations_from_config(
    sensor_location: Any,
    data_root: str,
    datasets: List[str],
    logger,
) -> List[str]:
    """sensor_location設定から使用するデバイスリストを取得

    Args:
        sensor_location: 設定値
            - "all": 全デバイス（自動検出）
            - "Torso": 単一デバイス
            - ["Torso", "RightArm"]: 指定デバイスリスト
        data_root: データルートディレクトリ
        datasets: データセット名のリスト
        logger: ロガー

    Returns:
        device_locations: デバイス名のリスト
    """
    # リストで直接指定された場合
    if isinstance(sensor_location, list):
        logger.info(f"Using specified devices: {sensor_location}")
        return sensor_location

    # 文字列の場合
    if isinstance(sensor_location, str):
        if sensor_location == "all":
            # 全デバイスを自動検出
            logger.info("Detecting all available devices...")
            all_devices = set()

            for dataset_name in datasets:
                dataset_name_lower = dataset_name.lower()
                # パターン: data_root/dataset/USER*/*/ACC/X.npy
                pattern = f"{data_root}/{dataset_name_lower}/*/*/ACC/X.npy"
                paths = glob.glob(pattern)

                # パスからデバイス名を抽出
                for path in paths:
                    parts = Path(path).parts
                    # USER* の次がデバイス名
                    for i, part in enumerate(parts):
                        if part.startswith("USER") and i + 1 < len(parts):
                            device_name = parts[i + 1]
                            all_devices.add(device_name)
                            break

            device_locations = sorted(list(all_devices))
            logger.info(f"Detected devices: {device_locations}")
            return device_locations
        else:
            # 単一デバイス名として扱う
            logger.info(f"Using single device: {sensor_location}")
            return [sensor_location]

    raise ValueError(
        f"Invalid sensor_location: {sensor_location}. "
        f"Expected 'all', device name string, or list of device names."
    )


def collect_multi_device_data_paths(
    data_root: str,
    datasets: List[str],
    device_locations: List[str],
    exclude_patterns: List[str],
    test_users: List[str],
    val_users: List[str],
    logger,
) -> Tuple[
    Dict[str, List[Tuple[str, str]]],
    Dict[str, List[Tuple[str, str]]],
    Dict[str, List[Tuple[str, str]]],
    Dict[str, Dict[int, str]],
]:
    """マルチデバイス対応：デバイスごとにデータパスを収集してtrain/val/testに分割

    Args:
        data_root: データルートディレクトリ
        datasets: データセット名のリスト
        device_locations: デバイス部位のリスト（例: ['Torso', 'RightArm', 'LeftArm']）
        exclude_patterns: 除外パターン
        test_users: テスト用ユーザーID
        val_users: 検証用ユーザーID
        logger: ロガー

    Returns:
        (train_paths_per_device, val_paths_per_device, test_paths_per_device, dataset_labels):
            - train/val/test_paths_per_device: {device_name: [(X.npy, Y.npy), ...]}
            - dataset_labels: {dataset_name: {label_id: label_name}}
    """
    train_paths_per_device = {device: [] for device in device_locations}
    val_paths_per_device = {device: [] for device in device_locations}
    test_paths_per_device = {device: [] for device in device_locations}
    dataset_labels = {}

    for dataset_name in datasets:
        dataset_name_lower = dataset_name.lower()
        dataset_name_upper = dataset_name.upper()

        # データセット情報からラベルを取得
        if dataset_name_upper in DATASETS:
            dataset_labels[dataset_name_upper] = DATASETS[dataset_name_upper]["labels"]
        else:
            logger.warning(f"Dataset {dataset_name_upper} not found in DATASETS metadata")

        # 各デバイスについてデータを収集
        for device in device_locations:
            # パターン: data_root/dataset/USER*/device/ACC/X.npy
            pattern = f"{data_root}/{dataset_name_lower}/*/{device}/ACC/X.npy"
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
                    test_paths_per_device[device].append((x_path, y_path))
                elif user_id in val_users:
                    val_paths_per_device[device].append((x_path, y_path))
                else:
                    train_paths_per_device[device].append((x_path, y_path))

            logger.info(
                f"Dataset '{dataset_name}', Device '{device}': "
                f"{len([p for p in x_paths if Path(p.replace('/X.npy', '/Y.npy')).exists()])} files"
            )

    # 統計情報をログ
    for device in device_locations:
        logger.info(
            f"Device '{device}': "
            f"Train={len(train_paths_per_device[device])}, "
            f"Val={len(val_paths_per_device[device])}, "
            f"Test={len(test_paths_per_device[device])}"
        )

    return train_paths_per_device, val_paths_per_device, test_paths_per_device, dataset_labels


def get_num_classes_from_labels(dataset_labels: Dict[str, Dict[int, str]]) -> int:
    """ラベル辞書からクラス数を計算

    Args:
        dataset_labels: {dataset_name: {label_id: label_name}}

    Returns:
        クラス数（負のラベルは除外）

    Note:
        各データセットの最大ラベルIDを使用してクラス数を計算します。
        複数データセットがある場合は、最大値を取ります。
        例: DSADS (0-18) → 19クラス, OPENPACK (0-8) → 9クラス
    """
    max_label = -1
    for labels in dataset_labels.values():
        valid_labels = [k for k in labels.keys() if k >= 0]
        if valid_labels:
            max_label = max(max_label, max(valid_labels))

    # クラス数 = 最大ラベルID + 1 (0-indexedのため)
    return max_label + 1 if max_label >= 0 else 0


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

    # サンプル数制限の設定を取得
    max_samples_per_epoch = config.get("training", {}).get("max_samples_per_epoch", None)

    # DataLoaderを作成
    batch_size = batch_loader_config.get("batch_size", 64)

    # サンプル数制限がある場合、RandomSamplerを使用
    if max_samples_per_epoch is not None and len(train_dataset) > max_samples_per_epoch:
        logger.info(f"Limiting training samples per epoch: {max_samples_per_epoch} (out of {len(train_dataset)})")
        train_sampler = RandomSampler(
            train_dataset,
            replacement=True,  # 復元抽出（同じサンプルを複数回選択可能）
            num_samples=max_samples_per_epoch
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        # Validationもmax_samples_per_epochの1/4に制限
        max_val_samples = max_samples_per_epoch // 4

        logger.info(f"Limiting validation samples: {max_val_samples} (out of {len(val_dataset)})")

        val_sampler = RandomSampler(
            val_dataset,
            replacement=True,
            num_samples=min(max_val_samples, len(val_dataset))
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        # Testは全データを使用
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
    else:
        # 制限なし、通常のシャッフル
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
    in_channels, data_sequence_length = get_input_shape(train_dataset)

    # window_clip が有効な場合は、設定から sequence_length を取得
    # （実際のモデルアーキテクチャはクリップ後のサイズで決まる）
    if sensor_config.get("window_clip", {}).get("enabled", False):
        sequence_length = sensor_config.get("window_size", data_sequence_length)
        logger.info(f"Input shape: ({in_channels}, {data_sequence_length}) -> Window clip enabled, using window_size={sequence_length} for model architecture")
    else:
        sequence_length = data_sequence_length
        logger.info(f"Input shape: ({in_channels}, {sequence_length})")

    logger.info(f"Training batches per epoch: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, num_classes, in_channels, sequence_length


def setup_multi_device_dataloaders(
    config: Dict[str, Any], logger
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int, int, List[str]]:
    """マルチデバイス対応バッチデータローダーのセットアップ

    Args:
        config: 設定辞書
        logger: ロガー

    Returns:
        (train_loader, val_loader, test_loader, num_classes, in_channels, sequence_length, device_locations)
    """
    sensor_config = config["sensor_data"]
    batch_loader_config = sensor_config["batch_loader"]
    user_split = sensor_config["user_split"]

    logger.info("Using multi-device in-memory data loader")

    # sensor_locationからデバイスリストを取得
    device_locations = get_device_locations_from_config(
        sensor_config.get("sensor_location", "all"),
        sensor_config.get("data_root", DEFAULT_DATA_ROOT),
        sensor_config.get("datasets", []),
        logger,
    )
    logger.info(f"Device locations: {device_locations}")

    # デバイスごとにデータパスを収集してtrain/val/testに分割
    train_paths_per_device, val_paths_per_device, test_paths_per_device, dataset_labels = (
        collect_multi_device_data_paths(
            sensor_config.get("data_root", DEFAULT_DATA_ROOT),
            sensor_config.get("datasets", []),
            device_locations,
            batch_loader_config.get("exclude_patterns", []),
            user_split.get("test_users", []),
            user_split.get("val_users", []),
            logger,
        )
    )

    # クラス数を計算
    num_classes = get_num_classes_from_labels(dataset_labels)
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Dataset labels: {dataset_labels}")

    # データセットを作成（全データをメモリに読み込む）
    logger.info("Loading multi-device data into memory...")
    train_dataset = MultiDeviceInMemoryDataset(train_paths_per_device, filter_negative_labels=True)
    val_dataset = MultiDeviceInMemoryDataset(val_paths_per_device, filter_negative_labels=True)
    test_dataset = MultiDeviceInMemoryDataset(test_paths_per_device, filter_negative_labels=True)

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # サンプル数制限の設定を取得
    max_samples_per_epoch = config.get("training", {}).get("max_samples_per_epoch", None)

    # DataLoaderを作成
    batch_size = batch_loader_config.get("batch_size", 64)

    # サンプル数制限がある場合、RandomSamplerを使用
    if max_samples_per_epoch is not None and len(train_dataset) > max_samples_per_epoch:
        logger.info(f"Limiting training samples per epoch: {max_samples_per_epoch} (out of {len(train_dataset)})")
        train_sampler = RandomSampler(
            train_dataset,
            replacement=True,  # 復元抽出（同じサンプルを複数回選択可能）
            num_samples=max_samples_per_epoch
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        # Validationもmax_samples_per_epochの1/4に制限
        max_val_samples = max_samples_per_epoch // 4

        logger.info(f"Limiting validation samples: {max_val_samples} (out of {len(val_dataset)})")

        val_sampler = RandomSampler(
            val_dataset,
            replacement=True,
            num_samples=min(max_val_samples, len(val_dataset))
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        # Testは全データを使用
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
    else:
        # 制限なし、通常のシャッフル
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

    # 入力形状を取得（最初のデバイスのデータから）
    sample_device_data_list, _ = train_dataset[0]
    sample_device_data = sample_device_data_list[0]
    in_channels = sample_device_data.shape[0]
    data_sequence_length = sample_device_data.shape[1]

    # window_clip が有効な場合は、設定から sequence_length を取得
    # （実際のモデルアーキテクチャはクリップ後のサイズで決まる）
    if sensor_config.get("window_clip", {}).get("enabled", False):
        sequence_length = sensor_config.get("window_size", data_sequence_length)
        logger.info(f"Number of devices: {len(device_locations)}")
        logger.info(f"Input shape per device: ({in_channels}, {data_sequence_length}) -> Window clip enabled, using window_size={sequence_length} for model architecture")
    else:
        sequence_length = data_sequence_length
        logger.info(f"Number of devices: {len(device_locations)}")
        logger.info(f"Input shape per device: ({in_channels}, {sequence_length})")

    logger.info(f"Training batches per epoch: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    return (
        train_loader,
        val_loader,
        test_loader,
        num_classes,
        in_channels,
        sequence_length,
        device_locations,
    )


def create_model(
    config: Dict[str, Any],
    num_classes: int,
    in_channels: int,
    sequence_length: int,
    device: torch.device,
    logger,
) -> nn.Module:
    """モデルを作成（シングルデバイス）

    Args:
        config: 設定辞書
        num_classes: クラス数
        in_channels: 入力チャネル数
        sequence_length: 入力シーケンス長
        device: デバイス
        logger: ロガー

    Returns:
        作成されたモデル
    """
    backbone = config["model"].get("backbone", "simple_cnn")

    # LIMU-BERT の場合は専用のクラスを使用
    if backbone == "limu_bert":
        model = LIMUBertClassifier(
            num_classes=num_classes,
            feature_num=in_channels,
            hidden=config["model"].get("hidden", 72),
            hidden_ff=config["model"].get("hidden_ff", 144),
            n_layers=config["model"].get("n_layers", 4),
            n_heads=config["model"].get("n_heads", 4),
            seq_len=sequence_length,
            emb_norm=config["model"].get("emb_norm", True),
            pretrained_path=config["model"].get("pretrained_path"),
            freeze_encoder=config["model"].get("freeze_backbone", False),
            device=device,
        ).to(device)
    else:
        # ウィンドウサイズに応じて適切なアーキテクチャを自動選択
        # （事前学習時と同じロジック）
        nano_window = sequence_length < 20   # 15サンプル用
        micro_window = 20 <= sequence_length < 100  # 30, 60サンプル用

        model = SensorClassificationModel(
            in_channels=in_channels,
            num_classes=num_classes,
            backbone=backbone,
            pretrained_path=config["model"].get("pretrained_path"),
            freeze_backbone=config["model"].get("freeze_backbone", False),
            micro_window=micro_window,
            nano_window=nano_window,
            device=device,
        )

    param_info = count_parameters(model)
    logger.info(
        f"Model created: {backbone}, "
        f"Total params: {param_info['total']:,}, "
        f"Trainable: {param_info['trainable']:,}"
    )

    return model


def create_multi_device_model(
    config: Dict[str, Any],
    num_classes: int,
    in_channels: int,
    sequence_length: int,
    device_locations: List[str],
    device: torch.device,
    logger,
) -> nn.Module:
    """マルチデバイスモデルを作成

    Args:
        config: 設定辞書
        num_classes: クラス数
        in_channels: 各デバイスの入力チャネル数
        sequence_length: 入力シーケンス長
        device_locations: デバイス部位のリスト
        device: デバイス
        logger: ロガー

    Returns:
        作成されたマルチデバイスモデル
    """
    num_devices = len(device_locations)

    # ウィンドウサイズに応じて適切なアーキテクチャを自動選択
    # （事前学習時と同じロジック）
    nano_window = sequence_length < 20   # 15サンプル用
    micro_window = 20 <= sequence_length < 100  # 30, 60サンプル用

    model = MultiDeviceSensorClassificationModel(
        num_devices=num_devices,
        in_channels=in_channels,
        num_classes=num_classes,
        backbone=config["model"].get("backbone", "resnet"),
        pretrained_path=config["model"].get("pretrained_path"),
        freeze_backbone=config["model"].get("freeze_backbone", False),
        micro_window=micro_window,
        nano_window=nano_window,
        device_names=device_locations,
        device=device,
    )

    param_info = count_parameters(model)
    logger.info(
        f"Multi-device model created: {config['model']['backbone']}, "
        f"Devices: {num_devices} ({', '.join(device_locations)}), "
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
            },
            step=epoch,
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
    is_multi_device: bool = False,
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
        is_multi_device: マルチデバイスモードか

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
            model, train_loader, criterion, optimizer, device, epoch, use_wandb, is_multi_device
        )

        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # 評価
        if epoch % eval_interval == 0:
            val_metrics = evaluate(model, val_loader, criterion, device, is_multi_device)

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
    test_metrics = evaluate(model, test_loader, criterion, device, is_multi_device)

    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
    logger.info(f"Test F1: {test_metrics['f1']:.4f}")

    # W&Bにログ（最終epochとして記録）
    if use_wandb:
        wandb.log({
            "test/loss": test_metrics["loss"],
            "test/accuracy": test_metrics["accuracy"],
            "test/precision": test_metrics["precision"],
            "test/recall": test_metrics["recall"],
            "test/f1": test_metrics["f1"],
        }, step=epoch)

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

    # 実験ディレクトリを判定・作成
    # run_experiments.py から呼ばれた場合: 設定ファイルが実験ディレクトリ内にある
    # 直接呼ばれた場合: configs/ 内の設定ファイルを使用
    config_path = Path(args.config)

    if config_path.parent.name != "configs" and (config_path.parent / "config.yaml").exists():
        # run_experiments.py から呼ばれた場合
        # 設定ファイルが実験ディレクトリ内にあるので、そのディレクトリを使用
        experiment_dirs = ExperimentDirs(root=config_path.parent)
    else:
        # 直接呼ばれた場合
        # 新しい実験ディレクトリを作成
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dirs = ExperimentDirs.create(Path("experiments"), run_id)

        # 設定ファイルを実験ディレクトリにコピー
        config_copy_path = experiment_dirs.root / "config.yaml"
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

    # dataset_location_pairsから設定を抽出
    sensor_config = config.get("sensor_data", {})

    # dataset_location_pairsがある場合は新形式
    if "dataset_location_pairs" in sensor_config:
        datasets, locations, is_multi_device = extract_datasets_and_locations_from_pairs(
            sensor_config["dataset_location_pairs"]
        )
        # 設定を更新（既存のコードとの互換性のため）
        sensor_config["datasets"] = datasets
        sensor_config["sensor_location"] = locations if is_multi_device else locations[0] if locations else "all"
        device_locations = locations
        logger.info(f"Using dataset_location_pairs format:")
        logger.info(f"  Datasets: {datasets}")
        logger.info(f"  Locations: {locations}")
    else:
        # 従来形式（後方互換性）
        sensor_location = sensor_config.get("sensor_location", None)
        is_multi_device = False
        device_locations = None

        # マルチデバイスモードの判定条件:
        # 1. sensor_location="all" の場合
        # 2. sensor_locationがリストで複数要素の場合
        if sensor_location == "all":
            is_multi_device = True
        elif isinstance(sensor_location, list) and len(sensor_location) > 1:
            is_multi_device = True

    logger.info(f"Mode: {'Multi-device' if is_multi_device else 'Single-device'}")

    # データローダーを作成
    try:
        if is_multi_device:
            (
                train_loader,
                val_loader,
                test_loader,
                num_classes,
                in_channels,
                sequence_length,
                device_locations,
            ) = setup_multi_device_dataloaders(config, logger)
        else:
            train_loader, val_loader, test_loader, num_classes, in_channels, sequence_length = (
                setup_batch_dataloaders(config, logger)
            )

        logger.info(f"Number of classes: {num_classes}")

    except Exception as e:
        logger.error(f"Failed to create dataloaders: {e}")
        raise

    # モデルを作成
    if is_multi_device:
        model = create_multi_device_model(
            config, num_classes, in_channels, sequence_length, device_locations, device, logger
        )
    else:
        model = create_model(config, num_classes, in_channels, sequence_length, device, logger)

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
        is_multi_device,
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
