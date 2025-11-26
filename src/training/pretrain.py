"""
Self-Supervised Pre-training スクリプト

SSL手法（SimCLR、MoCo等）を用いた事前学習を実行します。
"""

import argparse
import atexit
import glob
import os
import shutil
import signal
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data.augmentations import (
    Permutation,
    Reverse,
    TimeWarping,
    ChannelMasking,
    TimeMasking,
    TimeChannelMasking,
    get_augmentation_pipeline,
)
from src.data.batch_dataset import SubjectWiseLoader
from src.data.hierarchical_dataset import (
    HierarchicalSSLDataset,
    BodyPartBatchSampler,
    collate_hierarchical,
    get_activity_name,
)
from src.losses import IntegratedSSLLoss, CombinedSSLLoss
from src.losses.hierarchical_loss import HierarchicalSSLLoss
from src.models.backbones import IntegratedSSLModel, Resnet
from src.utils.atlas_loader import AtlasLoader
from src.utils.common import count_parameters, get_device, save_checkpoint, set_seed
from src.utils.config import load_config, validate_config
from src.utils.logger import setup_logger
from src.utils.training import AverageMeter, get_optimizer, get_scheduler, init_wandb

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# 定数
DEFAULT_DATA_ROOT = "har-unified-dataset/data/processed"
DEFAULT_SSL_TASKS = ["binary_permute", "binary_reverse", "binary_timewarp"]
DEFAULT_TASK_WEIGHTS = [1.0, 1.0, 1.0]
DEFAULT_APPLY_PROB = 0.5
DEFAULT_N_SEGMENTS = 5  # 均等分割のセグメント数
DEFAULT_MAX_WARP_FACTOR = 1.5  # TimeWarping の最大ワープ係数
LOG_INTERVAL = 10
NUM_TASK_METERS = 3


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


@dataclass
class ExperimentDirs:
    """実験ディレクトリ構造を保持するデータクラス"""

    root: Path
    checkpoint: Path

    @classmethod
    def create(cls, base_dir: Path, run_id: str) -> "ExperimentDirs":
        """実験ディレクトリを作成"""
        root = base_dir / "pretrain" / f"run_{run_id}"
        checkpoint = root / "checkpoints"

        root.mkdir(parents=True, exist_ok=True)
        checkpoint.mkdir(exist_ok=True)

        return cls(root=root, checkpoint=checkpoint)


def get_ssl_tasks_from_config(config: Dict[str, Any]) -> List[str]:
    """設定からSSLタスクリストを取得

    Args:
        config: 設定辞書

    Returns:
        SSLタスクのリスト
    """
    multitask_config = config.get("multitask", {})
    return multitask_config.get("ssl_tasks", DEFAULT_SSL_TASKS)


def create_augmentation_transforms(
    ssl_tasks: List[str], config: Dict[str, Any]
) -> Dict[str, Any]:
    """SSL taskに応じた拡張変換を作成（通常の拡張パイプラインも含む）

    Args:
        ssl_tasks: SSLタスクのリスト
        config: 設定辞書（マスク比率等の取得用）

    Returns:
        拡張変換の辞書
        - binary_*タスク用: {'permute': Transform, 'reverse': Transform, ...}
        - masking_*タスク用: {'channel': Transform, 'time': Transform, ...}
        - 'base_augmentation': 通常のデータ拡張パイプライン
    """
    transforms = {}

    # 通常のデータ拡張パイプラインを追加（有効な場合のみ）
    augmentation_config = config.get("augmentation", {})
    aug_enabled = augmentation_config.get("enabled", True)  # デフォルトは有効

    if aug_enabled:
        aug_mode = augmentation_config.get("mode", "heavy")  # デフォルトはheavy
        training_config = config.get("training", {})
        max_epochs = training_config.get("epochs", 100)

        transforms["base_augmentation"] = get_augmentation_pipeline(
            mode=aug_mode, max_epochs=max_epochs
        )

    # Binary拡張タスクの処理
    binary_tasks = [task for task in ssl_tasks if task.startswith("binary_")]
    aug_names = [task.replace("binary_", "") for task in binary_tasks]

    for aug_name in aug_names:
        if aug_name == "permute" or aug_name == "permute_fast":
            transforms["permute"] = Permutation(n_segments=DEFAULT_N_SEGMENTS)
        elif aug_name == "reverse":
            transforms["reverse"] = Reverse()
        elif aug_name == "timewarp" or aug_name == "timewarp_fast":
            transforms["timewarp"] = TimeWarping(max_warp_factor=DEFAULT_MAX_WARP_FACTOR)

    # Maskingタスクの処理
    masking_tasks = [task for task in ssl_tasks if task.startswith("masking_")]
    mask_types = [task.replace("masking_", "") for task in masking_tasks]

    # マスク比率を設定から取得
    multitask_config = config.get("multitask", {})
    mask_ratio = multitask_config.get("mask_ratio", 0.15)

    for mask_type in mask_types:
        if mask_type == "channel":
            transforms["channel"] = ChannelMasking(mask_ratio=mask_ratio)
        elif mask_type == "time":
            transforms["time"] = TimeMasking(mask_ratio=mask_ratio)
        elif mask_type == "time_channel":
            transforms["time_channel"] = TimeChannelMasking(
                time_mask_ratio=mask_ratio, channel_mask_ratio=mask_ratio
            )

    # 回転拡張の処理（オプション、すべてのタスクに適用）
    rotation_config = config.get("rotation_augmentation", {})
    rotation_enabled = rotation_config.get("enabled", False)

    if rotation_enabled:
        from src.data.augmentations import RandomRotation3D
        rotation_type = rotation_config.get("rotation_type", "random")
        transforms["rotation_augmentation"] = RandomRotation3D(rotation_type=rotation_type)

    return transforms


def collect_data_paths(
    data_root: str,
    dataset_location_pairs: List[List],
    exclude_patterns: List[str],
    logger,
) -> List[Union[str, Tuple[str, ...]]]:
    """データセットからファイルパスを収集

    Args:
        data_root: データルートディレクトリ
        dataset_location_pairs: (dataset, location)のペアのリスト
            - シングルデバイス: [["dsads", "LeftArm"], ["mhealth", "Chest"]]
            - マルチデバイス: [["dsads", ["LeftArm", "RightArm"]]]
        exclude_patterns: 除外パターンのリスト
        logger: ロガー

    Returns:
        収集されたファイルパスのリスト
        - シングルデバイスの場合: 各ファイルパスを個別に返す
        - マルチデバイスの場合: 同一ユーザーの複数デバイスファイルをまとめて返す

    Raises:
        ValueError: データセットが空、またはファイルが見つからない場合
    """
    all_paths = []

    logger.info(f"Using dataset_location_pairs: {len(dataset_location_pairs)} pairs")

    for pair in dataset_location_pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(
                f"Each pair in dataset_location_pairs must be [dataset, location] or [dataset, [loc1, loc2]], got: {pair}"
            )
        dataset_name, location = pair

        # locationが文字列（シングルデバイス）かリスト（マルチデバイス）か判定
        if isinstance(location, str):
            # シングルデバイス: 従来通りの処理
            pattern = f"{data_root}/{dataset_name}/*/{location}/ACC/X.npy"
            paths = sorted(glob.glob(pattern))
            logger.info(
                f"Dataset '{dataset_name}', Location '{location}' (single): {len(paths)} files"
            )
            all_paths.extend(paths)

        elif isinstance(location, list):
            # マルチデバイス: 同一ユーザーの複数デバイスファイルを収集
            # まず最初のlocationでユーザーIDを取得
            first_location = location[0]
            pattern = f"{data_root}/{dataset_name}/*/{first_location}/ACC/X.npy"
            first_paths = sorted(glob.glob(pattern))

            logger.info(
                f"Dataset '{dataset_name}', Locations {location} (multi-device): "
                f"Found {len(first_paths)} users"
            )

            # 各ユーザーについて、全てのlocationのファイルが存在するか確認
            for first_path in first_paths:
                # パスからユーザーIDを抽出
                # 例: data_root/dsads/USER00001/LeftArm/ACC/X.npy -> USER00001
                parts = Path(first_path).parts
                dataset_idx = parts.index(dataset_name)
                user_id = parts[dataset_idx + 1]

                # 全てのlocationのファイルが存在するか確認
                device_paths = []
                all_exist = True
                for loc in location:
                    device_path = f"{data_root}/{dataset_name}/{user_id}/{loc}/ACC/X.npy"
                    if os.path.exists(device_path):
                        device_paths.append(device_path)
                    else:
                        all_exist = False
                        logger.warning(
                            f"Missing file for user {user_id}, location {loc}: {device_path}"
                        )
                        break

                # 全てのデバイスのファイルが存在する場合のみ追加
                if all_exist:
                    # マルチデバイスの場合はタプルとして格納（後で結合するため）
                    all_paths.append(tuple(device_paths))
                else:
                    logger.warning(f"Skipping user {user_id} due to missing device data")

        else:
            raise ValueError(f"Invalid location type: {type(location)}, expected str or list")

    # 除外パターンのフィルタリング
    if exclude_patterns:
        filtered_paths = []
        for p in all_paths:
            # マルチデバイスの場合はタプル、シングルデバイスは文字列
            if isinstance(p, tuple):
                # タプル内のいずれかのパスに除外パターンが含まれる場合はスキップ
                if not any(any(exclude in path for exclude in exclude_patterns) for path in p):
                    filtered_paths.append(p)
            else:
                # 文字列の場合は従来通り
                if not any(exclude in p for exclude in exclude_patterns):
                    filtered_paths.append(p)
        all_paths = filtered_paths

    if len(all_paths) == 0:
        raise ValueError("No data files found. Check dataset_location_pairs configuration.")

    logger.info(f"Total: {len(all_paths)} files")

    return all_paths


def create_data_loaders(
    paths: List[Union[str, Tuple[str, ...]]],
    config: DataLoaderConfig,
    use_multitask: bool,
    ssl_tasks: List[str],
    specific_transforms: Optional[Dict[str, Any]] = None,
    apply_prob: float = DEFAULT_APPLY_PROB,
    window_size: Optional[int] = None,
    original_window_size: Optional[int] = None,
    window_clip_strategy: str = "random",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """データローダーを作成

    Args:
        paths: データファイルパスのリスト
        config: データローダー設定
        use_multitask: マルチタスク学習を使用するか
        ssl_tasks: SSLタスクのリスト
        specific_transforms: 拡張変換の辞書
        apply_prob: 拡張適用確率
        window_size: クリップ後のウィンドウサイズ
        original_window_size: 元のウィンドウサイズ
        window_clip_strategy: クリップ戦略

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # データセットを作成（統一されたSubjectWiseLoaderを使用）
    dataset = SubjectWiseLoader(
        paths,
        sample_threshold=config.sample_threshold,
        ssl_tasks=ssl_tasks if use_multitask else None,
        window_size=window_size,
        original_window_size=original_window_size,
        window_clip_strategy=window_clip_strategy,
    )

    # DataLoaderを作成（複数の被験者データを1バッチに含める）
    train_sampler = RandomSampler(
        dataset, replacement=True, num_samples=config.train_num_samples
    )
    train_loader = DataLoader(
        dataset, batch_size=config.batch_size, drop_last=True, sampler=train_sampler
    )

    val_sampler = RandomSampler(
        dataset, replacement=True, num_samples=config.val_num_samples
    )
    val_loader = DataLoader(
        dataset, batch_size=config.batch_size, drop_last=True, sampler=val_sampler
    )

    test_sampler = RandomSampler(
        dataset, replacement=True, num_samples=config.test_num_samples
    )
    test_loader = DataLoader(
        dataset, batch_size=config.batch_size, drop_last=True, sampler=test_sampler
    )

    return train_loader, val_loader, test_loader


def get_input_shape(dataset) -> Tuple[int, int]:
    """データセットから入力形状を取得

    Args:
        dataset: データセット

    Returns:
        (channels, sequence_length)
    """
    sample_x = dataset[0]
    # sample_x shape: (sample_threshold, channels, sequence_length)
    if sample_x.dim() == 3:
        in_channels = sample_x.shape[1]
        sequence_length = sample_x.shape[2]
    else:
        # 旧形式との互換性
        in_channels = sample_x.shape[1]
        sequence_length = sample_x.shape[2]
    return in_channels, sequence_length


def setup_batch_dataloaders(
    config: Dict[str, Any],
    logger,
    use_multitask: bool,
    multitask_config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """バッチデータローダーのセットアップ

    Args:
        config: 設定辞書
        logger: ロガー
        use_multitask: マルチタスク学習を使用するか
        multitask_config: マルチタスク設定

    Returns:
        (train_loader, val_loader, test_loader, in_channels, sequence_length)
    """
    # dataセクションを使用（sensor_dataは廃止）
    data_config = config.get("data", {})
    batch_loader_config = data_config.get("batch_loader", {})

    logger.info("Using batch data loader")

    # 拡張を準備（マルチタスクの場合）
    specific_transforms = {}
    ssl_tasks = []
    if use_multitask:
        ssl_tasks = get_ssl_tasks_from_config(config)
        specific_transforms = create_augmentation_transforms(ssl_tasks, config)

        binary_tasks = [task for task in ssl_tasks if task.startswith("binary_")]
        masking_tasks = [task for task in ssl_tasks if task.startswith("masking_")]

        logger.info(f"SSL tasks: {ssl_tasks}")
        logger.info(f"Binary tasks: {binary_tasks}")
        logger.info(f"Masking tasks: {masking_tasks}")
        logger.info(f"Transforms: {list(specific_transforms.keys())}")

    # データローダー設定を構築
    loader_config = DataLoaderConfig(
        data_root=data_config.get("data_root", DEFAULT_DATA_ROOT),
        datasets=data_config.get("datasets", []),
        exclude_patterns=batch_loader_config.get("exclude_patterns", []),
        sample_threshold=batch_loader_config.get("sample_threshold", 2000),
        train_num_samples=batch_loader_config.get("train_num_samples", 1000),
        val_num_samples=batch_loader_config.get("val_num_samples", 100),
        test_num_samples=batch_loader_config.get("test_num_samples", 100),
        batch_size=batch_loader_config.get("batch_size", 4),
    )

    # データパスを収集
    dataset_location_pairs = data_config.get("dataset_location_pairs")
    if dataset_location_pairs is None:
        raise ValueError(
            "dataset_location_pairs is required in data configuration. "
            "Example: dataset_location_pairs: [['dsads', 'LeftArm'], ['mhealth', 'chest']]"
        )

    all_paths = collect_data_paths(
        loader_config.data_root,
        dataset_location_pairs,
        loader_config.exclude_patterns,
        logger,
    )

    # ウィンドウクリップ設定を取得
    window_size = data_config.get("window_size")
    original_window_size = data_config.get("original_window_size")
    window_clip_config = data_config.get("window_clip", {})
    window_clip_strategy = window_clip_config.get("strategy", "random")

    # データローダーを作成
    apply_prob = multitask_config.get("apply_prob", DEFAULT_APPLY_PROB)
    train_loader, val_loader, test_loader = create_data_loaders(
        all_paths,
        loader_config,
        use_multitask,
        ssl_tasks,
        specific_transforms,
        apply_prob,
        window_size=window_size,
        original_window_size=original_window_size,
        window_clip_strategy=window_clip_strategy,
    )

    # 入力形状を取得
    dataset = (
        train_loader.dataset.dataset
        if hasattr(train_loader.dataset, "dataset")
        else train_loader.dataset
    )
    in_channels, sequence_length = get_input_shape(dataset)

    logger.info(f"Input shape: ({in_channels}, {sequence_length})")
    logger.info(f"Training batches per epoch: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, in_channels, sequence_length


def setup_hierarchical_dataloaders(
    config: Dict[str, Any],
    logger,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int, AtlasLoader]:
    """階層的SSL用のデータローダーをセットアップ

    Args:
        config: 設定辞書
        logger: ロガー

    Returns:
        (train_loader, val_loader, test_loader, in_channels, sequence_length, atlas)
    """
    hierarchical_config = config.get("hierarchical", {})
    data_config = config.get("data", {})

    # Atlas読み込み
    atlas_path = hierarchical_config.get("atlas_path", "docs/atlas/activity_mapping.json")
    atlas = AtlasLoader(atlas_path)
    logger.info(f"Atlas loaded: {len(atlas.get_datasets())} datasets")

    # データセット設定
    data_root = data_config.get("data_root", DEFAULT_DATA_ROOT)
    dataset_location_pairs = data_config.get("dataset_location_pairs", [])

    if not dataset_location_pairs:
        raise ValueError("dataset_location_pairs is required for hierarchical SSL")

    # データセット作成
    train_dataset = HierarchicalSSLDataset(
        data_root=data_root,
        dataset_location_pairs=dataset_location_pairs,
        split="train",
    )
    val_dataset = HierarchicalSSLDataset(
        data_root=data_root,
        dataset_location_pairs=dataset_location_pairs,
        split="val",
    )
    test_dataset = HierarchicalSSLDataset(
        data_root=data_root,
        dataset_location_pairs=dataset_location_pairs,
        split="test",
    )

    logger.info(f"Hierarchical SSL - Train files: {len(train_dataset)}")
    logger.info(f"Hierarchical SSL - Val files: {len(val_dataset)}")
    logger.info(f"Hierarchical SSL - Test files: {len(test_dataset)}")

    # データローダー設定
    training_config = config.get("training", {})
    batch_size = training_config.get("batch_size", 8)
    samples_per_source = training_config.get("samples_per_source", 32)
    batches_per_epoch = training_config.get("batches_per_epoch", 100)
    unlabeled_datasets = hierarchical_config.get("unlabeled_datasets", ["nhanes"])
    unlabeled_per_batch = hierarchical_config.get("unlabeled_per_batch", 2)

    # collate_fn をカリー化して samples_per_source を渡す
    from functools import partial
    collate_fn = partial(collate_hierarchical, samples_per_source=samples_per_source)

    # Body Part別バッチサンプラーを使用
    train_sampler = BodyPartBatchSampler(
        dataset=train_dataset,
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        unlabeled_datasets=unlabeled_datasets,
        unlabeled_per_batch=unlabeled_per_batch,
        seed=config.get("seed", 42),
    )

    # Body Part別の統計をログ
    logger.info("Body Part distribution (train):")
    for bp, indices in train_sampler.body_part_indices.items():
        logger.info(f"  {bp}: {len(indices)} files")
    logger.info(f"  unlabeled ({unlabeled_datasets}): {len(train_sampler.unlabeled_indices)} files")

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Val/TestもBodyPartBatchSamplerを使用（Body Part混在を避ける）
    val_sampler = BodyPartBatchSampler(
        dataset=val_dataset,
        batch_size=batch_size,
        batches_per_epoch=None,  # 全データ使用
        unlabeled_datasets=unlabeled_datasets,
        unlabeled_per_batch=unlabeled_per_batch,
        seed=config.get("seed", 42) + 1,  # 異なるシード
    )
    test_sampler = BodyPartBatchSampler(
        dataset=test_dataset,
        batch_size=batch_size,
        batches_per_epoch=None,  # 全データ使用
        unlabeled_datasets=unlabeled_datasets,
        unlabeled_per_batch=unlabeled_per_batch,
        seed=config.get("seed", 42) + 2,  # 異なるシード
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # 入力形状を取得（最初のサンプルから）
    sample = train_dataset[0]
    in_channels = sample["data"].shape[1]
    sequence_length = sample["data"].shape[2]

    logger.info(f"Input shape: ({in_channels}, {sequence_length})")
    logger.info(f"Training batches per epoch: {len(train_sampler)}")
    logger.info(f"Actual batch size: {batch_size} files × {samples_per_source} samples = ~{batch_size * samples_per_source} samples/batch")

    return train_loader, val_loader, test_loader, in_channels, sequence_length, atlas


def apply_augmentations_batch(
    x: torch.Tensor,
    ssl_tasks: List[str],
    transforms: Dict[str, Any],
    apply_prob: float,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """バッチ全体に拡張を適用してラベルを生成（各タスクで独立した拡張を適用）

    Args:
        x: 入力データ (batch_size, channels, time_steps)
        ssl_tasks: SSLタスクのリスト
        transforms: 拡張辞書
            - 'base_augmentation': 通常の拡張パイプライン
            - 'rotation_augmentation': 回転拡張（オプション）
        apply_prob: binary_*タスクの適用確率
        device: デバイス

    Returns:
        (task_inputs_dict, labels_dict):
            - task_inputs_dict: 各タスク用の拡張データ {'binary_permute': tensor, ...}
            - labels_dict: 各タスクのラベル {'binary_permute': tensor, ...}
    """
    import torch
    import numpy as np

    batch_size = x.shape[0]
    original_x = x.clone()
    task_inputs_dict = {}
    labels_dict = {}

    # 通常のデータ拡張を最初に適用（全タスク共通のベース）
    x_base = x.to(device)  # 入力を確実にデバイスに転送
    if "base_augmentation" in transforms:
        base_aug = transforms["base_augmentation"]
        # GPU上で直接適用
        augmented_batch = []
        for sample in x_base:
            augmented_sample = base_aug(sample)
            augmented_batch.append(augmented_sample)
        x_base = torch.stack(augmented_batch).to(device)  # 確実にデバイスに転送

    # 回転拡張を適用（オプション、すべてのタスクに適用）
    # バッチ内の各サンプルに異なるランダム回転を適用して偏りを解消
    if "rotation_augmentation" in transforms:
        rotation_aug = transforms["rotation_augmentation"]
        x_base = rotation_aug(x_base)  # バッチ全体に高速適用

    # Binary拡張タスク - バッチ全体に変換を適用してからマスクで選択
    binary_tasks = [t for t in ssl_tasks if t.startswith("binary_")]
    for task in binary_tasks:
        aug_name = task.replace("binary_", "")

        # サンプルごとにランダムに決定
        apply_mask = np.random.random(batch_size) < apply_prob  # shape: (batch_size,)

        if aug_name in transforms:
            # バッチ全体に変換を適用
            x_augmented = torch.stack([transforms[aug_name](sample) for sample in x_base])
            # デバイスに明示的に転送
            x_augmented = x_augmented.to(device)

            # マスクで選択（ベクトル演算）
            # apply_mask を (batch_size, 1, 1) に変形してブロードキャスト
            mask_tensor = torch.from_numpy(apply_mask.astype(np.float32)).to(device)
            mask_tensor = mask_tensor.view(batch_size, 1, 1)  # (B, 1, 1)

            # mask * augmented + (1 - mask) * original
            task_inputs_dict[task] = mask_tensor * x_augmented + (1 - mask_tensor) * x_base
        else:
            task_inputs_dict[task] = x_base.clone()
            apply_mask = np.zeros(batch_size, dtype=bool)

        # サンプルごとに異なるラベル
        labels_dict[task] = torch.from_numpy(apply_mask.astype(np.int64)).to(device)

    # Maskingタスク - 各タスクで独立したマスキングを適用
    masking_tasks = [t for t in ssl_tasks if t.startswith("masking_")]
    for task in masking_tasks:
        mask_type = task.replace("masking_", "")

        if mask_type in transforms:
            masking_transform = transforms[mask_type]

            # ベースデータをコピーして、このタスク専用のマスキングを適用
            x_task = x_base.clone()
            # CPU上でマスキングを適用
            x_np = x_task.cpu().numpy()
            masked_batch = []
            for sample in x_np:
                masked_sample, _ = masking_transform(sample)
                masked_batch.append(masked_sample)
            task_inputs_dict[task] = torch.from_numpy(np.stack(masked_batch)).float()
        else:
            task_inputs_dict[task] = x_base.clone()

        # ラベル: 元データ（再構成ターゲット）
        labels_dict[task] = original_x

    # Invarianceタスク - Rotation Contrastive Learning等
    invariant_tasks = [t for t in ssl_tasks if t.startswith("invariant_")]
    for task in invariant_tasks:
        invariant_type = task.replace("invariant_", "")

        # view1: すでに回転済みのデータ（rotation_augmentationが有効な場合）
        task_inputs_dict[task] = x_base.clone()

        # view2: さらに別の回転を適用（対照学習用）
        if invariant_type == "orientation":
            # 回転変換を取得（rotation_augmentationと同じものを使用）
            if "rotation_augmentation" in transforms:
                rotation_transform = transforms["rotation_augmentation"]
            else:
                # なければその場で作成
                from src.data.augmentations import RandomRotation3D
                rotation_transform = RandomRotation3D(rotation_type='random')

            # バッチ全体にさらに別の回転を適用
            # x_base: すでに回転済み → さらに回転 = 2つの異なる回転状態
            x_rotated = rotation_transform(x_base)

            # ラベル: さらに回転後のデータ（contrastive lossではview2として使用）
            labels_dict[task] = x_rotated
        else:
            # 他のinvarianceタイプ（将来の拡張用）
            labels_dict[task] = x_base.clone()

    # デバイスに移動（Binary変換は既にGPU上、Masking変換はCPUから転送）
    task_inputs_dict = {k: v.to(device) if v.device != device else v for k, v in task_inputs_dict.items()}
    labels_dict = {k: v.to(device) if v.device != device else v for k, v in labels_dict.items()}

    return task_inputs_dict, labels_dict


def process_multitask_batch(
    x: torch.Tensor,
    model: nn.Module,
    criterion: nn.Module,
    ssl_tasks: List[str],
    transforms: Dict[str, Any],
    apply_prob: float,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """マルチタスクバッチを処理（各タスクに独立した拡張データを渡す）

    Args:
        x: 入力データ
        model: モデル
        criterion: 損失関数
        ssl_tasks: SSLタスクのリスト
        transforms: 拡張辞書
        apply_prob: binary_*タスクの適用確率
        device: デバイス

    Returns:
        (total_loss, task_losses, x_base) 処理後のデータ
    """
    # バッチローダー使用時は [batch_size, sample_threshold, channels, time]
    # -> [batch_size * sample_threshold, channels, time] にreshape
    if x.dim() == 4:
        batch_size_loader, sample_threshold, channels, time = x.shape
        x = x.reshape(batch_size_loader * sample_threshold, channels, time)

    # 拡張を適用してタスクごとの入力とラベルを生成
    task_inputs_dict, labels_dict = apply_augmentations_batch(
        x, ssl_tasks, transforms, apply_prob, device
    )

    # 各タスクについて、専用の拡張データで順伝播
    predictions = {}
    labels_for_loss = {}
    for task in ssl_tasks:
        x_task = task_inputs_dict[task]

        if task.startswith("invariant_"):
            # Invarianceタスク: 2つのビューを両方エンコード
            z1 = model(x_task, task)  # 元データの埋め込み
            x_rotated = labels_dict[task]  # 回転後のデータ
            z2 = model(x_rotated, task)  # 回転データの埋め込み

            # predictions: z1, labels_for_loss: z2（埋め込み）として損失関数に渡す
            predictions[task] = z1
            labels_for_loss[task] = z2
        else:
            # 他のタスク: 通常通り
            pred = model(x_task, task)
            predictions[task] = pred
            labels_for_loss[task] = labels_dict[task]

    # 損失計算
    total_loss, task_losses = criterion(predictions, labels_for_loss)

    # バッチサイズ計算用に最初のタスクの入力を返す
    x_base = task_inputs_dict[ssl_tasks[0]]

    return total_loss, task_losses, x_base


def train_hierarchical_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger,
) -> Dict[str, float]:
    """階層的SSL: 1エポック分の学習を実行

    Args:
        model: 学習するモデル
        loss_fn: HierarchicalSSLLoss
        dataloader: データローダー
        optimizer: オプティマイザー
        device: デバイス
        epoch: 現在のエポック
        logger: ロガー

    Returns:
        平均損失の辞書
    """
    model.train()
    loss_fn.train()

    loss_meter = AverageMeter("Loss")
    activity_loss_meter = AverageMeter("Activity")
    prototype_loss_meter = AverageMeter("Prototype")

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        if batch is None:
            continue

        data = batch["data"].to(device)  # (B, C, T)
        labels = batch["labels"]  # (B,)
        datasets = batch["datasets"]  # List[str]
        body_parts = batch["body_parts"]  # List[str]

        # ラベルからActivity名を取得
        activity_ids = []
        for ds, label in zip(datasets, labels.tolist()):
            activity_name = get_activity_name(ds, label)
            activity_ids.append(activity_name)

        # Forward
        embeddings = model(data)

        # Loss計算
        total_loss, loss_dict = loss_fn(
            embeddings=embeddings,
            dataset_ids=datasets,
            activity_ids=activity_ids,
            body_parts=body_parts,
        )

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # メーター更新
        batch_size = data.size(0)
        loss_meter.update(total_loss.item(), batch_size)
        activity_loss_meter.update(loss_dict["activity"].item(), batch_size)
        prototype_loss_meter.update(loss_dict["prototype"].item(), batch_size)

        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "act": f"{activity_loss_meter.avg:.4f}",
            "proto": f"{prototype_loss_meter.avg:.4f}",
        })

    return {
        "loss": loss_meter.avg,
        "activity_loss": activity_loss_meter.avg,
        "prototype_loss": prototype_loss_meter.avg,
    }


def evaluate_hierarchical_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """階層的SSL: 検証

    Args:
        model: 評価するモデル
        loss_fn: HierarchicalSSLLoss
        dataloader: データローダー
        device: デバイス

    Returns:
        平均損失の辞書
    """
    model.eval()
    loss_fn.eval()

    loss_meter = AverageMeter("Loss")
    activity_loss_meter = AverageMeter("Activity")
    prototype_loss_meter = AverageMeter("Prototype")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            if batch is None:
                continue

            data = batch["data"].to(device)
            labels = batch["labels"]
            datasets = batch["datasets"]
            body_parts = batch["body_parts"]

            activity_ids = []
            for ds, label in zip(datasets, labels.tolist()):
                activity_name = get_activity_name(ds, label)
                activity_ids.append(activity_name)

            embeddings = model(data)

            total_loss, loss_dict = loss_fn(
                embeddings=embeddings,
                dataset_ids=datasets,
                activity_ids=activity_ids,
                body_parts=body_parts,
            )

            batch_size = data.size(0)
            loss_meter.update(total_loss.item(), batch_size)
            activity_loss_meter.update(loss_dict["activity"].item(), batch_size)
            prototype_loss_meter.update(loss_dict["prototype"].item(), batch_size)

    return {
        "val_loss": loss_meter.avg,
        "val_activity_loss": activity_loss_meter.avg,
        "val_prototype_loss": prototype_loss_meter.avg,
    }


def train_combined_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    ssl_tasks: List[str],
    transforms: Dict[str, Any],
    apply_prob: float,
    logger,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_clip_norm: Optional[float] = None,
) -> Dict[str, float]:
    """MTL + 階層的SSL統合: 1エポック分の学習を実行

    Args:
        model: IntegratedSSLModel（MTL用）
        loss_fn: CombinedSSLLoss
        dataloader: HierarchicalSSLDataset用データローダー
        optimizer: オプティマイザー
        device: デバイス
        epoch: 現在のエポック
        ssl_tasks: SSLタスクのリスト
        transforms: 拡張辞書
        apply_prob: binary_*タスクの適用確率
        logger: ロガー
        scaler: Mixed Precision用
        grad_clip_norm: 勾配クリッピング

    Returns:
        平均損失の辞書
    """
    model.train()
    loss_fn.train()

    loss_meter = AverageMeter("Loss")
    mtl_loss_meter = AverageMeter("MTL")
    hier_loss_meter = AverageMeter("Hier")

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    use_amp = scaler is not None

    for batch in pbar:
        if batch is None:
            continue

        data = batch["data"].to(device)  # (B, C, T)
        labels = batch["labels"]  # (B,)
        datasets = batch["datasets"]  # List[str]
        body_parts = batch["body_parts"]  # List[str]

        # ラベルからActivity名を取得
        activity_ids = []
        for ds, label in zip(datasets, labels.tolist()):
            activity_name = get_activity_name(ds, label)
            activity_ids.append(activity_name)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            # MTL用の拡張を適用
            task_inputs_dict, labels_dict = apply_augmentations_batch(
                data, ssl_tasks, transforms, apply_prob, device
            )

            # MTLタスクの予測を計算
            predictions = {}
            labels_for_mtl = {}
            for task in ssl_tasks:
                x_task = task_inputs_dict[task]

                if task.startswith("invariant_"):
                    # Invarianceタスク: 2つのビューを両方エンコード
                    z1 = model(x_task, task)
                    x_rotated = labels_dict[task]
                    z2 = model(x_rotated, task)
                    predictions[task] = z1
                    labels_for_mtl[task] = z2
                else:
                    pred = model(x_task, task)
                    predictions[task] = pred
                    labels_for_mtl[task] = labels_dict[task]

            # 階層的Loss用のembeddings（encoder出力）
            # 最初のタスクの入力を使ってembeddingsを取得
            with torch.no_grad():
                # バックボーンの出力を取得（projection head前）
                backbone_features = model.backbone(task_inputs_dict[ssl_tasks[0]])
            # embeddingsは勾配を流すために再計算
            embeddings = model.backbone(task_inputs_dict[ssl_tasks[0]])
            # Global Average Pooling
            if embeddings.dim() == 3:
                embeddings = embeddings.mean(dim=2)  # (B, C, T) -> (B, C)

            # CombinedSSLLossで両方のLossを計算
            total_loss, loss_dict = loss_fn(
                predictions=predictions,
                labels=labels_for_mtl,
                embeddings=embeddings,
                dataset_ids=datasets,
                activity_ids=activity_ids,
                body_parts=body_parts,
            )

        # 逆伝播
        if use_amp:
            scaler.scale(total_loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        # メーター更新
        batch_size = data.size(0)
        loss_meter.update(total_loss.item(), batch_size)
        if "mtl_total" in loss_dict:
            mtl_loss_meter.update(loss_dict["mtl_total"].item(), batch_size)
        if "hier_total" in loss_dict:
            hier_loss_meter.update(loss_dict["hier_total"].item(), batch_size)

        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "mtl": f"{mtl_loss_meter.avg:.4f}",
            "hier": f"{hier_loss_meter.avg:.4f}",
        })

    return {
        "loss": loss_meter.avg,
        "mtl_loss": mtl_loss_meter.avg,
        "hier_loss": hier_loss_meter.avg,
    }


def evaluate_combined_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    ssl_tasks: List[str],
    transforms: Dict[str, Any],
    apply_prob: float,
) -> Dict[str, float]:
    """MTL + 階層的SSL統合: 検証

    Args:
        model: IntegratedSSLModel
        loss_fn: CombinedSSLLoss
        dataloader: データローダー
        device: デバイス
        ssl_tasks: SSLタスクのリスト
        transforms: 拡張辞書
        apply_prob: binary_*タスクの適用確率

    Returns:
        平均損失の辞書
    """
    model.eval()
    loss_fn.eval()

    loss_meter = AverageMeter("Loss")
    mtl_loss_meter = AverageMeter("MTL")
    hier_loss_meter = AverageMeter("Hier")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            if batch is None:
                continue

            data = batch["data"].to(device)
            labels = batch["labels"]
            datasets = batch["datasets"]
            body_parts = batch["body_parts"]

            activity_ids = []
            for ds, label in zip(datasets, labels.tolist()):
                activity_name = get_activity_name(ds, label)
                activity_ids.append(activity_name)

            # MTL用の拡張を適用
            task_inputs_dict, labels_dict = apply_augmentations_batch(
                data, ssl_tasks, transforms, apply_prob, device
            )

            # MTLタスクの予測
            predictions = {}
            labels_for_mtl = {}
            for task in ssl_tasks:
                x_task = task_inputs_dict[task]

                if task.startswith("invariant_"):
                    z1 = model(x_task, task)
                    x_rotated = labels_dict[task]
                    z2 = model(x_rotated, task)
                    predictions[task] = z1
                    labels_for_mtl[task] = z2
                else:
                    pred = model(x_task, task)
                    predictions[task] = pred
                    labels_for_mtl[task] = labels_dict[task]

            # 階層的Loss用のembeddings
            embeddings = model.backbone(task_inputs_dict[ssl_tasks[0]])
            if embeddings.dim() == 3:
                embeddings = embeddings.mean(dim=2)

            # Loss計算
            total_loss, loss_dict = loss_fn(
                predictions=predictions,
                labels=labels_for_mtl,
                embeddings=embeddings,
                dataset_ids=datasets,
                activity_ids=activity_ids,
                body_parts=body_parts,
            )

            batch_size = data.size(0)
            loss_meter.update(total_loss.item(), batch_size)
            if "mtl_total" in loss_dict:
                mtl_loss_meter.update(loss_dict["mtl_total"].item(), batch_size)
            if "hier_total" in loss_dict:
                hier_loss_meter.update(loss_dict["hier_total"].item(), batch_size)

    return {
        "val_loss": loss_meter.avg,
        "val_mtl_loss": mtl_loss_meter.avg,
        "val_hier_loss": hier_loss_meter.avg,
    }


def run_combined_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    config: Dict[str, Any],
    experiment_dirs: ExperimentDirs,
    device: torch.device,
    use_wandb: bool,
    ssl_tasks: List[str],
    transforms: Dict[str, Any],
    logger,
) -> None:
    """MTL + 階層的SSL統合のトレーニングループ

    Args:
        model: モデル
        train_loader: トレーニングデータローダー
        val_loader: 検証データローダー
        criterion: CombinedSSLLoss
        optimizer: オプティマイザー
        scheduler: スケジューラー
        config: 設定辞書
        experiment_dirs: 実験ディレクトリ
        device: デバイス
        use_wandb: W&Bを使用するか
        ssl_tasks: SSLタスクのリスト
        transforms: 拡張辞書
        logger: ロガー
    """
    best_loss = float("inf")
    save_path = str(experiment_dirs.checkpoint)
    num_epochs = config.get("training", {}).get("epochs", 100)
    eval_interval = config.get("evaluation", {}).get("eval_interval", 1)
    save_freq = config.get("checkpoint", {}).get("save_freq", 10)

    # Mixed Precision
    use_amp = config.get("mixed_precision", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Gradient Clipping
    grad_clip_norm = config.get("training", {}).get("grad_clip_norm", None)

    # Apply prob
    multitask_config = config.get("multitask", {})
    apply_prob = multitask_config.get("apply_prob", 0.5)

    logger.info("=" * 80)
    logger.info("Starting Combined MTL + Hierarchical SSL training...")
    logger.info(f"SSL Tasks: {ssl_tasks}")
    logger.info("=" * 80)

    for epoch in range(1, num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{num_epochs}")

        # 学習
        train_metrics = train_combined_epoch(
            model, criterion, train_loader, optimizer, device, epoch,
            ssl_tasks, transforms, apply_prob, logger, scaler, grad_clip_norm
        )

        # 検証
        val_metrics = None
        if epoch % eval_interval == 0:
            val_metrics = evaluate_combined_epoch(
                model, criterion, val_loader, device,
                ssl_tasks, transforms, apply_prob
            )

        # ログ
        logger.info(
            f"Epoch {epoch} - Train: loss={train_metrics['loss']:.4f}, "
            f"mtl={train_metrics['mtl_loss']:.4f}, "
            f"hier={train_metrics['hier_loss']:.4f}"
        )

        if val_metrics is not None:
            logger.info(
                f"Epoch {epoch} - Val: loss={val_metrics['val_loss']:.4f}, "
                f"mtl={val_metrics['val_mtl_loss']:.4f}, "
                f"hier={val_metrics['val_hier_loss']:.4f}"
            )

        # W&Bログ
        if use_wandb and WANDB_AVAILABLE and wandb.run:
            log_dict = {
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/mtl_loss": train_metrics["mtl_loss"],
                "train/hier_loss": train_metrics["hier_loss"],
                "train/learning_rate": optimizer.param_groups[0]["lr"],
            }
            if val_metrics is not None:
                log_dict.update({
                    "val/loss": val_metrics["val_loss"],
                    "val/mtl_loss": val_metrics["val_mtl_loss"],
                    "val/hier_loss": val_metrics["val_hier_loss"],
                })
            wandb.log(log_dict, step=epoch)

        # ベストモデルの判定
        current_loss = val_metrics["val_loss"] if val_metrics is not None else train_metrics["loss"]

        # スケジューラー更新
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_loss)
            else:
                scheduler.step()

        # チェックポイント保存
        if current_loss < best_loss:
            best_loss = current_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss_fn_state_dict": criterion.state_dict(),
                "metrics": {"train": train_metrics, "val": val_metrics},
            }
            best_file = os.path.join(save_path, "best_model.pth")
            torch.save(checkpoint, best_file)
            logger.info(f"New best model saved (loss={best_loss:.4f})")

        # 定期保存
        if epoch % save_freq == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss_fn_state_dict": criterion.state_dict(),
                "metrics": {"train": train_metrics, "val": val_metrics},
            }
            filename = f"checkpoint_epoch_{epoch}.pth"
            save_file = os.path.join(save_path, filename)
            torch.save(checkpoint, save_file)
            logger.info(f"Checkpoint saved: {filename}")

    logger.info("=" * 80)
    logger.info("Combined MTL + Hierarchical SSL training completed!")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info("=" * 80)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_wandb: bool = False,
    multitask: bool = False,
    ssl_tasks: Optional[List[str]] = None,
    transforms: Optional[Dict[str, Any]] = None,
    apply_prob: float = 0.5,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_clip_norm: Optional[float] = None,
) -> Dict[str, float]:
    """1エポック分の学習を実行

    Args:
        model: 学習するモデル
        dataloader: データローダー
        criterion: 損失関数
        optimizer: オプティマイザー
        device: デバイス
        epoch: 現在のエポック
        use_wandb: W&Bへのログを有効化するか
        multitask: マルチタスク学習モードか
        ssl_tasks: SSLタスクのリスト
        transforms: 拡張辞書
        apply_prob: binary_*タスクの適用確率
        scaler: Mixed Precision用のGradScaler (Noneの場合は通常のfloat32学習)
        grad_clip_norm: 勾配クリッピングの最大ノルム (Noneの場合はクリッピングなし)

    Returns:
        平均損失の辞書
    """
    model.train()
    loss_meter = AverageMeter("Loss")

    # タスク数に応じて動的にメーターを作成
    task_loss_meters = None
    if multitask:
        num_tasks = len(model.ssl_tasks)
        task_loss_meters = [AverageMeter(f"Task{i+1}") for i in range(num_tasks)]

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch_data in enumerate(pbar):
        optimizer.zero_grad()

        # Mixed Precision Training
        use_amp = scaler is not None
        with torch.cuda.amp.autocast(enabled=use_amp):
            if multitask:
                # マルチタスク学習: データのみ受け取る
                x = batch_data
                total_loss, task_losses, x = process_multitask_batch(
                    x, model, criterion, ssl_tasks, transforms, apply_prob, device
                )
                loss = total_loss

                # タスク別損失を更新
                for i, task in enumerate(ssl_tasks):
                    task_loss_meters[i].update(task_losses[task].item(), x.size(0))
            else:
                # 通常のSSL: データを直接受け取る（統一されたデータローダー）
                x = batch_data.to(device)

                # 順伝播（同じデータを2回使う）
                z1, z2 = model(x, x)
                loss = criterion(z1, z2)

        # 逆伝播
        if use_amp:
            scaler.scale(loss).backward()

            # Gradient Clipping (AMPの場合はunscaleしてからクリップ)
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            # Gradient Clipping
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()

        # 統計を更新
        loss_meter.update(loss.item(), x.size(0))

        # プログレスバーを更新
        if multitask:
            pbar.set_postfix(
                {
                    "loss": f"{loss_meter.avg:.4f}",
                    **{
                        meter.name: f"{meter.avg:.4f}"
                        for meter in task_loss_meters[: len(model.ssl_tasks)]
                    },
                }
            )
        else:
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

        # バッチごとのログは不要（ストレージとログ転送時間の削減）
        pass

    # 結果を返す
    if multitask:
        result = {"loss": loss_meter.avg}
        for i, meter in enumerate(task_loss_meters[: len(model.ssl_tasks)]):
            result[f"task{i+1}_loss"] = meter.avg
        return result
    else:
        return {"loss": loss_meter.avg}


def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    multitask: bool = False,
    ssl_tasks: Optional[List[str]] = None,
    transforms: Optional[Dict[str, Any]] = None,
    apply_prob: float = 0.5,
) -> Dict[str, float]:
    """1エポック分の検証を実行

    Args:
        model: 評価するモデル
        dataloader: データローダー
        criterion: 損失関数
        device: デバイス
        multitask: マルチタスク学習モードか
        ssl_tasks: SSLタスクのリスト
        transforms: 拡張辞書
        apply_prob: binary_*タスクの適用確率

    Returns:
        平均損失の辞書
    """
    model.eval()
    loss_meter = AverageMeter("Loss")

    # タスク数に応じて動的にメーターを作成
    task_loss_meters = None
    if multitask:
        num_tasks = len(model.ssl_tasks)
        task_loss_meters = [AverageMeter(f"Task{i+1}") for i in range(num_tasks)]

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Validation"):
            if multitask:
                # マルチタスク学習: データのみ受け取る
                x = batch_data
                total_loss, task_losses, x = process_multitask_batch(
                    x, model, criterion, ssl_tasks, transforms, apply_prob, device
                )
                loss = total_loss

                # タスク別損失を更新
                for i, task in enumerate(ssl_tasks):
                    task_loss_meters[i].update(task_losses[task].item(), x.size(0))
            else:
                # 通常のSSL: データを直接受け取る（統一されたデータローダー）
                x = batch_data.to(device)

                # 順伝播（同じデータを2回使う）
                z1, z2 = model(x, x)
                loss = criterion(z1, z2)

            # 統計を更新
            loss_meter.update(loss.item(), x.size(0))

    # 結果を返す
    if multitask:
        result = {"loss": loss_meter.avg}
        for i, meter in enumerate(task_loss_meters[: len(model.ssl_tasks)]):
            result[f"task{i+1}_loss"] = meter.avg
        return result
    else:
        return {"loss": loss_meter.avg}


def create_model(
    config: Dict[str, Any],
    in_channels: int,
    sequence_length: int,
    use_multitask: bool,
    multitask_config: Dict[str, Any],
    device: torch.device,
    logger,
) -> nn.Module:
    """モデルを作成

    Args:
        config: 設定辞書
        in_channels: 入力チャネル数
        sequence_length: 入力系列長
        use_multitask: マルチタスク学習を使用するか
        multitask_config: マルチタスク設定
        device: デバイス
        logger: ロガー

    Returns:
        作成されたモデル
    """
    if use_multitask:
        # 統合型SSLモデル
        ssl_tasks = get_ssl_tasks_from_config(config)
        hidden_dim = config["model"].get("feature_dim", 256)
        backbone_name = config["model"].get("backbone", "resnet")

        # バックボーンを作成
        if backbone_name == "resnet":
            from src.models.backbones import Resnet
            # ウィンドウサイズに応じて適切なアーキテクチャを自動選択
            nano_window = sequence_length < 20   # 15サンプル用
            micro_window = 20 <= sequence_length < 100  # 30, 60サンプル用
            backbone = Resnet(n_channels=in_channels, foundationUK=False,
                            micro_window=micro_window, nano_window=nano_window)
        elif backbone_name == "resnet1d":
            from src.models.backbones import ResNet1D
            backbone = ResNet1D(in_channels, num_classes=hidden_dim, dropout=0.0)
            backbone.fc = nn.Identity()  # Remove final FC layer
        elif backbone_name == "simple_cnn":
            from src.models.backbones import SimpleCNN
            backbone = SimpleCNN(in_channels, num_classes=hidden_dim, dropout=0.0)
            backbone.fc = nn.Identity()  # Remove final FC layer
        elif backbone_name == "deepconvlstm":
            from src.models.backbones import DeepConvLSTM
            backbone = DeepConvLSTM(in_channels, num_classes=hidden_dim, dropout=0.0)
            backbone.fc = nn.Identity()  # Remove final FC layer
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        # 統合モデルを作成（GPU上で直接構築）
        model = IntegratedSSLModel(
            backbone=backbone,
            ssl_tasks=ssl_tasks,
            hidden_dim=hidden_dim,
            n_channels=in_channels,
            sequence_length=sequence_length,
            device=device,
        )

        # 事前学習済みモデルをロード（オプション）
        pretrained_path = config["model"].get("pretrained_path", None)
        if pretrained_path:
            logger.info(f"Loading pretrained model from: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=device)

            # checkpoint形式を確認
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                epoch = checkpoint.get("epoch", "unknown")
                logger.info(f"Loading from checkpoint epoch: {epoch}")
            else:
                state_dict = checkpoint

            # モデルにロード
            model.load_state_dict(state_dict, strict=True)
            logger.info("Pretrained model loaded successfully")

        logger.info(f"SSL tasks: {ssl_tasks}")
        logger.info(f"Backbone: {backbone_name}")
    else:
        # 通常のセンサーSSLモデル（現在は未サポート）
        raise NotImplementedError("Non-multitask SSL model is not yet implemented")

    param_info = count_parameters(model)
    logger.info(
        f"Model created: {config['model']['backbone']}, "
        f"Total params: {param_info['total']:,}, "
        f"Trainable: {param_info['trainable']:,}"
    )

    return model


class HierarchicalSSLModel(nn.Module):
    """
    階層的SSL用モデル

    エンコーダー + プロジェクションヘッド
    """

    def __init__(
        self,
        in_channels: int = 3,
        sequence_length: int = 150,
        embed_dim: int = 512,
    ):
        super().__init__()

        # バックボーンエンコーダー
        self.encoder = Resnet(n_channels=in_channels)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # エンコーダーの出力チャンネル数を取得
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, sequence_length)
            enc_out = self.encoder(dummy)  # (1, channels, time)
            encoder_dim = enc_out.shape[1]  # チャンネル数

        # プロジェクションヘッド
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time_steps)

        Returns:
            embeddings: (batch, embed_dim)
        """
        features = self.encoder(x)  # (batch, encoder_channels, time)
        features = self.gap(features)  # (batch, encoder_channels, 1)
        features = features.squeeze(-1)  # (batch, encoder_channels)
        embeddings = self.projector(features)  # (batch, embed_dim)
        return embeddings

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """エンコーダー出力のみ（プロジェクションなし）"""
        features = self.encoder(x)  # (batch, encoder_channels, time)
        features = self.gap(features)  # (batch, encoder_channels, 1)
        return features.squeeze(-1)  # (batch, encoder_channels)


def create_hierarchical_model(
    config: Dict[str, Any],
    in_channels: int,
    sequence_length: int,
    device: torch.device,
    logger,
) -> nn.Module:
    """階層的SSL用モデルを作成

    Args:
        config: 設定辞書
        in_channels: 入力チャネル数
        sequence_length: 入力系列長
        device: デバイス
        logger: ロガー

    Returns:
        作成されたモデル
    """
    model_config = config.get("model", {})
    embed_dim = model_config.get("embed_dim", 512)

    model = HierarchicalSSLModel(
        in_channels=in_channels,
        sequence_length=sequence_length,
        embed_dim=embed_dim,
    ).to(device)

    param_info = count_parameters(model)
    logger.info(
        f"Hierarchical SSL Model created: "
        f"Total params: {param_info['total']:,}, "
        f"Trainable: {param_info['trainable']:,}"
    )

    return model


def create_hierarchical_criterion(
    config: Dict[str, Any],
    device: torch.device,
    logger,
) -> nn.Module:
    """階層的SSL用損失関数を作成

    Args:
        config: 設定辞書
        device: デバイス
        logger: ロガー

    Returns:
        損失関数
    """
    hierarchical_config = config.get("hierarchical", {})
    model_config = config.get("model", {})
    loss_config = config.get("loss", {})

    atlas_path = hierarchical_config.get("atlas_path", "docs/atlas/activity_mapping.json")
    embed_dim = model_config.get("embed_dim", 512)
    prototype_dim = loss_config.get("prototype_dim", 128)
    temperature = loss_config.get("temperature", 0.1)
    lambda_activity = loss_config.get("lambda_activity", 0.5)
    lambda_prototype = loss_config.get("lambda_prototype", 0.5)

    criterion = HierarchicalSSLLoss(
        atlas_path=atlas_path,
        embed_dim=embed_dim,
        prototype_dim=prototype_dim,
        temperature=temperature,
        lambda_activity=lambda_activity,
        lambda_prototype=lambda_prototype,
    ).to(device)

    logger.info(f"Hierarchical SSL Loss created:")
    logger.info(f"  - prototype_dim: {prototype_dim}")
    logger.info(f"  - temperature: {temperature}")
    logger.info(f"  - lambda_activity: {lambda_activity}")
    logger.info(f"  - lambda_prototype: {lambda_prototype}")

    return criterion


def create_criterion(
    use_multitask: bool, multitask_config: Dict[str, Any], logger
) -> nn.Module:
    """損失関数を作成

    Args:
        use_multitask: マルチタスク学習を使用するか
        multitask_config: マルチタスク設定
        logger: ロガー

    Returns:
        損失関数
    """
    if use_multitask:
        # 統合型SSL損失関数
        ssl_tasks = multitask_config.get("ssl_tasks", DEFAULT_SSL_TASKS)
        task_weights_list = multitask_config.get("task_weights", DEFAULT_TASK_WEIGHTS)
        task_weights = {
            task: weight for task, weight in zip(ssl_tasks, task_weights_list)
        }

        criterion = IntegratedSSLLoss(ssl_tasks=ssl_tasks, task_weights=task_weights)
        logger.info(f"Task weights: {task_weights}")
    else:
        # 通常のSSL損失（現在は未サポート）
        raise NotImplementedError("Non-multitask SSL loss is not yet implemented")

    return criterion


def log_epoch_metrics(
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Optional[Dict[str, float]],
    use_multitask: bool,
    multitask_config: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    use_wandb: bool,
    logger,
) -> None:
    """エポックメトリクスをログ

    Args:
        epoch: エポック数
        train_metrics: トレーニングメトリクス
        val_metrics: 検証メトリクス（Noneの場合は検証を実行していない）
        use_multitask: マルチタスク学習を使用するか
        multitask_config: マルチタスク設定
        optimizer: オプティマイザー
        use_wandb: W&Bを使用するか
        logger: ロガー
    """
    train_loss = train_metrics["loss"]

    if use_multitask:
        ssl_tasks = get_ssl_tasks_from_config({"multitask": multitask_config})
        task_losses_str = ", ".join(
            [
                f"{task}: {train_metrics[f'task{i+1}_loss']:.4f}"
                for i, task in enumerate(ssl_tasks)
            ]
        )
        logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, {task_losses_str}")

        # 検証メトリクスをログ
        if val_metrics is not None:
            val_loss = val_metrics["loss"]
            val_task_losses_str = ", ".join(
                [
                    f"{task}: {val_metrics[f'task{i+1}_loss']:.4f}"
                    for i, task in enumerate(ssl_tasks)
                ]
            )
            logger.info(f"Epoch {epoch} - Val Loss: {val_loss:.4f}, {val_task_losses_str}")
    else:
        logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")
        if val_metrics is not None:
            logger.info(f"Epoch {epoch} - Val Loss: {val_metrics['loss']:.4f}")

    # W&Bにログ
    if use_wandb:
        log_dict = {
            "train/epoch_loss": train_loss,
            "train/learning_rate": optimizer.param_groups[0]["lr"],
        }
        if use_multitask:
            ssl_tasks = get_ssl_tasks_from_config({"multitask": multitask_config})
            for i, task in enumerate(ssl_tasks):
                log_dict[f"train/epoch_{task}_loss"] = train_metrics[f"task{i+1}_loss"]

        # 検証メトリクスを追加
        if val_metrics is not None:
            log_dict["val/epoch_loss"] = val_metrics["loss"]
            if use_multitask:
                for i, task in enumerate(ssl_tasks):
                    log_dict[f"val/epoch_{task}_loss"] = val_metrics[f"task{i+1}_loss"]

        wandb.log(log_dict, step=epoch)


def run_hierarchical_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    config: Dict[str, Any],
    experiment_dirs: ExperimentDirs,
    device: torch.device,
    use_wandb: bool,
    logger,
) -> None:
    """階層的SSLのトレーニングループを実行

    Args:
        model: モデル
        train_loader: トレーニングデータローダー
        val_loader: 検証データローダー
        criterion: HierarchicalSSLLoss
        optimizer: オプティマイザー
        scheduler: スケジューラー
        config: 設定辞書
        experiment_dirs: 実験ディレクトリ
        device: デバイス
        use_wandb: W&Bを使用するか
        logger: ロガー
    """
    best_loss = float("inf")
    save_path = str(experiment_dirs.checkpoint)
    num_epochs = config.get("training", {}).get("num_epochs", 100)
    # training.epochs も確認（互換性のため）
    if "epochs" in config.get("training", {}):
        num_epochs = config["training"]["epochs"]
    eval_interval = config.get("evaluation", {}).get("eval_interval", 1)
    save_freq = config.get("training", {}).get("save_every", 10)

    logger.info("=" * 80)
    logger.info("Starting Hierarchical SSL training...")
    logger.info("=" * 80)

    for epoch in range(1, num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{num_epochs}")

        # 学習
        train_metrics = train_hierarchical_epoch(
            model, criterion, train_loader, optimizer, device, epoch, logger
        )

        # 検証
        val_metrics = None
        if epoch % eval_interval == 0:
            val_metrics = evaluate_hierarchical_epoch(
                model, criterion, val_loader, device
            )

        # ログ
        train_loss = train_metrics["loss"]
        logger.info(
            f"Epoch {epoch} - Train: loss={train_loss:.4f}, "
            f"activity={train_metrics['activity_loss']:.4f}, "
            f"prototype={train_metrics['prototype_loss']:.4f}"
        )

        if val_metrics is not None:
            val_loss = val_metrics["val_loss"]
            logger.info(
                f"Epoch {epoch} - Val: loss={val_loss:.4f}, "
                f"activity={val_metrics['val_activity_loss']:.4f}, "
                f"prototype={val_metrics['val_prototype_loss']:.4f}"
            )

        # W&Bログ
        if use_wandb and WANDB_AVAILABLE and wandb.run:
            log_dict = {
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/activity_loss": train_metrics["activity_loss"],
                "train/prototype_loss": train_metrics["prototype_loss"],
                "train/learning_rate": optimizer.param_groups[0]["lr"],
            }
            if val_metrics is not None:
                log_dict.update({
                    "val/loss": val_metrics["val_loss"],
                    "val/activity_loss": val_metrics["val_activity_loss"],
                    "val/prototype_loss": val_metrics["val_prototype_loss"],
                })
            wandb.log(log_dict, step=epoch)

        # ベストモデルの判定
        current_loss = val_metrics["val_loss"] if val_metrics is not None else train_loss

        # スケジューラー更新
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_loss)
            else:
                scheduler.step()

        # チェックポイント保存
        if current_loss < best_loss:
            best_loss = current_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss_fn_state_dict": criterion.state_dict(),
                "metrics": {"train": train_metrics, "val": val_metrics},
            }
            best_file = os.path.join(save_path, "best_model.pth")
            torch.save(checkpoint, best_file)
            logger.info(f"New best model saved (loss={best_loss:.4f})")

        # 定期保存
        if epoch % save_freq == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss_fn_state_dict": criterion.state_dict(),
                "metrics": {"train": train_metrics, "val": val_metrics},
            }
            filename = f"checkpoint_epoch_{epoch}.pth"
            save_file = os.path.join(save_path, filename)
            torch.save(checkpoint, save_file)
            logger.info(f"Checkpoint saved: {filename}")

    logger.info("=" * 80)
    logger.info("Hierarchical SSL training completed!")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info("=" * 80)


def run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    config: Dict[str, Any],
    experiment_dirs: ExperimentDirs,
    device: torch.device,
    use_wandb: bool,
    use_multitask: bool,
    multitask_config: Dict[str, Any],
    ssl_tasks: List[str],
    transforms: Dict[str, Any],
    logger,
) -> None:
    """トレーニングループを実行

    Args:
        model: モデル
        train_loader: トレーニングデータローダー
        val_loader: 検証データローダー
        criterion: 損失関数
        optimizer: オプティマイザー
        scheduler: スケジューラー
        config: 設定辞書
        experiment_dirs: 実験ディレクトリ
        device: デバイス
        use_wandb: W&Bを使用するか
        use_multitask: マルチタスク学習を使用するか
        multitask_config: マルチタスク設定
        ssl_tasks: SSLタスクのリスト
        transforms: 拡張辞書
        logger: ロガー
    """
    best_loss = float("inf")
    save_path = str(experiment_dirs.checkpoint)
    num_epochs = config["training"]["epochs"]
    eval_interval = config.get("evaluation", {}).get("eval_interval", 1)
    save_freq = config["checkpoint"].get("save_freq", 10)

    # Mixed Precision Training
    use_amp = config.get("mixed_precision", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        logger.info("Mixed Precision Training: Enabled")
    else:
        logger.info("Mixed Precision Training: Disabled")

    # Gradient Clipping
    grad_clip_norm = config["training"].get("grad_clip_norm", None)
    if grad_clip_norm is not None:
        logger.info(f"Gradient Clipping: Enabled (max_norm={grad_clip_norm})")
    else:
        logger.info("Gradient Clipping: Disabled")

    # Early Stopping
    early_stopping_config = config.get("early_stopping", {})
    use_early_stopping = early_stopping_config.get("enabled", False)
    if use_early_stopping:
        from src.utils.training import EarlyStopping
        early_stopping = EarlyStopping(
            patience=early_stopping_config.get("patience", 10),
            min_delta=early_stopping_config.get("min_delta", 0.0),
            mode="min"
        )
        logger.info(f"Early Stopping: Enabled (patience={early_stopping.patience}, min_delta={early_stopping.min_delta})")
    else:
        early_stopping = None
        logger.info("Early Stopping: Disabled")

    # save_freq期間内での最良モデルを追跡
    window_best_loss = float("inf")
    window_best_state = None

    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)

    for epoch in range(1, num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{num_epochs}")

        # 学習
        apply_prob = multitask_config.get("apply_prob", DEFAULT_APPLY_PROB)
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            use_wandb,
            use_multitask,
            ssl_tasks,
            transforms,
            apply_prob,
            scaler,
            grad_clip_norm,
        )

        # 検証
        val_metrics = None
        if epoch % eval_interval == 0:
            val_metrics = evaluate_epoch(
                model,
                val_loader,
                criterion,
                device,
                use_multitask,
                ssl_tasks,
                transforms,
                apply_prob,
            )

        # メトリクスをログ
        log_epoch_metrics(
            epoch,
            train_metrics,
            val_metrics,
            use_multitask,
            multitask_config,
            optimizer,
            use_wandb,
            logger,
        )

        # ベストモデルの判定にはval_lossを優先、なければtrain_lossを使用
        current_loss = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]

        # 全体のベストモデルを更新
        if current_loss < best_loss:
            best_loss = current_loss
            loss_type = "val" if val_metrics is not None else "train"
            logger.info(f"New best {loss_type} loss: {best_loss:.4f}")

        # Early Stopping チェック
        if early_stopping is not None:
            if early_stopping(current_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                logger.info(f"Best loss: {best_loss:.4f}")
                break

        # save_freq期間内でのベストモデルを追跡
        if current_loss < window_best_loss:
            window_best_loss = current_loss
            # メトリクスにval_metricsも含める
            metrics_to_save = {"train": train_metrics}
            if val_metrics is not None:
                metrics_to_save["val"] = val_metrics

            # ベストモデルの状態を保存（メモリ内）
            window_best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict().copy(),
                "optimizer_state_dict": optimizer.state_dict().copy(),
                "metrics": metrics_to_save,
            }

        # 学習率を更新
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # val_metricsがあればそれを使用、なければtrain_metricsを使用
                metric_for_scheduler = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
                scheduler.step(metric_for_scheduler)
            else:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(f"Learning rate: {current_lr:.6f}")

        # save_freq期間ごと、または最終エポックでチェックポイントを保存
        if epoch % save_freq == 0 or epoch == num_epochs:
            if window_best_state is not None:
                # save_freq期間内で最良だったエポックのチェックポイントを保存
                best_epoch = window_best_state["epoch"]
                is_best = window_best_loss <= best_loss

                # チェックポイントを保存
                checkpoint = {
                    "epoch": window_best_state["epoch"],
                    "model_state_dict": window_best_state["model_state_dict"],
                    "optimizer_state_dict": window_best_state["optimizer_state_dict"],
                    "metrics": window_best_state["metrics"],
                }

                filename = f"checkpoint_epoch_{best_epoch}.pth"
                save_file = os.path.join(save_path, filename)
                torch.save(checkpoint, save_file)

                logger.info(f"Checkpoint saved: {filename} (best in epochs {epoch-save_freq+1}-{epoch}, loss={window_best_loss:.4f})")

                # ベストモデルとして保存
                if is_best:
                    best_file = os.path.join(save_path, "best_model.pth")
                    torch.save(checkpoint, best_file)
                    logger.info(f"Updated best model: {best_file}")

                # 次のウィンドウのためにリセット
                window_best_loss = float("inf")
                window_best_state = None

    # 完了
    logger.info("=" * 80)
    logger.info("Pre-training completed!")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info("=" * 80)


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
    validate_config(config, mode="pretrain")

    # 実験ディレクトリを判定・作成
    # run_experiments.py から呼ばれた場合: 設定ファイルが実験ディレクトリ内にある
    # 直接呼ばれた場合: configs/ 内の設定ファイルを使用
    config_path = Path(args.config)

    if config_path.parent.name != "configs" and (config_path.parent / "config.yaml").exists():
        # run_experiments.py から呼ばれた場合
        # 設定ファイルが実験ディレクトリ内にあるので、そのディレクトリを使用
        experiment_root = config_path.parent

        # checkpoint.save_path がある場合はそれを使用
        if "checkpoint" in config and "save_path" in config["checkpoint"]:
            checkpoint_dir = Path(config["checkpoint"]["save_path"])
        else:
            checkpoint_dir = experiment_root / "models"

        experiment_dirs = ExperimentDirs(
            root=experiment_root,
            checkpoint=checkpoint_dir,
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    else:
        # 直接呼ばれた場合
        # 新しい実験ディレクトリを作成
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dirs = ExperimentDirs.create(Path("experiments"), run_id)

        # 設定ファイルを実験ディレクトリにコピー
        config_copy_path = experiment_dirs.root / "config.yaml"
        shutil.copy(args.config, config_copy_path)

    # ロガーをセットアップ（ファイルとコンソール両方に出力）
    logger = setup_logger("pretrain", log_dir=str(experiment_dirs.root), log_file="experiment.log")
    logger.info(f"Experiment directory: {experiment_dirs.root}")
    logger.info(f"Configuration loaded from {args.config}")
    logger.info(f"Starting pre-training with config: {config['model']['name']}")

    # シード設定
    set_seed(config["seed"])
    logger.info(f"Random seed set to {config['seed']}")

    # デバイス設定
    device = get_device(config["device"])
    logger.info(f"Using device: {device}")

    # 階層的SSLの設定確認
    hierarchical_config = config.get("hierarchical", {})
    use_hierarchical = hierarchical_config.get("enabled", False)

    # マルチタスク学習の設定
    multitask_config = config.get("multitask", {})
    use_multitask = multitask_config.get("enabled", False)

    # 統合モード: hierarchical + multitask両方有効
    use_combined = use_hierarchical and use_multitask

    if use_combined:
        logger.info("Mode: Combined MTL + Hierarchical SSL")
    elif use_hierarchical:
        logger.info("Mode: Hierarchical SSL (Activity + Prototype learning)")
    elif use_multitask:
        logger.info("Mode: Multitask SSL")
    else:
        logger.info("Mode: Standard SSL")

    # ===============================
    # 統合モード（MTL + 階層的SSL）
    # ===============================
    if use_combined:
        # データセットとデータローダーを作成（階層的データセットを使用）
        try:
            train_loader, val_loader, test_loader, in_channels, sequence_length, atlas = (
                setup_hierarchical_dataloaders(config, logger)
            )
            logger.info(f"Number of batches: {len(train_loader)}")
        except Exception as e:
            logger.error(f"Failed to create hierarchical dataset: {e}")
            raise

        # SSLタスクを取得
        ssl_tasks = get_ssl_tasks_from_config(config)
        specific_transforms = create_augmentation_transforms(ssl_tasks, config)

        # モデルを作成（IntegratedSSLModel - MTL用）
        model = create_model(
            config, in_channels, sequence_length, True, multitask_config, device, logger
        )

        # CombinedSSLLossを作成
        loss_config = config.get("loss", {})
        criterion = CombinedSSLLoss(
            ssl_tasks=ssl_tasks,
            task_weights={task: weight for task, weight in zip(
                ssl_tasks, multitask_config.get("task_weights", [1.0] * len(ssl_tasks))
            )},
            atlas_path=hierarchical_config.get("atlas_path", "docs/atlas/activity_mapping.json"),
            embed_dim=config.get("model", {}).get("feature_dim", 256),
            prototype_dim=loss_config.get("prototype_dim", 128),
            temperature=loss_config.get("temperature", 0.1),
            lambda_complex=loss_config.get("lambda_complex", 0.1),
            lambda_activity=loss_config.get("lambda_activity", 0.3),
            lambda_atomic=loss_config.get("lambda_atomic", 0.6),
            lambda_mtl=loss_config.get("lambda_mtl", 1.0),
            lambda_hierarchical=loss_config.get("lambda_hierarchical", 1.0),
        ).to(device)

        logger.info(f"CombinedSSLLoss created:")
        logger.info(f"  - SSL tasks: {ssl_tasks}")
        logger.info(f"  - lambda_mtl: {loss_config.get('lambda_mtl', 1.0)}")
        logger.info(f"  - lambda_hierarchical: {loss_config.get('lambda_hierarchical', 1.0)}")

        # オプティマイザーを作成（モデル + Loss関数のパラメータ）
        all_params = list(model.parameters()) + list(criterion.parameters())
        optimizer_name = config["training"].get("optimizer", "adam").lower()
        learning_rate = config["training"]["learning_rate"]
        weight_decay = config["training"].get("weight_decay", 0.0)

        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(all_params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(all_params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(all_params, lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # スケジューラーを作成
        scheduler = get_scheduler(
            optimizer=optimizer,
            scheduler_name=config["training"].get("scheduler", "cosine"),
            total_epochs=config["training"]["epochs"],
            warmup_epochs=config["training"].get("warmup_epochs", 0),
            T_max=config["training"]["epochs"],
        )

        # W&Bを初期化
        use_wandb = init_wandb(config, model)

        # 統合トレーニングループを実行
        run_combined_training_loop(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            config,
            experiment_dirs,
            device,
            use_wandb,
            ssl_tasks,
            specific_transforms,
            logger,
        )

    # ===============================
    # 階層的SSLモード（単体）
    # ===============================
    elif use_hierarchical:
        # データセットとデータローダーを作成
        try:
            train_loader, val_loader, test_loader, in_channels, sequence_length, atlas = (
                setup_hierarchical_dataloaders(config, logger)
            )
            logger.info(f"Number of batches: {len(train_loader)}")
        except Exception as e:
            logger.error(f"Failed to create hierarchical dataset: {e}")
            raise

        # モデルを作成
        model = create_hierarchical_model(
            config, in_channels, sequence_length, device, logger
        )

        # 損失関数を作成
        criterion = create_hierarchical_criterion(config, device, logger)

        # オプティマイザーを作成（損失関数のパラメータも含む）
        optimizer = get_optimizer(
            model=list(model.parameters()) + list(criterion.parameters()),
            optimizer_name=config.get("training", {}).get("optimizer", "adam"),
            learning_rate=config.get("training", {}).get("learning_rate", 0.001),
            weight_decay=config.get("training", {}).get("weight_decay", 0.0001),
        )

        # スケジューラーを作成
        num_epochs = config.get("training", {}).get("num_epochs", 100)
        if "epochs" in config.get("training", {}):
            num_epochs = config["training"]["epochs"]

        scheduler = get_scheduler(
            optimizer=optimizer,
            scheduler_name=config.get("training", {}).get("scheduler", "cosine"),
            total_epochs=num_epochs,
            warmup_epochs=config.get("training", {}).get("warmup_epochs", 0),
            T_max=num_epochs,
        )

        # W&Bを初期化
        use_wandb = init_wandb(config, model)

        # トレーニングループを実行
        run_hierarchical_training_loop(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            config,
            experiment_dirs,
            device,
            use_wandb,
            logger,
        )

    # ===============================
    # マルチタスクSSLモード（従来）
    # ===============================
    else:
        # データセットとデータローダーを作成
        try:
            train_loader, val_loader, test_loader, in_channels, sequence_length = (
                setup_batch_dataloaders(config, logger, use_multitask, multitask_config)
            )
            logger.info(f"Number of batches: {len(train_loader)}")
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            raise

        # モデルを作成
        model = create_model(
            config, in_channels, sequence_length, use_multitask, multitask_config, device, logger
        )

        # 損失関数を作成
        criterion = create_criterion(use_multitask, multitask_config, logger)

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
            scheduler_name=config["training"].get("scheduler", "cosine"),
            total_epochs=config["training"]["epochs"],
            warmup_epochs=config["training"].get("warmup_epochs", 0),
            T_max=config["training"]["epochs"],
        )

        # W&Bを初期化
        use_wandb = init_wandb(config, model)

        # 拡張とタスクを準備
        ssl_tasks = []
        specific_transforms = {}
        if use_multitask:
            ssl_tasks = get_ssl_tasks_from_config(config)
            specific_transforms = create_augmentation_transforms(ssl_tasks, config)

        # トレーニングループを実行
        run_training_loop(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            config,
            experiment_dirs,
            device,
            use_wandb,
            use_multitask,
            multitask_config,
            ssl_tasks,
            specific_transforms,
            logger,
        )

    # クリーンアップ
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Self-Supervised Pre-training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain.yaml",
        help="Path to configuration file",
    )

    args = parser.parse_args()
    main(args)
