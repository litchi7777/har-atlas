"""
Self-Supervised Pre-training スクリプト

SSL手法（SimCLR、MoCo等）を用いた事前学習を実行します。
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
from src.data.batch_dataset import MultiTaskSubjectWiseLoader, SubjectWiseLoader
from src.losses import IntegratedSSLLoss
from src.models.sensor_models import IntegratedSSLModel, Resnet
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
DEFAULT_N_SEGMENTS = 4
DEFAULT_TIMEWARP_SIGMA = 0.2
DEFAULT_TIMEWARP_KNOT = 4
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
    log: Path

    @classmethod
    def create(cls, base_dir: Path, run_id: str) -> "ExperimentDirs":
        """実験ディレクトリを作成"""
        root = base_dir / "pretrain" / f"run_{run_id}"
        checkpoint = root / "checkpoints"
        log = root / "logs"

        root.mkdir(parents=True, exist_ok=True)
        checkpoint.mkdir(exist_ok=True)
        log.mkdir(exist_ok=True)

        return cls(root=root, checkpoint=checkpoint, log=log)


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
        if aug_name == "permute":
            transforms["permute"] = Permutation(n_segments=DEFAULT_N_SEGMENTS)
        elif aug_name == "reverse":
            transforms["reverse"] = Reverse()
        elif aug_name == "timewarp":
            transforms["timewarp"] = TimeWarping(
                sigma=DEFAULT_TIMEWARP_SIGMA, knot=DEFAULT_TIMEWARP_KNOT
            )

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

    return transforms


def collect_data_paths(
    data_root: str, datasets: List[str], exclude_patterns: List[str], logger
) -> List[str]:
    """データセットからファイルパスを収集

    Args:
        data_root: データルートディレクトリ
        datasets: データセット名のリスト
        exclude_patterns: 除外パターンのリスト
        logger: ロガー

    Returns:
        収集されたファイルパスのリスト

    Raises:
        ValueError: データセットが空、またはファイルが見つからない場合
    """
    if not datasets:
        raise ValueError("datasets list is empty in sensor_data config")

    # 各データセットからパスパターンを自動生成して収集
    # データ構造: {dataset}/{USER}/{Device}/{Sensor}/X.npy
    # ACCセンサーのみを使用（チャネル数を統一するため）
    all_paths = []
    for dataset_name in datasets:
        pattern = f"{data_root}/{dataset_name}/*/*/ACC/X.npy"
        paths = sorted(glob.glob(pattern))
        logger.info(
            f"Dataset '{dataset_name}' (ACC only): {pattern} -> {len(paths)} files"
        )
        all_paths.extend(paths)

    # 除外パターンのフィルタリング
    if exclude_patterns:
        all_paths = [
            p
            for p in all_paths
            if not any(exclude in p for exclude in exclude_patterns)
        ]

    if len(all_paths) == 0:
        raise ValueError("No data files found")

    logger.info(f"Total: {len(all_paths)} files")

    return all_paths


def create_data_loaders(
    paths: List[str],
    config: DataLoaderConfig,
    use_multitask: bool,
    ssl_tasks: List[str],
    specific_transforms: Optional[Dict[str, Any]] = None,
    apply_prob: float = DEFAULT_APPLY_PROB,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """データローダーを作成

    Args:
        paths: データファイルパスのリスト
        config: データローダー設定
        use_multitask: マルチタスク学習を使用するか
        ssl_tasks: SSLタスクのリスト
        specific_transforms: 拡張変換の辞書
        apply_prob: 拡張適用確率

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # データセットを作成
    if use_multitask:
        dataset = MultiTaskSubjectWiseLoader(
            paths,
            sample_threshold=config.sample_threshold,
            ssl_tasks=ssl_tasks,
            specific_transforms=specific_transforms or {},
            apply_prob=apply_prob,
        )
    else:
        dataset = SubjectWiseLoader(paths, sample_threshold=config.sample_threshold)

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
    sensor_config = config["sensor_data"]
    batch_loader_config = sensor_config["batch_loader"]

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
        data_root=sensor_config.get("data_root", DEFAULT_DATA_ROOT),
        datasets=sensor_config.get("datasets", []),
        exclude_patterns=batch_loader_config.get("exclude_patterns", []),
        sample_threshold=batch_loader_config["sample_threshold"],
        train_num_samples=batch_loader_config.get("train_num_samples", 1000),
        val_num_samples=batch_loader_config.get("val_num_samples", 100),
        test_num_samples=batch_loader_config.get("test_num_samples", 100),
        batch_size=batch_loader_config.get("batch_size", 4),
    )

    # データパスを収集
    all_paths = collect_data_paths(
        loader_config.data_root,
        loader_config.datasets,
        loader_config.exclude_patterns,
        logger,
    )

    # データローダーを作成
    apply_prob = multitask_config.get("apply_prob", DEFAULT_APPLY_PROB)
    train_loader, val_loader, test_loader = create_data_loaders(
        all_paths,
        loader_config,
        use_multitask,
        ssl_tasks,
        specific_transforms,
        apply_prob,
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
        transforms: 拡張辞書（'base_augmentation'キーに通常の拡張パイプラインを含む）
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
    x_base = x
    if "base_augmentation" in transforms:
        base_aug = transforms["base_augmentation"]
        # GPU上で直接適用
        augmented_batch = []
        for sample in x:
            augmented_sample = base_aug(sample)
            augmented_batch.append(augmented_sample)
        x_base = torch.stack(augmented_batch)

    # Binary拡張タスク - 各タスクで独立した拡張を適用
    binary_tasks = [t for t in ssl_tasks if t.startswith("binary_")]
    for task in binary_tasks:
        aug_name = task.replace("binary_", "")

        # ランダムに適用するか決定
        apply = np.random.random() < apply_prob

        if apply and aug_name in transforms:
            # ベースデータをコピーして、このタスク専用の拡張を適用
            x_task = x_base.clone()
            # CPU上でnumpy配列として拡張を適用
            x_np = x_task.cpu().numpy()
            augmented = np.stack([transforms[aug_name](sample) for sample in x_np])
            task_inputs_dict[task] = torch.from_numpy(augmented).float()
        else:
            # 拡張なしの場合はベースデータをそのまま使用
            task_inputs_dict[task] = x_base.clone()

        # ラベル: 全サンプルで同じ（バッチ全体に同じ拡張を適用）
        labels_dict[task] = torch.full((batch_size,), 1 if apply else 0, dtype=torch.long)

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

    # デバイスに移動
    task_inputs_dict = {k: v.to(device) for k, v in task_inputs_dict.items()}
    labels_dict = {k: v.to(device) for k, v in labels_dict.items()}

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
    for task in ssl_tasks:
        x_task = task_inputs_dict[task]
        pred = model(x_task, task)
        predictions[task] = pred

    # 損失計算
    total_loss, task_losses = criterion(predictions, labels_dict)

    # バッチサイズ計算用に最初のタスクの入力を返す
    x_base = task_inputs_dict[ssl_tasks[0]]

    return total_loss, task_losses, x_base


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
            # 通常のSSL: (views, _)
            views, _ = batch_data
            view1, view2 = views[0].to(device), views[1].to(device)

            # 順伝播
            z1, z2 = model(view1, view2)
            loss = criterion(z1, z2)
            x = view1  # for batch size

        # 逆伝播
        loss.backward()
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

        # W&Bにログ
        if use_wandb and batch_idx % LOG_INTERVAL == 0:
            log_dict = {
                "train/batch_loss": loss.item(),
                "train/step": epoch * len(dataloader) + batch_idx,
            }
            if multitask:
                for i, task in enumerate(ssl_tasks):
                    log_dict[f"train/batch_{task}_loss"] = task_losses[task].item()
            wandb.log(log_dict)

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

        # バックボーンを作成
        backbone = Resnet(n_channels=in_channels, foundationUK=False)

        # 統合モデルを作成
        model = IntegratedSSLModel(
            backbone=backbone,
            ssl_tasks=ssl_tasks,
            hidden_dim=hidden_dim,
            n_channels=in_channels,
            sequence_length=sequence_length,
        ).to(device)

        logger.info(f"SSL tasks: {ssl_tasks}")
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
        logger.info(f"Epoch {epoch} - Loss: {train_loss:.4f}, {task_losses_str}")
    else:
        logger.info(f"Epoch {epoch} - Average Loss: {train_loss:.4f}")

    # W&Bにログ
    if use_wandb:
        log_dict = {
            "train/epoch_loss": train_loss,
            "train/learning_rate": optimizer.param_groups[0]["lr"],
            "train/epoch": epoch,
        }
        if use_multitask:
            ssl_tasks = get_ssl_tasks_from_config({"multitask": multitask_config})
            for i, task in enumerate(ssl_tasks):
                log_dict[f"train/epoch_{task}_loss"] = train_metrics[f"task{i+1}_loss"]
        wandb.log(log_dict)


def run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
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
        criterion: 損失関数
        optimizer: オプティマイザー
        scheduler: スケジューラー
        config: 設定辞書
        experiment_dirs: 実験ディレクトリ
        device: デバイス
        use_wandb: W&Bを使用するか
        use_multitask: マルチタスク学習を使用するか
        multitask_config: マルチタスク設定
        logger: ロガー
    """
    best_loss = float("inf")
    save_path = str(experiment_dirs.checkpoint)
    num_epochs = config["training"]["epochs"]

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
        )

        # メトリクスをログ
        log_epoch_metrics(
            epoch,
            train_metrics,
            use_multitask,
            multitask_config,
            optimizer,
            use_wandb,
            logger,
        )

        # 学習率を更新
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_metrics["loss"])
            else:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(f"Learning rate: {current_lr:.6f}")

        # チェックポイントを保存
        save_freq = config["checkpoint"].get("save_freq", 10)
        if epoch % save_freq == 0 or epoch == num_epochs:
            is_best = train_metrics["loss"] < best_loss

            if is_best:
                best_loss = train_metrics["loss"]
                logger.info(f"New best loss: {best_loss:.4f}")

            save_file = save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=train_metrics,
                save_path=save_path,
                is_best=is_best,
            )
            logger.info(f"Checkpoint saved to {save_file}")

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

    # 実験ディレクトリを作成
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dirs = ExperimentDirs.create(Path("experiments"), run_id)

    # 設定ファイルを実験ディレクトリにコピー
    shutil.copy(args.config, experiment_dirs.root / "config.yaml")

    # ロガーをセットアップ
    logger = setup_logger("pretrain", str(experiment_dirs.log))
    logger.info(f"Experiment directory: {experiment_dirs.root}")
    logger.info(f"Configuration loaded from {args.config}")
    logger.info(f"Starting pre-training with config: {config['model']['name']}")

    # シード設定
    set_seed(config["seed"])
    logger.info(f"Random seed set to {config['seed']}")

    # デバイス設定
    device = get_device(config["device"])
    logger.info(f"Using device: {device}")

    # マルチタスク学習の設定
    multitask_config = config.get("multitask", {})
    use_multitask = multitask_config.get("enabled", False)
    logger.info(f"Multitask learning: {'Enabled' if use_multitask else 'Disabled'}")

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
