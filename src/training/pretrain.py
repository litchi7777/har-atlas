"""
Self-Supervised Pre-training スクリプト

SSL手法（SimCLR、MoCo等）を用いた事前学習を実行します。
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models.sensor_models import IntegratedSSLModel, Resnet
from src.data.batch_dataset import SubjectWiseLoader, MultiTaskSubjectWiseLoader
from src.data.augmentations import Permutation, Reverse, TimeWarping
from src.losses import IntegratedSSLLoss
from src.utils.config import load_config, validate_config
from src.utils.common import set_seed, get_device, save_checkpoint, count_parameters
from src.utils.logger import setup_logger
from src.utils.training import get_optimizer, get_scheduler, init_wandb, AverageMeter

# Import from har-unified-dataset submodule (after project_root is defined)
# This will be moved to the main function to avoid import-time errors

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_batch_dataloaders(config, logger, use_multitask, multitask_config):
    """
    バッチデータローダーのセットアップ

    Args:
        config: 設定辞書
        logger: ロガー
        use_multitask: マルチタスク学習を使用するか
        multitask_config: マルチタスク設定

    Returns:
        train_loader, val_loader, test_loader, in_channels, sequence_length
    """
    import glob
    from torch.utils.data import RandomSampler

    sensor_config = config["sensor_data"]
    batch_loader_config = sensor_config["batch_loader"]

    logger.info(f"Using batch data loader")

    # 拡張を準備（マルチタスクの場合）
    specific_transforms = {}
    if use_multitask:
        ssl_tasks = multitask_config.get(
            "ssl_tasks", ["binary_permute", "binary_reverse", "binary_timewarp"]
        )
        binary_tasks = [task for task in ssl_tasks if task.startswith("binary_")]
        aug_names = [task.replace("binary_", "") for task in binary_tasks]

        for aug_name in aug_names:
            if aug_name == "permute":
                specific_transforms["permute"] = Permutation(n_segments=4)
            elif aug_name == "reverse":
                specific_transforms["reverse"] = Reverse()
            elif aug_name == "timewarp":
                specific_transforms["timewarp"] = TimeWarping(sigma=0.2, knot=4)

        apply_prob = multitask_config.get("apply_prob", 0.5)
        logger.info(f"Binary SSL tasks: {binary_tasks}")
        logger.info(f"Augmentations: {aug_names}")

    # データセット設定を取得
    data_root = sensor_config.get("data_root", "har-unified-dataset/data/processed")
    datasets = sensor_config.get("datasets", [])
    exclude_patterns = batch_loader_config.get("exclude_patterns", [])

    if not datasets:
        raise ValueError("datasets list is empty in sensor_data config")

    # 各データセットからパスパターンを自動生成して収集
    # データ構造: {dataset}/{USER}/{Device}/{Sensor}/X.npy
    # ACCセンサーのみを使用（チャネル数を統一するため）
    all_paths = []
    for dataset_name in datasets:
        pattern = f"{data_root}/{dataset_name}/*/*/ACC/X.npy"
        paths = sorted(glob.glob(pattern))
        logger.info(f"Dataset '{dataset_name}' (ACC only): {pattern} -> {len(paths)} files")
        all_paths.extend(paths)

    # 除外パターンのフィルタリング
    if exclude_patterns:
        all_paths = [p for p in all_paths if not any(exclude in p for exclude in exclude_patterns)]

    if len(all_paths) == 0:
        raise ValueError(f"No data files found")

    logger.info(f"Total: {len(all_paths)} files")

    sample_threshold = batch_loader_config["sample_threshold"]

    if use_multitask:
        train_set = MultiTaskSubjectWiseLoader(
            all_paths,
            sample_threshold=sample_threshold,
            specific_transforms=specific_transforms,
            apply_prob=apply_prob,
        )
    else:
        train_set = SubjectWiseLoader(all_paths, sample_threshold=sample_threshold)

    # サンプル数設定
    train_num_samples = batch_loader_config.get("train_num_samples", 1000)
    val_num_samples = batch_loader_config.get("val_num_samples", 100)
    test_num_samples = batch_loader_config.get("test_num_samples", 100)
    batch_size = batch_loader_config.get("batch_size", 4)

    # DataLoaderを作成（複数の被験者データを1バッチに含める）
    train_sampler = RandomSampler(train_set, replacement=True, num_samples=train_num_samples)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, drop_last=True, sampler=train_sampler
    )

    val_sampler = RandomSampler(train_set, replacement=True, num_samples=val_num_samples)
    val_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True, sampler=val_sampler)

    test_sampler = RandomSampler(train_set, replacement=True, num_samples=test_num_samples)
    test_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True, sampler=test_sampler)

    # データ形状を取得（最初のバッチから）
    sample_x, _ = train_set[0]
    in_channels = sample_x.shape[1]
    sequence_length = sample_x.shape[2]

    logger.info(f"Input shape: ({in_channels}, {sequence_length})")
    logger.info(f"Training batches per epoch: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, in_channels, sequence_length


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_wandb: bool = False,
    multitask: bool = False,
) -> Dict[str, float]:
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
        multitask: マルチタスク学習モードか

    Returns:
        平均損失の辞書
    """
    model.train()
    loss_meter = AverageMeter("Loss")
    task_loss_meters = [AverageMeter(f"Task{i+1}") for i in range(3)] if multitask else None

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch_data in enumerate(pbar):
        if multitask:
            # マルチタスク学習: (x, labels)
            x, labels_tensor = batch_data

            # バッチローダー使用時は [batch_size, sample_threshold, channels, time]
            # -> [batch_size * sample_threshold, channels, time] にreshape
            if x.dim() == 4:
                batch_size_loader, sample_threshold, channels, time = x.shape
                x = x.reshape(batch_size_loader * sample_threshold, channels, time)
                labels_tensor = labels_tensor.reshape(batch_size_loader * sample_threshold, -1)

            x = x.to(device)
            labels_tensor = labels_tensor.to(device)

            optimizer.zero_grad()

            # 各タスクについて順伝播
            predictions = {}
            labels = {}
            ssl_tasks = model.ssl_tasks

            for i, task in enumerate(ssl_tasks):
                pred = model(x, task)  # (batch_size, num_classes) or (batch_size, 1)
                predictions[task] = pred
                labels[task] = labels_tensor[:, i]  # (batch_size,)

            # 損失計算
            total_loss, task_losses = criterion(predictions, labels)

            # 統計を更新
            for i, task in enumerate(ssl_tasks):
                task_loss_meters[i].update(task_losses[task].item(), x.size(0))

            loss = total_loss
        else:
            # 通常のSSL: (views, _)
            views, _ = batch_data
            view1, view2 = views[0].to(device), views[1].to(device)

            optimizer.zero_grad()

            # 順伝播
            z1, z2 = model(view1, view2)
            loss = criterion(z1, z2)

        # 逆伝播
        loss.backward()
        optimizer.step()

        # 統計を更新
        loss_meter.update(loss.item(), x.size(0) if multitask else view1.size(0))

        if multitask:
            pbar.set_postfix(
                {
                    "loss": f"{loss_meter.avg:.4f}",
                    task_loss_meters[0].name: f"{task_loss_meters[0].avg:.4f}",
                    task_loss_meters[1].name: f"{task_loss_meters[1].avg:.4f}",
                    task_loss_meters[2].name: f"{task_loss_meters[2].avg:.4f}",
                }
            )
        else:
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

        # W&Bにログ
        if use_wandb and batch_idx % 10 == 0:
            log_dict = {
                "train/batch_loss": loss.item(),
                "train/step": epoch * len(dataloader) + batch_idx,
            }
            if multitask:
                for i, task in enumerate(ssl_tasks):
                    log_dict[f"train/batch_{task}_loss"] = task_losses[task].item()
            wandb.log(log_dict)

    if multitask:
        result = {"loss": loss_meter.avg}
        for i, meter in enumerate(task_loss_meters):
            result[f"task{i+1}_loss"] = meter.avg
        return result
    else:
        return {"loss": loss_meter.avg}


def main(args: argparse.Namespace) -> None:
    """
    メイン関数

    Args:
        args: コマンドライン引数
    """
    # Import from har-unified-dataset submodule
    import sys

    har_dataset_path = project_root / "har-unified-dataset" / "src"
    sys.path.insert(0, str(har_dataset_path))
    from dataset_info import get_dataset_info, select_sensors

    # 設定をロード
    config = load_config(args.config)
    validate_config(config, mode="pretrain")

    # 実験ディレクトリを作成
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path("experiments") / "pretrain" / f"run_{run_id}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # checkpoint保存用ディレクトリ
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # ログディレクトリ
    log_dir = experiment_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # 設定ファイルを実験ディレクトリにコピー
    import shutil

    shutil.copy(args.config, experiment_dir / "config.yaml")

    # ロガーをセットアップ（実験ディレクトリ内に）
    logger = setup_logger("pretrain", str(log_dir))
    logger.info(f"Experiment directory: {experiment_dir}")
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
    dataset_type = config.get("dataset_type", "sensor")

    try:
        if dataset_type == "sensor":
            # センサーデータの場合（常にバッチローダーを使用）
            sensor_config = config["sensor_data"]

            # バッチローダーを使用（メモリ効率化）
            train_loader, val_loader, test_loader, in_channels, sequence_length = (
                setup_batch_dataloaders(config, logger, use_multitask, multitask_config)
            )

        logger.info(f"Number of batches: {len(train_loader)}")

    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        raise

    # モデルを作成
    if dataset_type == "sensor":
        if use_multitask:
            # 統合型SSLモデル
            ssl_tasks = multitask_config.get("ssl_tasks", ["permute", "reverse", "timewarp"])
            hidden_dim = config["model"].get("feature_dim", 256)

            # バックボーンを作成
            backbone = Resnet(n_channels=in_channels, foundationUK=False)

            # 統合モデルを作成
            model = IntegratedSSLModel(
                backbone=backbone, ssl_tasks=ssl_tasks, hidden_dim=hidden_dim
            ).to(device)

            logger.info(f"SSL tasks: {ssl_tasks}")
        else:
            # 通常のセンサーSSLモデル
            model = SensorSSLModel(
                in_channels=in_channels,
                backbone=config["model"]["backbone"],
                projection_dim=config["model"].get("projection_dim", 128),
                hidden_dim=config["model"].get("feature_dim", 256),
            ).to(device)
    else:
        # 画像モデル（現在は未サポート）
        model = SSLModel(config["model"]).to(device)

    param_info = count_parameters(model)
    logger.info(
        f"Model created: {config['model']['backbone']}, "
        f"Total params: {param_info['total']:,}, "
        f"Trainable: {param_info['trainable']:,}"
    )

    # 損失関数を設定
    if use_multitask:
        # 統合型SSL損失関数
        ssl_tasks = multitask_config.get("ssl_tasks", ["permute", "reverse", "timewarp"])
        task_weights_list = multitask_config.get("task_weights", [1.0, 1.0, 1.0])
        task_weights = {task: weight for task, weight in zip(ssl_tasks, task_weights_list)}

        criterion = IntegratedSSLLoss(ssl_tasks=ssl_tasks, task_weights=task_weights)
        logger.info(f"Task weights: {task_weights}")
    else:
        # 通常のSSL損失
        ssl_method = config.get("ssl", {}).get("method", "simclr")
        ssl_config = config.get("ssl", {}).copy()
        # methodキーを削除（get_ssl_lossの第一引数として渡すため）
        ssl_config.pop("method", None)
        criterion = get_ssl_loss(ssl_method, **ssl_config)
        logger.info(f"Using SSL method: {ssl_method}")

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
        T_max=config["training"]["epochs"],
    )

    # W&Bを初期化
    use_wandb = init_wandb(config, model)

    # トレーニングループ
    best_loss = float("inf")
    save_path = str(checkpoint_dir)

    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)

    for epoch in range(1, config["training"]["epochs"] + 1):
        logger.info(f"\nEpoch {epoch}/{config['training']['epochs']}")

        # 学習
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, use_wandb, use_multitask
        )

        train_loss = train_metrics["loss"]
        if use_multitask:
            ssl_tasks = multitask_config.get("ssl_tasks", ["permute", "reverse", "timewarp"])
            task_losses_str = ", ".join(
                [
                    f"{task}: {train_metrics[f'task{i+1}_loss']:.4f}"
                    for i, task in enumerate(ssl_tasks)
                ]
            )
            logger.info(f"Epoch {epoch} - Loss: {train_loss:.4f}, {task_losses_str}")
        else:
            logger.info(f"Epoch {epoch} - Average Loss: {train_loss:.4f}")

        # 学習率を更新
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_loss)
            else:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(f"Learning rate: {current_lr:.6f}")

        # W&Bにログ
        if use_wandb:
            log_dict = {
                "train/epoch_loss": train_loss,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/epoch": epoch,
            }
            if use_multitask:
                ssl_tasks = multitask_config.get("ssl_tasks", ["permute", "reverse", "timewarp"])
                for i, task in enumerate(ssl_tasks):
                    log_dict[f"train/epoch_{task}_loss"] = train_metrics[f"task{i+1}_loss"]
            wandb.log(log_dict)

        # チェックポイントを保存
        save_freq = config["checkpoint"].get("save_freq", 10)
        if epoch % save_freq == 0 or epoch == config["training"]["epochs"]:
            is_best = train_loss < best_loss

            if is_best:
                best_loss = train_loss
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

    # クリーンアップ
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Self-Supervised Pre-training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="configs/pretrain.yaml", help="Path to configuration file"
    )

    args = parser.parse_args()
    main(args)
