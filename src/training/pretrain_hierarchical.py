"""
Hierarchical SSL Pre-training スクリプト

PiCO inspired の階層的対照学習を用いた事前学習を実行します。
- Activity-level: 同じActivity同士をpositive
- Prototype-level: Body Part別のAtomic Motion Prototype学習

Usage:
    python src/training/pretrain_hierarchical.py --config configs/pretrain_hierarchical.yaml
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.losses.hierarchical_loss import HierarchicalSSLLoss
from src.models.backbones import Resnet
from src.utils.atlas_loader import AtlasLoader
from src.utils.common import count_parameters, get_device, save_checkpoint, set_seed
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.training import AverageMeter, get_optimizer, get_scheduler

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# Dataset
# =============================================================================

class HierarchicalSSLDataset(Dataset):
    """
    階層的SSL学習用データセット

    各サンプルに対して (data, label, dataset_name, body_part) を返す
    """

    def __init__(
        self,
        data_root: str,
        dataset_location_pairs: List[List[str]],
        atlas: AtlasLoader,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        """
        Args:
            data_root: データルートディレクトリ
            dataset_location_pairs: [[dataset, location], ...] のリスト
            atlas: AtlasLoaderインスタンス
            split: "train", "val", "test"
            train_ratio: 訓練データの割合
            val_ratio: 検証データの割合
            seed: ランダムシード
        """
        self.data_root = Path(data_root)
        self.atlas = atlas
        self.split = split

        # データを収集
        self.samples = []  # [(X_path, Y_path, dataset, location), ...]
        self._collect_samples(dataset_location_pairs)

        # 分割
        self._split_data(train_ratio, val_ratio, seed)

        # ラベル→Activity名のマッピングを構築
        self._build_label_mapping()

    def _collect_samples(self, dataset_location_pairs: List[List[str]]):
        """データパスを収集"""
        for pair in dataset_location_pairs:
            dataset, location = pair[0], pair[1]
            dataset_path = self.data_root / dataset.lower()

            if not dataset_path.exists():
                continue

            # ユーザーごとにデータを収集
            for user_dir in sorted(dataset_path.iterdir()):
                if not user_dir.is_dir() or user_dir.name.startswith('.'):
                    continue
                if user_dir.name == 'metadata.json':
                    continue

                # センサー位置
                location_dir = user_dir / location
                if not location_dir.exists():
                    continue

                # ACCデータを探す
                acc_dir = location_dir / "ACC"
                if not acc_dir.exists():
                    continue

                x_path = acc_dir / "X.npy"
                y_path = acc_dir / "Y.npy"

                if x_path.exists() and y_path.exists():
                    self.samples.append((
                        str(x_path),
                        str(y_path),
                        dataset.lower(),
                        location,
                    ))

    def _split_data(self, train_ratio: float, val_ratio: float, seed: int):
        """データを分割"""
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(self.samples))

        n_train = int(len(indices) * train_ratio)
        n_val = int(len(indices) * val_ratio)

        if self.split == "train":
            selected = indices[:n_train]
        elif self.split == "val":
            selected = indices[n_train:n_train + n_val]
        else:  # test
            selected = indices[n_train + n_val:]

        self.samples = [self.samples[i] for i in selected]

    def _build_label_mapping(self):
        """ラベル→Activity名のマッピングを構築（dataset_info.pyから）"""
        # dataset_info.pyからラベルマッピングを読み込む
        try:
            from har_unified_dataset.src.dataset_info import DATASETS
            self.label_to_activity = {}
            for dataset, info in DATASETS.items():
                self.label_to_activity[dataset.lower()] = {
                    v: k for k, v in info.get("labels", {}).items()
                    if isinstance(k, int)  # -1などの特殊ラベルも含む
                }
        except ImportError:
            # フォールバック: 直接読み込み
            self.label_to_activity = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Returns:
            {
                "data": (batch_samples, channels, time_steps),
                "labels": (batch_samples,),
                "dataset": str,
                "body_part": str,
            }
        """
        x_path, y_path, dataset, location = self.samples[index]

        # データ読み込み
        X = np.load(x_path)  # (N, channels, time_steps)
        Y = np.load(y_path)  # (N,)

        # 有効なラベルのみ（-1を除外）
        valid_mask = Y >= 0
        X = X[valid_mask]
        Y = Y[valid_mask]

        if len(X) == 0:
            # フォールバック: 全データを使用
            X = np.load(x_path)
            Y = np.load(y_path)

        return {
            "data": torch.tensor(X, dtype=torch.float32),
            "labels": torch.tensor(Y, dtype=torch.long),
            "dataset": dataset,
            "body_part": location,
        }


def collate_hierarchical(batch: List[Dict]) -> Dict[str, Any]:
    """
    カスタムcollate関数

    各サンプルからランダムにウィンドウを抽出してバッチを構成
    """
    samples_per_source = 4  # 各ソースから抽出するサンプル数

    all_data = []
    all_labels = []
    all_datasets = []
    all_body_parts = []

    for item in batch:
        data = item["data"]  # (N, C, T)
        labels = item["labels"]  # (N,)
        dataset = item["dataset"]
        body_part = item["body_part"]

        n_samples = len(data)
        if n_samples == 0:
            continue

        # ランダムにサンプリング
        n_select = min(samples_per_source, n_samples)
        indices = np.random.choice(n_samples, n_select, replace=False)

        all_data.append(data[indices])
        all_labels.append(labels[indices])
        all_datasets.extend([dataset] * n_select)
        all_body_parts.extend([body_part] * n_select)

    if not all_data:
        return None

    return {
        "data": torch.cat(all_data, dim=0),  # (total_samples, C, T)
        "labels": torch.cat(all_labels, dim=0),  # (total_samples,)
        "datasets": all_datasets,
        "body_parts": all_body_parts,
    }


# =============================================================================
# Model
# =============================================================================

class HierarchicalSSLModel(nn.Module):
    """
    階層的SSL用モデル

    エンコーダー + プロジェクションヘッド
    """

    def __init__(
        self,
        in_channels: int = 3,
        sequence_length: int = 60,
        embed_dim: int = 512,
        projection_dim: int = 256,
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


# =============================================================================
# Training
# =============================================================================

def train_epoch(
    model: nn.Module,
    loss_fn: HierarchicalSSLLoss,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    atlas: AtlasLoader,
    epoch: int,
    logger,
) -> Dict[str, float]:
    """1エポックの訓練"""
    model.train()
    loss_fn.train()

    loss_meter = AverageMeter()
    activity_loss_meter = AverageMeter()
    prototype_loss_meter = AverageMeter()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch in pbar:
        if batch is None:
            continue

        data = batch["data"].to(device)  # (B, C, T)
        labels = batch["labels"]  # (B,)
        datasets = batch["datasets"]  # List[str]
        body_parts = batch["body_parts"]  # List[str]

        # ラベルからActivity名を取得
        activity_ids = []
        for i, (ds, label) in enumerate(zip(datasets, labels.tolist())):
            # dataset_info.pyのラベルマッピングを使用
            activity_name = get_activity_name(ds, label, atlas)
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


def validate(
    model: nn.Module,
    loss_fn: HierarchicalSSLLoss,
    val_loader: DataLoader,
    device: torch.device,
    atlas: AtlasLoader,
) -> Dict[str, float]:
    """検証"""
    model.eval()
    loss_fn.eval()

    loss_meter = AverageMeter()

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue

            data = batch["data"].to(device)
            labels = batch["labels"]
            datasets = batch["datasets"]
            body_parts = batch["body_parts"]

            activity_ids = []
            for ds, label in zip(datasets, labels.tolist()):
                activity_name = get_activity_name(ds, label, atlas)
                activity_ids.append(activity_name)

            embeddings = model(data)

            total_loss, _ = loss_fn(
                embeddings=embeddings,
                dataset_ids=datasets,
                activity_ids=activity_ids,
                body_parts=body_parts,
            )

            loss_meter.update(total_loss.item(), data.size(0))

    return {"val_loss": loss_meter.avg}


def camel_to_snake(name: str) -> str:
    """CamelCaseをsnake_caseに変換"""
    import re
    # CamelCase -> snake_case
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


# Dataset-specific activity name mappings to Atlas names
ACTIVITY_NAME_MAPPING = {
    "dsads": {
        "lying_back": "lying_on_back",
        "lying_right": "lying_on_right",
        "stairs_up": "ascending_stairs",
        "stairs_down": "descending_stairs",
        "standing_elevator_still": "standing_in_elevator",
        "moving_elevator": "moving_in_elevator",
        "walking_parking": "walking_in_parking_lot",
        "walking_treadmill_flat": "walking_on_treadmill_flat",
        "walking_treadmill_slope": "walking_on_treadmill_inclined",
        "running_treadmill": "running_on_treadmill",
        "exercising_stepper": "exercising_on_stepper",
        "exercising_cross_trainer": "exercising_on_cross_trainer",
        "cycling_exercise_bike_vertical": "cycling_vertical",
        "cycling_exercise_bike_horizontal": "cycling_horizontal",
    },
    "mhealth": {
        "standing": "standing_still",
        "sitting": "sitting_relaxing",
        "stairs_up": "climbing_stairs",
    },
    "pamap2": {
        "lying": "lying_down",
        "stairs_up": "ascending_stairs",
        "stairs_down": "descending_stairs",
        "vacuum_cleaning": "vacuuming",
        "ironing": "ironing_clothes",
        "rope_jumping": "rope_skipping",
    },
}


def get_activity_name(dataset: str, label: int, atlas: AtlasLoader) -> str:
    """ラベルIDからActivity名を取得（Atlas互換形式）"""
    # dataset_info.pyからラベルマッピングを取得
    try:
        sys.path.insert(0, str(project_root / "har-unified-dataset" / "src"))
        from dataset_info import DATASETS

        ds_upper = dataset.upper()
        ds_lower = dataset.lower()
        if ds_upper in DATASETS:
            labels_dict = DATASETS[ds_upper].get("labels", {})
            if label in labels_dict:
                raw_name = labels_dict[label]
                # ラベル名をAtlas形式に変換
                # 1. CamelCase -> snake_case
                activity = camel_to_snake(raw_name)
                # 2. 括弧やカンマを除去/置換
                activity = activity.replace("(", "_").replace(")", "").replace(",", "")
                # 3. スペースをアンダースコアに
                activity = activity.replace(" ", "_")
                # 4. 連続アンダースコアを単一に
                while "__" in activity:
                    activity = activity.replace("__", "_")
                # 5. 末尾のアンダースコアを除去
                activity = activity.strip("_")

                # 6. Dataset固有のマッピングを適用
                if ds_lower in ACTIVITY_NAME_MAPPING:
                    activity = ACTIVITY_NAME_MAPPING[ds_lower].get(activity, activity)

                return activity
    except Exception:
        pass

    # フォールバック: ラベルIDをそのまま使用
    return f"activity_{label}"


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hierarchical SSL Pre-training")
    parser.add_argument("--config", type=str, required=True, help="設定ファイルパス")
    parser.add_argument("--seed", type=int, default=42, help="ランダムシード")
    args = parser.parse_args()

    # 設定読み込み
    config = load_config(args.config)

    # シード設定
    set_seed(args.seed)

    # デバイス
    device = get_device()

    # 実験ディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(config.get("experiment_dir", "experiments/hierarchical")) / f"run_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ロガー
    logger = setup_logger(exp_dir / "experiment.log")
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Device: {device}")

    # Atlas読み込み
    atlas_path = config.get("atlas_path", "docs/atlas/activity_mapping.json")
    atlas = AtlasLoader(atlas_path)
    logger.info(f"Atlas loaded: {len(atlas.get_datasets())} datasets")

    # データセット
    data_config = config.get("data", {})
    dataset_location_pairs = data_config.get("dataset_location_pairs", [])

    train_dataset = HierarchicalSSLDataset(
        data_root=data_config.get("data_root", "har-unified-dataset/data/processed"),
        dataset_location_pairs=dataset_location_pairs,
        atlas=atlas,
        split="train",
    )
    val_dataset = HierarchicalSSLDataset(
        data_root=data_config.get("data_root", "har-unified-dataset/data/processed"),
        dataset_location_pairs=dataset_location_pairs,
        atlas=atlas,
        split="val",
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # データローダー
    batch_size = config.get("training", {}).get("batch_size", 8)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_hierarchical,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_hierarchical,
        num_workers=0,
    )

    # モデル
    model_config = config.get("model", {})
    model = HierarchicalSSLModel(
        in_channels=model_config.get("in_channels", 3),
        sequence_length=model_config.get("sequence_length", 60),
        embed_dim=model_config.get("embed_dim", 512),
    ).to(device)

    logger.info(f"Model parameters: {count_parameters(model):,}")

    # Loss関数
    loss_config = config.get("loss", {})
    loss_fn = HierarchicalSSLLoss(
        atlas_path=atlas_path,
        embed_dim=model_config.get("embed_dim", 512),
        prototype_dim=loss_config.get("prototype_dim", 128),
        temperature=loss_config.get("temperature", 0.1),
        lambda_activity=loss_config.get("lambda_activity", 0.5),
        lambda_prototype=loss_config.get("lambda_prototype", 0.5),
    ).to(device)

    # Optimizer
    training_config = config.get("training", {})
    optimizer = get_optimizer(
        list(model.parameters()) + list(loss_fn.parameters()),
        training_config.get("optimizer", "adam"),
        training_config.get("learning_rate", 0.001),
        training_config.get("weight_decay", 0.0001),
    )

    # Scheduler
    num_epochs = training_config.get("num_epochs", 100)
    scheduler = get_scheduler(
        optimizer,
        training_config.get("scheduler", "cosine"),
        num_epochs,
    )

    # W&B
    if WANDB_AVAILABLE and config.get("wandb", {}).get("enabled", False):
        wandb.init(
            project=config["wandb"].get("project", "har-hierarchical"),
            name=f"run_{timestamp}",
            config=config,
        )

    # 訓練ループ
    best_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, loss_fn, train_loader, optimizer, device, atlas, epoch, logger
        )

        # Validate
        val_metrics = validate(model, loss_fn, val_loader, device, atlas)

        # Scheduler step
        if scheduler:
            scheduler.step()

        # ログ
        logger.info(
            f"Epoch {epoch}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"val_loss={val_metrics['val_loss']:.4f}"
        )

        # W&B
        if WANDB_AVAILABLE and wandb.run:
            wandb.log({
                "epoch": epoch,
                **train_metrics,
                **val_metrics,
            })

        # チェックポイント保存
        if val_metrics["val_loss"] < best_loss:
            best_loss = val_metrics["val_loss"]
            save_checkpoint(
                model,
                optimizer,
                epoch,
                exp_dir / "checkpoints" / "best_model.pth",
            )
            logger.info(f"New best model saved (loss={best_loss:.4f})")

        # 定期保存
        if epoch % training_config.get("save_every", 10) == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                exp_dir / "checkpoints" / f"epoch_{epoch}.pth",
            )

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
