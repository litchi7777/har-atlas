"""
Hierarchical SSL用データセット

Activity + Body Part情報を含むデータセットとcollate関数を提供
AtlasLoaderを使用してActivity名とBody Partを正規化
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.utils.atlas_loader import AtlasLoader

# グローバルAtlasLoaderインスタンス（遅延初期化）
_atlas: Optional[AtlasLoader] = None


def get_atlas() -> AtlasLoader:
    """AtlasLoaderのシングルトンインスタンスを取得"""
    global _atlas
    if _atlas is None:
        atlas_path = project_root / "docs" / "atlas" / "activity_mapping.json"
        _atlas = AtlasLoader(str(atlas_path))
    return _atlas


def get_activity_name(dataset: str, label: int) -> str:
    """
    ラベルIDからActivity名を取得（Atlas互換形式）

    AtlasLoaderを使用してデータセット・ラベルIDから正規化されたActivity名を取得

    Args:
        dataset: データセット名
        label: ラベルID

    Returns:
        正規化されたActivity名
    """
    atlas = get_atlas()
    return atlas.get_activity_name_by_label(dataset, label)


def normalize_body_part(dataset: str, sensor_location: str) -> str:
    """
    センサー位置を5つの学習用カテゴリに正規化

    Args:
        dataset: データセット名
        sensor_location: センサー位置名

    Returns:
        正規化されたBody Part (wrist, hip, chest, leg, head)
    """
    atlas = get_atlas()
    return atlas.normalize_body_part(dataset, sensor_location)


class HierarchicalSSLDataset(Dataset):
    """
    階層的SSL学習用データセット

    各サンプルに対して (data, label, dataset_name, body_part) を返す
    """

    def __init__(
        self,
        data_root: str,
        dataset_location_pairs: List[List[str]],
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        """
        Args:
            data_root: データルートディレクトリ
            dataset_location_pairs: [[dataset, location], ...] のリスト
            split: "train", "val", "test"
            train_ratio: 訓練データの割合
            val_ratio: 検証データの割合
            seed: ランダムシード
        """
        self.data_root = Path(data_root)
        self.split = split

        # データを収集
        self.samples = []  # [(X_path, Y_path, dataset, location), ...]
        self._collect_samples(dataset_location_pairs)

        # 分割
        self._split_data(train_ratio, val_ratio, seed)

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

        # Body Partを5カテゴリに正規化
        normalized_body_part = normalize_body_part(dataset, location)

        return {
            "data": torch.tensor(X, dtype=torch.float32),
            "labels": torch.tensor(Y, dtype=torch.long),
            "dataset": dataset,
            "body_part": normalized_body_part,
        }


def collate_hierarchical(batch: List[Dict], samples_per_source: int = 4) -> Optional[Dict[str, Any]]:
    """
    カスタムcollate関数

    各サンプルからランダムにウィンドウを抽出してバッチを構成
    """
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
