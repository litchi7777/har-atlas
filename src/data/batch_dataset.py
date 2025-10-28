"""
バッチごとにデータを読み込むDataset実装

メモリ効率を考慮して、mmap_mode='r'でデータをロードし、
必要な部分だけをメモリに読み込む。
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Tuple


class SubjectWiseLoader(Dataset):
    """
    被験者ごとにデータを読み込むDataset

    複数データセットのパスを受け取り、各エポックでランダムにサンプリング。
    """

    def __init__(self, paths: List[str], sample_threshold: int):
        """
        Args:
            paths: .npy ファイルのパスリスト（複数データセット可）
            sample_threshold: 最小サンプル数の閾値
        """
        # サンプル数が閾値以上のパスのみをフィルタリング
        self.paths = []
        for path in paths:
            X = np.load(path, mmap_mode="r")
            if X.shape[0] >= sample_threshold:
                self.paths.append(path)

        if len(self.paths) == 0:
            raise ValueError(f"No valid paths found with sample_threshold={sample_threshold}")

        self.sample_threshold = sample_threshold

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        指定されたインデックスのファイルから、sample_threshold分のデータを取得

        Args:
            index: パスリストのインデックス

        Returns:
            (data, data): 入力データ（自己教師ありの場合は同じものを返す）
        """
        path = self.paths[index]
        X = np.load(path, mmap_mode="r")

        # ランダムにsample_threshold分のサンプルを取得
        num_samples = X.shape[0]
        sampled_indices = np.random.choice(num_samples, self.sample_threshold, replace=False)
        data = X[sampled_indices]

        # NaN を 0 に置き換え
        data = np.nan_to_num(data, nan=0.0)

        return (torch.tensor(data, dtype=torch.float32), torch.tensor(data, dtype=torch.float32))

    def __len__(self) -> int:
        return len(self.paths)


class MultiTaskSubjectWiseLoader(Dataset):
    """
    マルチタスク学習用の被験者ごとデータローダー

    データ拡張検出タスクのラベルを生成。
    """

    def __init__(
        self,
        paths: List[str],
        sample_threshold: int,
        specific_transforms: Optional[Dict[str, Any]] = None,
        apply_prob: float = 0.5,
    ):
        """
        Args:
            paths: .npy ファイルのパスリスト（複数データセット可）
            sample_threshold: 最小サンプル数の閾値
            specific_transforms: 拡張辞書 {'permute': Transform, 'reverse': Transform, 'timewarp': Transform}
            apply_prob: 各拡張を適用する確率
        """
        # サンプル数が閾値以上のパスのみをフィルタリング
        self.paths = []
        for path in paths:
            X = np.load(path, mmap_mode="r")
            if X.shape[0] >= sample_threshold:
                self.paths.append(path)

        if len(self.paths) == 0:
            raise ValueError(f"No valid paths found with sample_threshold={sample_threshold}")

        self.sample_threshold = sample_threshold
        self.specific_transforms = specific_transforms or {}
        self.apply_prob = apply_prob
        self.aug_names = ["permute", "reverse", "timewarp"]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        指定されたインデックスのファイルから、sample_threshold分のデータを取得し、
        マルチタスク用のラベルを生成

        Args:
            index: パスリストのインデックス

        Returns:
            (data, labels):
                - data: [sample_threshold, channels, time_steps]
                - labels: [sample_threshold, 3] (3つのバイナリタスクのラベル)
        """
        path = self.paths[index]
        X = np.load(path, mmap_mode="r")

        # ランダムにsample_threshold分のサンプルを取得
        num_samples = X.shape[0]
        sampled_indices = np.random.choice(num_samples, self.sample_threshold, replace=False)
        data = X[sampled_indices]

        # NaN を 0 に置き換え
        data = np.nan_to_num(data, nan=0.0)

        # バッチ全体に対してラベルを生成
        labels = []
        for aug_name in self.aug_names:
            # ランダムに適用するか決定
            apply = np.random.random() < self.apply_prob
            labels.append(1 if apply else 0)

            # 拡張を適用
            if apply and aug_name in self.specific_transforms:
                # バッチ全体に適用
                data = np.array(
                    [
                        self.specific_transforms[aug_name](sample).astype(np.float32)
                        for sample in data
                    ]
                )

        # ラベルをサンプル数分繰り返す
        batch_labels = np.tile(labels, (self.sample_threshold, 1))

        return (
            torch.tensor(data, dtype=torch.float32),
            torch.tensor(batch_labels, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.paths)
