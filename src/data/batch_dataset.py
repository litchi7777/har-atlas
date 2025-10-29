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


class LabeledSubjectWiseLoader(Dataset):
    """
    ファインチューニング用の被験者ごとデータローダー（ラベル付き）

    データとラベルのペアを読み込む。
    X.npyとy.npyのパスをペアで管理。
    """

    def __init__(self, paths: List[Tuple[str, str]], sample_threshold: int):
        """
        Args:
            paths: (X.npy, y.npy) パスのタプルリスト
            sample_threshold: 最小サンプル数の閾値
        """
        # サンプル数が閾値以上のパスのみをフィルタリング
        self.paths = []
        for x_path, y_path in paths:
            X = np.load(x_path, mmap_mode="r")
            if X.shape[0] >= sample_threshold:
                self.paths.append((x_path, y_path))

        if len(self.paths) == 0:
            raise ValueError(
                f"No valid paths found with sample_threshold={sample_threshold}"
            )

        self.sample_threshold = sample_threshold

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        指定されたインデックスのファイルから、sample_threshold分のデータとラベルを取得

        Args:
            index: パスリストのインデックス

        Returns:
            (data, labels):
                - data: [sample_threshold, channels, time_steps]
                - labels: [sample_threshold] (クラスラベル)
        """
        x_path, y_path = self.paths[index]
        X = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")

        # 負のラベル（Undefinedなど）を除外
        valid_mask = y >= 0
        valid_indices = np.where(valid_mask)[0]

        # 有効なサンプルが十分にない場合は、少ない数でサンプリング
        num_valid = len(valid_indices)
        if num_valid < self.sample_threshold:
            sampled_indices = valid_indices
        else:
            sampled_indices = np.random.choice(
                valid_indices, self.sample_threshold, replace=False
            )

        data = X[sampled_indices]
        labels = y[sampled_indices]

        # NaN を 0 に置き換え
        data = np.nan_to_num(data, nan=0.0)

        return (
            torch.tensor(data, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        )

    def __len__(self) -> int:
        return len(self.paths)


class InMemoryDataset(Dataset):
    """
    全データをメモリに読み込むデータセット（Finetune用）

    SubjectWiseではなく、全サンプルをシャッフルして使用。
    小規模データセットに適している。
    """

    def __init__(self, paths: List[Tuple[str, str]], filter_negative_labels: bool = True):
        """
        Args:
            paths: (X.npy, y.npy) パスのタプルリスト
            filter_negative_labels: 負のラベルを除外するか
        """
        self.data = []
        self.labels = []

        for x_path, y_path in paths:
            X = np.load(x_path, mmap_mode="r")
            y = np.load(y_path, mmap_mode="r")

            # 負のラベルをフィルタリング
            if filter_negative_labels:
                valid_mask = y >= 0
                X = X[valid_mask]
                y = y[valid_mask]

            # NaN を 0 に置き換え
            X = np.nan_to_num(X, nan=0.0)

            self.data.append(X)
            self.labels.append(y)

        # 全データを結合
        self.data = np.concatenate(self.data, axis=0).astype(np.float32)
        self.labels = np.concatenate(self.labels, axis=0).astype(np.int64)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        指定されたインデックスのサンプルを取得

        Args:
            index: サンプルインデックス

        Returns:
            (data, label):
                - data: [channels, time_steps]
                - label: scalar (クラスラベル)
        """
        return (
            torch.tensor(self.data[index], dtype=torch.float32),
            torch.tensor(self.labels[index], dtype=torch.long),
        )

    def __len__(self) -> int:
        return len(self.data)
