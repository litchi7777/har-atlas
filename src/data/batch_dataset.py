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

    データは元のまま返し、拡張はトレーニングループ側で行う（高速化のため）
    """

    def __init__(
        self,
        paths: List[str],
        sample_threshold: int,
        ssl_tasks: List[str],
        specific_transforms: Optional[Dict[str, Any]] = None,
        apply_prob: float = 0.5,
    ):
        """
        Args:
            paths: .npy ファイルのパスリスト（複数データセット可）
            sample_threshold: 最小サンプル数の閾値
            ssl_tasks: SSLタスクのリスト（例: ["binary_permute", "masking_channel"]）
            specific_transforms: 拡張辞書（使用しない、互換性のため残す）
            apply_prob: 各拡張を適用する確率（使用しない、互換性のため残す）
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
        self.ssl_tasks = ssl_tasks

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        指定されたインデックスのファイルから、sample_threshold分のデータを取得

        Args:
            index: パスリストのインデックス

        Returns:
            data: [sample_threshold, channels, time_steps]
        """
        path = self.paths[index]
        X = np.load(path, mmap_mode="r")

        # ランダムにsample_threshold分のサンプルを取得
        num_samples = X.shape[0]
        sampled_indices = np.random.choice(num_samples, self.sample_threshold, replace=False)
        data = X[sampled_indices]

        # NaN を 0 に置き換え
        data = np.nan_to_num(data, nan=0.0)

        return torch.tensor(data, dtype=torch.float32)

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


class MultiDeviceInMemoryDataset(Dataset):
    """
    マルチデバイス対応の全データメモリ読み込みデータセット

    各デバイス（装着部位）のデータを個別に保持し、リストとして返す。
    任意の数のデバイスに対応（1つ、3つ、5つなど）。

    Example:
        # 5デバイスの場合
        devices = ['Torso', 'RightArm', 'LeftArm', 'RightLeg', 'LeftLeg']

        # 3デバイスの場合
        devices = ['Torso', 'RightArm', 'LeftArm']

        # 1デバイスの場合
        devices = ['Torso']
    """

    def __init__(
        self,
        paths_per_device: Dict[str, List[Tuple[str, str]]],
        filter_negative_labels: bool = True
    ):
        """
        Args:
            paths_per_device: {device_name: [(X.npy, y.npy), ...]} の辞書
            filter_negative_labels: 負のラベルを除外するか
        """
        self.device_names = sorted(paths_per_device.keys())  # 順序を固定
        self.num_devices = len(self.device_names)

        if self.num_devices == 0:
            raise ValueError("No devices specified")

        # 各デバイスのデータを格納
        self.device_data = {device: [] for device in self.device_names}
        self.labels = []  # ラベルは全デバイスで共通

        # 最初のデバイスのパスリストを基準にする
        reference_device = self.device_names[0]
        reference_paths = paths_per_device[reference_device]

        # 各パス（ファイル）について、全デバイスのデータを読み込む
        for path_idx, (ref_x_path, ref_y_path) in enumerate(reference_paths):
            # ラベルを読み込み（全デバイス共通）
            y = np.load(ref_y_path, mmap_mode="r")

            # 負のラベルをフィルタリング
            if filter_negative_labels:
                valid_mask = y >= 0
            else:
                valid_mask = np.ones(len(y), dtype=bool)

            # 各デバイスのデータを読み込む
            device_samples = {}
            valid_in_all = valid_mask.copy()

            for device in self.device_names:
                device_paths = paths_per_device[device]
                if path_idx >= len(device_paths):
                    # このデバイスには対応するパスがない場合はスキップ
                    continue

                x_path, _ = device_paths[path_idx]
                X = np.load(x_path, mmap_mode="r")

                # NaN を 0 に置き換え
                X = np.nan_to_num(X, nan=0.0)

                device_samples[device] = X

            # 全デバイスで有効なサンプルのみを保持
            if len(device_samples) == self.num_devices:
                for device in self.device_names:
                    self.device_data[device].append(device_samples[device][valid_in_all])

                self.labels.append(y[valid_in_all])

        # 各デバイスのデータを結合
        for device in self.device_names:
            if len(self.device_data[device]) > 0:
                self.device_data[device] = np.concatenate(
                    self.device_data[device], axis=0
                ).astype(np.float32)
            else:
                raise ValueError(f"No data found for device: {device}")

        # ラベルを結合
        if len(self.labels) > 0:
            self.labels = np.concatenate(self.labels, axis=0).astype(np.int64)
        else:
            raise ValueError("No labels found")

        # データ長の一貫性チェック
        data_length = len(self.labels)
        for device in self.device_names:
            if len(self.device_data[device]) != data_length:
                raise ValueError(
                    f"Data length mismatch for device {device}: "
                    f"{len(self.device_data[device])} vs {data_length}"
                )

    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        指定されたインデックスのサンプルを取得

        Args:
            index: サンプルインデックス

        Returns:
            (device_data_list, label):
                - device_data_list: [torch.Tensor, ...] 各デバイスのデータ
                  各Tensor形状: [channels, time_steps]
                - label: scalar (クラスラベル)
        """
        device_data_list = [
            torch.tensor(self.device_data[device][index], dtype=torch.float32)
            for device in self.device_names
        ]

        label = torch.tensor(self.labels[index], dtype=torch.long)

        return device_data_list, label

    def __len__(self) -> int:
        return len(self.labels)

    def get_device_names(self) -> List[str]:
        """使用しているデバイス名のリストを返す"""
        return self.device_names.copy()

    def get_num_devices(self) -> int:
        """デバイス数を返す"""
        return self.num_devices
