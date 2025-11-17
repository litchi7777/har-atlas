"""
バッチごとにデータを読み込むDataset実装

メモリ効率を考慮して、mmap_mode='r'でデータをロードし、
必要な部分だけをメモリに読み込む。
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Tuple, Union


class SubjectWiseLoader(Dataset):
    """
    被験者ごとにデータを読み込む汎用Dataset

    Pre-training用（ラベルなし）とFine-tuning用（ラベル付き）の両方に対応。
    ssl_tasksパラメータでマルチタスクSSLにも対応。

    マルチデバイス対応:
    - シングルデバイス: paths = ["/path/to/X.npy", ...]
    - マルチデバイス: paths = [("/path/to/device1/X.npy", "/path/to/device2/X.npy"), ...]
    """

    def __init__(
        self,
        paths: List[Union[str, Tuple[str, ...]]],
        sample_threshold: int,
        ssl_tasks: Optional[List[str]] = None,
        window_size: Optional[int] = None,
        original_window_size: Optional[int] = None,
        window_clip_strategy: str = "random",
    ):
        """
        Args:
            paths: .npy ファイルのパスリスト（複数データセット可）
                   - シングルデバイス: List[str]
                   - マルチデバイス: List[Tuple[str, ...]]
            sample_threshold: 最小サンプル数の閾値
            ssl_tasks: SSLタスクのリスト（互換性のため残すが未使用）
            window_size: クリップ後のウィンドウサイズ（時間ステップ数）
            original_window_size: 元のウィンドウサイズ（時間ステップ数）
            window_clip_strategy: クリップ戦略（"random", "center", "start", "end"）
        """
        # 全てのパスを追加（サンプル数に関わらず）
        self.paths = paths

        if len(self.paths) == 0:
            raise ValueError("No paths provided")

        self.sample_threshold = sample_threshold
        self.ssl_tasks = ssl_tasks

        # ウィンドウクリップ設定
        self.window_size = window_size
        self.original_window_size = original_window_size
        self.window_clip_strategy = window_clip_strategy

        # クリップが有効かどうか
        self.enable_clip = (
            window_size is not None
            and original_window_size is not None
            and window_size < original_window_size
        )

        # マルチデバイスかどうかを判定
        self.is_multi_device = isinstance(self.paths[0], tuple)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        指定されたインデックスのファイルから、sample_threshold分のデータを取得

        Args:
            index: パスリストのインデックス

        Returns:
            data: [sample_threshold, channels, time_steps]
                  - シングルデバイス: channels = 3
                  - マルチデバイス: channels = 3 * デバイス数
        """
        path = self.paths[index]

        if self.is_multi_device:
            # マルチデバイス: 複数のファイルを読み込んでチャネル方向に結合
            device_data_list = []
            num_samples = None

            for device_path in path:
                X = np.load(device_path, mmap_mode="r")
                device_data_list.append(X)

                # 最初のデバイスのサンプル数を基準にする
                if num_samples is None:
                    num_samples = X.shape[0]

            # ランダムにsample_threshold分のサンプルを取得
            replace = num_samples < self.sample_threshold
            sampled_indices = np.random.choice(num_samples, self.sample_threshold, replace=replace)

            # 各デバイスのデータをサンプリングして結合
            sampled_device_data = []
            for X in device_data_list:
                # サンプルインデックスを使用（全デバイスで同じサンプル）
                sampled_data = X[sampled_indices]
                sampled_device_data.append(sampled_data)

            # チャネル方向に結合 (sample_threshold, 3*num_devices, time_steps)
            data = np.concatenate(sampled_device_data, axis=1)

            # NaN を 0 に置き換え
            data = np.nan_to_num(data, nan=0.0)

            # ウィンドウクリッピング（有効な場合）
            if self.enable_clip:
                data = self._clip_windows(data)

            return torch.tensor(data, dtype=torch.float32)

        else:
            # シングルデバイス: 従来通りの処理
            X = np.load(path, mmap_mode="r")

            # ランダムにsample_threshold分のサンプルを取得
            num_samples = X.shape[0]
            replace = num_samples < self.sample_threshold
            sampled_indices = np.random.choice(num_samples, self.sample_threshold, replace=replace)
            data = X[sampled_indices]

            # NaN を 0 に置き換え
            data = np.nan_to_num(data, nan=0.0)

            # ウィンドウクリッピング（有効な場合）
            if self.enable_clip:
                data = self._clip_windows(data)

            return torch.tensor(data, dtype=torch.float32)

    def _clip_windows(self, data: np.ndarray) -> np.ndarray:
        """
        ウィンドウをクリップする

        Args:
            data: [sample_threshold, channels, original_window_size]

        Returns:
            clipped_data: [sample_threshold, channels, window_size]
        """
        num_samples, num_channels, time_steps = data.shape

        if time_steps != self.original_window_size:
            raise ValueError(
                f"Expected time_steps={self.original_window_size}, got {time_steps}"
            )

        # クリップ開始位置を決定
        max_start = self.original_window_size - self.window_size

        if self.window_clip_strategy == "random":
            # 各サンプルでランダムな開始位置を選択
            start_indices = np.random.randint(0, max_start + 1, size=num_samples)
            clipped_data = np.zeros((num_samples, num_channels, self.window_size), dtype=data.dtype)
            for i, start in enumerate(start_indices):
                clipped_data[i] = data[i, :, start : start + self.window_size]

        elif self.window_clip_strategy == "center":
            # 中央から切り出し
            start = max_start // 2
            clipped_data = data[:, :, start : start + self.window_size]

        elif self.window_clip_strategy == "start":
            # 先頭から切り出し
            clipped_data = data[:, :, : self.window_size]

        elif self.window_clip_strategy == "end":
            # 末尾から切り出し
            clipped_data = data[:, :, -self.window_size :]

        else:
            raise ValueError(f"Unknown clip strategy: {self.window_clip_strategy}")

        return clipped_data

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
