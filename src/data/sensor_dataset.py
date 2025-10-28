"""
時系列センサーデータ用のデータセット

Human Activity Recognition用のセンサーデータを扱います。
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SensorDataset(Dataset):
    """
    時系列センサーデータセット

    複数のセンサー部位からデータを読み込み、ユーザーIDベースで分割します。
    """

    def __init__(
        self,
        data_path: str,
        sensor_locations: List[str],
        user_ids: List[int],
        mode: str = "train",
        transform: Optional[Any] = None,
    ):
        """
        Args:
            data_path: データディレクトリのパス（例: /path/to/DSADS）
            sensor_locations: センサー部位のリスト（例: ['LeftArm', 'RightArm', 'Torso']）
            user_ids: 使用するユーザーIDのリスト
            mode: 'train', 'val', 'test'
            transform: データ拡張（オプション）

        Raises:
            ValueError: データパスが存在しない、またはデータが見つからない場合
        """
        self.data_path = Path(data_path)
        self.sensor_locations = sensor_locations
        self.user_ids = user_ids
        self.mode = mode
        self.transform = transform

        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {data_path}")

        # データをロード
        self.X, self.Y, self.U = self._load_data()

        # ユーザーIDでフィルタリング
        mask = np.isin(self.U, user_ids)
        self.X = self.X[mask]
        self.Y = self.Y[mask]
        self.U = self.U[mask]

        if len(self.X) == 0:
            raise ValueError(
                f"No data found for user IDs {user_ids}. "
                f"Available user IDs: {np.unique(self.U)}"
            )

        # クラス情報
        self.num_classes = len(np.unique(self.Y))
        self.class_counts = np.bincount(self.Y.astype(int))

        logger.info(
            f"Loaded {len(self.X)} samples for {mode} "
            f"(users: {user_ids}, classes: {self.num_classes})"
        )

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        複数のセンサー部位からデータを読み込み、結合

        Returns:
            (X, Y, U): センサーデータ、ラベル、ユーザーID
        """
        X_list = []
        Y_ref = None
        U_ref = None

        for location in self.sensor_locations:
            location_path = self.data_path / location

            if not location_path.exists():
                raise ValueError(f"Sensor location not found: {location_path}")

            # 各ファイルをロード
            X_loc = np.load(location_path / "X.npy")
            Y_loc = np.load(location_path / "Y.npy")
            U_loc = np.load(location_path / "U.npy")

            # ラベルとユーザーIDの一貫性チェック
            if Y_ref is None:
                Y_ref = Y_loc
                U_ref = U_loc
            else:
                if not np.array_equal(Y_loc, Y_ref):
                    raise ValueError(f"Labels mismatch for {location}")
                if not np.array_equal(U_loc, U_ref):
                    raise ValueError(f"User IDs mismatch for {location}")

            X_list.append(X_loc)

        # センサーデータを結合 (num_samples, channels, time_steps)
        # 各センサーは (num_samples, 3, 150)
        # 結合後: (num_samples, 3*num_sensors, 150)
        X = np.concatenate(X_list, axis=1)

        logger.info(
            f"Loaded data from {len(self.sensor_locations)} sensors: " f"{self.sensor_locations}"
        )
        logger.info(f"Data shape: {X.shape}")

        return X, Y_ref, U_ref

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        サンプルを取得

        Args:
            idx: サンプルインデックス

        Returns:
            (data, label): センサーデータとラベルのタプル
        """
        # データとラベルを取得
        x = self.X[idx].astype(np.float32)  # (channels, time_steps)
        y = int(self.Y[idx])

        # データ拡張（学習時のみ）
        if self.transform is not None and self.mode == "train":
            x = self.transform(x)
            # データ拡張後にfloat32に再変換（augmentationがfloat64を返す場合があるため）
            x = x.astype(np.float32)

        # Tensorに変換
        x = torch.from_numpy(x)

        return x, y

    def get_num_channels(self) -> int:
        """入力チャンネル数を取得"""
        return self.X.shape[1]

    def get_sequence_length(self) -> int:
        """時系列長を取得"""
        return self.X.shape[2]

    def get_class_weights(self) -> torch.Tensor:
        """
        クラスの不均衡に対する重みを計算

        Returns:
            クラス重み（不均衡対策用）
        """
        class_counts = np.bincount(self.Y.astype(int))
        total = len(self.Y)
        weights = total / (len(class_counts) * class_counts + 1e-6)
        return torch.FloatTensor(weights)


class PretrainSensorDataset(SensorDataset):
    """
    Self-Supervised Pre-training用のセンサーデータセット

    同じサンプルの2つの拡張ビューを返します。
    """

    def __init__(
        self,
        data_path: str,
        sensor_locations: List[str],
        user_ids: List[int],
        transform: Optional[Any] = None,
    ):
        super().__init__(
            data_path=data_path,
            sensor_locations=sensor_locations,
            user_ids=user_ids,
            mode="pretrain",
            transform=transform,
        )

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], int]:
        """
        2つの拡張ビューを返す（対照学習用）

        Returns:
            ([view1, view2], dummy_label): 2つの拡張ビューとダミーラベル
        """
        # データを取得
        x = self.X[idx].astype(np.float32)

        # 2つの異なる拡張ビューを生成
        if self.transform is not None:
            view1_np = self.transform(x.copy()).astype(np.float32)
            view2_np = self.transform(x.copy()).astype(np.float32)
            view1 = torch.from_numpy(view1_np)
            view2 = torch.from_numpy(view2_np)
        else:
            # 拡張なしの場合はそのまま返す
            x_tensor = torch.from_numpy(x)
            view1 = x_tensor
            view2 = x_tensor

        return [view1, view2], 0  # ダミーラベル


class MultiTaskPretrainDataset(SensorDataset):
    """
    マルチタスク学習用のPre-trainingデータセット

    3つの独立した2クラス分類タスク:
    - タスク1: Permute適用 vs 非適用
    - タスク2: Reverse適用 vs 非適用
    - タスク3: TimeWarp適用 vs 非適用
    """

    def __init__(
        self,
        data_path: str,
        sensor_locations: List[str],
        user_ids: List[int],
        specific_transforms: Optional[Dict[str, Any]] = None,
        apply_prob: float = 0.5,
    ):
        """
        Args:
            data_path: データディレクトリのパス
            sensor_locations: センサー部位のリスト
            user_ids: 使用するユーザーIDのリスト
            specific_transforms: 拡張辞書 {'permute': Transform, 'reverse': Transform, 'timewarp': Transform}
            apply_prob: 各拡張を適用する確率
        """
        super().__init__(
            data_path=data_path,
            sensor_locations=sensor_locations,
            user_ids=user_ids,
            mode="pretrain",
            transform=None,
        )

        self.specific_transforms = specific_transforms or {}
        self.apply_prob = apply_prob
        self.aug_names = ["permute", "reverse", "timewarp"]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        マルチタスク用のデータを返す

        Returns:
            (x, labels):
                - x: 拡張が適用されたデータ [channels, time_steps]
                - labels: 3つの2値ラベル [permute_applied, reverse_applied, timewarp_applied]
        """
        # データを取得
        x = self.X[idx].astype(np.float32)

        # 各拡張をランダムに適用
        labels = []
        for aug_name in self.aug_names:
            # ランダムに適用するか決定
            apply = np.random.random() < self.apply_prob
            labels.append(1 if apply else 0)

            # 拡張を適用
            if apply and aug_name in self.specific_transforms:
                x = self.specific_transforms[aug_name](x).astype(np.float32)

        # Tensorに変換
        x_tensor = torch.from_numpy(x)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        return x_tensor, labels_tensor


def get_available_users(data_path: str, sensor_location: str = "LeftArm") -> List[int]:
    """
    利用可能なユーザーIDを取得

    Args:
        data_path: データディレクトリのパス
        sensor_location: 参照するセンサー部位

    Returns:
        ユーザーIDのリスト
    """
    U = np.load(Path(data_path) / sensor_location / "U.npy")
    return sorted(np.unique(U).tolist())
