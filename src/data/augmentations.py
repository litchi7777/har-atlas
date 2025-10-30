"""
時系列センサーデータ用のデータ拡張

Human Activity Recognition用の各種拡張手法を提供します。
"""

import numpy as np
from typing import Optional


class Jittering:
    """ランダムノイズを追加"""

    def __init__(self, sigma: float = 0.05):
        """
        Args:
            sigma: ノイズの標準偏差
        """
        self.sigma = sigma

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (channels, time_steps)

        Returns:
            拡張されたデータ
        """
        noise = np.random.normal(0, self.sigma, x.shape)
        return x + noise


class Scaling:
    """振幅をランダムにスケーリング"""

    def __init__(self, sigma: float = 0.1):
        """
        Args:
            sigma: スケーリング係数の標準偏差
        """
        self.sigma = sigma

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (channels, time_steps)

        Returns:
            拡張されたデータ
        """
        # チャンネルごとに異なるスケーリング係数を生成
        scale = np.random.normal(1.0, self.sigma, size=(x.shape[0], 1))
        return x * scale


class Rotation:
    """3軸センサーデータの回転"""

    def __init__(self, max_angle: float = 15.0):
        """
        Args:
            max_angle: 最大回転角度（度）
        """
        self.max_angle = max_angle

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (channels, time_steps) - channels must be multiple of 3

        Returns:
            拡張されたデータ
        """
        if x.shape[0] % 3 != 0:
            # 3軸でない場合はそのまま返す
            return x

        # Copy data
        x_aug = x.copy()

        # 各3軸ごとに回転
        num_sensors = x.shape[0] // 3
        for i in range(num_sensors):
            # Extract 3-axis data
            start_idx = i * 3
            end_idx = start_idx + 3
            xyz = x[start_idx:end_idx, :]  # (3, time_steps)

            # Random rotation angles
            angle_x = np.random.uniform(-self.max_angle, self.max_angle) * np.pi / 180
            angle_y = np.random.uniform(-self.max_angle, self.max_angle) * np.pi / 180
            angle_z = np.random.uniform(-self.max_angle, self.max_angle) * np.pi / 180

            # Rotation matrices
            Rx = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(angle_x), -np.sin(angle_x)],
                    [0, np.sin(angle_x), np.cos(angle_x)],
                ]
            )

            Ry = np.array(
                [
                    [np.cos(angle_y), 0, np.sin(angle_y)],
                    [0, 1, 0],
                    [-np.sin(angle_y), 0, np.cos(angle_y)],
                ]
            )

            Rz = np.array(
                [
                    [np.cos(angle_z), -np.sin(angle_z), 0],
                    [np.sin(angle_z), np.cos(angle_z), 0],
                    [0, 0, 1],
                ]
            )

            # Combined rotation
            R = Rz @ Ry @ Rx

            # Apply rotation: (3, 3) @ (3, time_steps) = (3, time_steps)
            xyz_rotated = R @ xyz

            # Update
            x_aug[start_idx:end_idx, :] = xyz_rotated

        return x_aug


class TimeWarping:
    """時間軸の変形（伸縮）"""

    def __init__(self, sigma: float = 0.2, knot: int = 4):
        """
        Args:
            sigma: 変形の強さ
            knot: 制御点の数
        """
        self.sigma = sigma
        self.knot = knot

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (channels, time_steps)

        Returns:
            拡張されたデータ
        """
        time_steps = x.shape[1]

        # ランダムな変形曲線を生成
        warp = np.random.normal(1.0, self.sigma, size=(self.knot + 2,))
        warp = np.cumsum(warp)  # Cumulative sum
        warp_steps = np.linspace(0, time_steps - 1, num=self.knot + 2)

        # 線形補間で時間軸全体に拡張
        from scipy.interpolate import interp1d

        warper = interp1d(warp_steps, warp, kind="cubic")

        # 新しい時間軸インデックス
        new_indices = np.arange(time_steps)
        warped_indices = warper(new_indices)

        # 正規化して範囲内に収める
        warped_indices = (warped_indices - warped_indices.min()) / (
            warped_indices.max() - warped_indices.min()
        )
        warped_indices = warped_indices * (time_steps - 1)
        warped_indices = np.clip(warped_indices, 0, time_steps - 1)

        # 補間してサンプリング
        x_warped = np.zeros_like(x)
        for c in range(x.shape[0]):
            interp = interp1d(np.arange(time_steps), x[c, :], kind="linear")
            x_warped[c, :] = interp(warped_indices)

        return x_warped


class Permutation:
    """時系列を複数セグメントに分割して並び替え"""

    def __init__(self, n_segments: int = 4):
        """
        Args:
            n_segments: 分割するセグメント数
        """
        self.n_segments = n_segments

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (channels, time_steps)

        Returns:
            拡張されたデータ
        """
        time_steps = x.shape[1]
        segment_length = time_steps // self.n_segments

        # セグメントに分割
        segments = []
        for i in range(self.n_segments):
            start = i * segment_length
            end = start + segment_length if i < self.n_segments - 1 else time_steps
            segments.append(x[:, start:end])

        # ランダムに並び替え
        np.random.shuffle(segments)

        # 連結
        x_permuted = np.concatenate(segments, axis=1)

        return x_permuted


class Reverse:
    """時系列を時間軸に沿って反転"""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (channels, time_steps)

        Returns:
            拡張されたデータ
        """
        # 時間軸（axis=1）に沿って反転
        return np.flip(x, axis=1).copy()


class MagnitudeWarping:
    """振幅の変形"""

    def __init__(self, sigma: float = 0.2, knot: int = 4):
        """
        Args:
            sigma: 変形の強さ
            knot: 制御点の数
        """
        self.sigma = sigma
        self.knot = knot

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (channels, time_steps)

        Returns:
            拡張されたデータ
        """
        time_steps = x.shape[1]

        # ランダムな変形曲線を生成
        warp = np.random.normal(1.0, self.sigma, size=(self.knot + 2,))
        warp_steps = np.linspace(0, time_steps - 1, num=self.knot + 2)

        # 線形補間
        from scipy.interpolate import interp1d

        warper = interp1d(warp_steps, warp, kind="cubic")

        # 時間軸全体に拡張
        warp_curve = warper(np.arange(time_steps))

        # 各チャンネルに適用
        x_warped = x * warp_curve[np.newaxis, :]

        return x_warped


class RandomChoice:
    """複数の拡張手法からランダムに選択"""

    def __init__(self, transforms: list, p: Optional[list] = None):
        """
        Args:
            transforms: 拡張手法のリスト
            p: 各手法を選択する確率（Noneの場合は均等）
        """
        self.transforms = transforms
        self.p = p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (channels, time_steps)

        Returns:
            拡張されたデータ
        """
        transform = np.random.choice(self.transforms, p=self.p)
        return transform(x)


class Compose:
    """複数の拡張手法を順次適用"""

    def __init__(self, transforms: list):
        """
        Args:
            transforms: 拡張手法のリスト
        """
        self.transforms = transforms

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (channels, time_steps)

        Returns:
            拡張されたデータ
        """
        for transform in self.transforms:
            x = transform(x)
        return x


class RandomApply:
    """指定された確率で拡張を適用"""

    def __init__(self, transform, p: float = 0.5):
        """
        Args:
            transform: 拡張手法
            p: 適用確率
        """
        self.transform = transform
        self.p = p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (channels, time_steps)

        Returns:
            拡張されたデータ
        """
        if np.random.random() < self.p:
            return self.transform(x)
        return x


def get_augmentation_pipeline(mode: str = "light"):
    """
    データ拡張パイプラインを取得

    Args:
        mode: 'light', 'medium', 'heavy'

    Returns:
        拡張パイプライン
    """
    if mode == "light":
        # 軽度の拡張（Fine-tuning用）
        return Compose(
            [
                RandomApply(Jittering(sigma=0.05), p=0.5),
                RandomApply(Scaling(sigma=0.1), p=0.5),
            ]
        )

    elif mode == "medium":
        # 中程度の拡張
        return Compose(
            [
                RandomApply(Jittering(sigma=0.05), p=0.5),
                RandomApply(Scaling(sigma=0.1), p=0.5),
                RandomApply(Rotation(max_angle=15.0), p=0.3),
            ]
        )

    elif mode == "heavy":
        # 強い拡張（SSL Pre-training用）
        return Compose(
            [
                RandomApply(Jittering(sigma=0.1), p=0.8),
                RandomApply(Scaling(sigma=0.2), p=0.8),
                RandomApply(Rotation(max_angle=30.0), p=0.5),
                RandomChoice(
                    [
                        TimeWarping(sigma=0.2, knot=4),
                        Permutation(n_segments=4),
                        MagnitudeWarping(sigma=0.2, knot=4),
                    ],
                    p=[0.33, 0.33, 0.34],
                ),
            ]
        )

    else:
        raise ValueError(f"Unknown augmentation mode: {mode}")


class ChannelMasking:
    """チャンネルマスキング - ランダムにチャンネルをマスク"""

    def __init__(self, mask_ratio: float = 0.15):
        """
        Args:
            mask_ratio: マスクするチャンネルの割合
        """
        self.mask_ratio = mask_ratio

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            x: (channels, time_steps)

        Returns:
            (masked_x, mask): マスクされたデータとマスク情報
                - masked_x: マスクされたデータ (channels, time_steps)
                - mask: マスクされたチャンネルのインデックス配列
        """
        n_channels = x.shape[0]
        n_masked = max(1, int(n_channels * self.mask_ratio))

        # ランダムにマスクするチャンネルを選択
        masked_channels = np.random.choice(n_channels, n_masked, replace=False)

        # マスクを作成
        masked_x = x.copy()
        masked_x[masked_channels, :] = 0.0

        return masked_x, masked_channels


class TimeMasking:
    """時間マスキング - ランダムに時間ステップをマスク"""

    def __init__(self, mask_ratio: float = 0.15):
        """
        Args:
            mask_ratio: マスクする時間ステップの割合
        """
        self.mask_ratio = mask_ratio

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            x: (channels, time_steps)

        Returns:
            (masked_x, mask): マスクされたデータとマスク情報
                - masked_x: マスクされたデータ (channels, time_steps)
                - mask: マスクされた時間ステップのインデックス配列
        """
        time_steps = x.shape[1]
        n_masked = max(1, int(time_steps * self.mask_ratio))

        # ランダムにマスクする時間ステップを選択
        masked_timesteps = np.random.choice(time_steps, n_masked, replace=False)

        # マスクを作成
        masked_x = x.copy()
        masked_x[:, masked_timesteps] = 0.0

        return masked_x, masked_timesteps


class TimeChannelMasking:
    """時間-チャンネル同時マスキング"""

    def __init__(self, time_mask_ratio: float = 0.15, channel_mask_ratio: float = 0.15):
        """
        Args:
            time_mask_ratio: マスクする時間ステップの割合
            channel_mask_ratio: マスクするチャンネルの割合
        """
        self.time_mask_ratio = time_mask_ratio
        self.channel_mask_ratio = channel_mask_ratio

    def __call__(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Args:
            x: (channels, time_steps)

        Returns:
            (masked_x, (time_mask, channel_mask)): マスクされたデータとマスク情報
        """
        n_channels = x.shape[0]
        time_steps = x.shape[1]

        # 時間マスク
        n_time_masked = max(1, int(time_steps * self.time_mask_ratio))
        masked_timesteps = np.random.choice(time_steps, n_time_masked, replace=False)

        # チャンネルマスク
        n_channel_masked = max(1, int(n_channels * self.channel_mask_ratio))
        masked_channels = np.random.choice(n_channels, n_channel_masked, replace=False)

        # マスクを作成
        masked_x = x.copy()
        masked_x[:, masked_timesteps] = 0.0  # 時間マスク
        masked_x[masked_channels, :] = 0.0  # チャンネルマスク

        return masked_x, (masked_timesteps, masked_channels)
