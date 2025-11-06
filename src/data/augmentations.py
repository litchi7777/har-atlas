"""
時系列センサーデータ用のデータ拡張

Human Activity Recognition用の各種拡張手法を提供します。
PyTorchベースのGPU対応拡張を実装。
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Union, List
import math


class Jittering:
    """ランダムノイズを追加（PyTorch対応）"""

    def __init__(self, sigma: float = 0.05):
        """
        Args:
            sigma: ノイズの標準偏差
        """
        self.sigma = sigma

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            x: (channels, time_steps) - numpy or torch tensor

        Returns:
            拡張されたデータ
        """
        if isinstance(x, torch.Tensor):
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        else:
            noise = np.random.normal(0, self.sigma, x.shape)
            return x + noise


class Scaling:
    """振幅をランダムにスケーリング（PyTorch対応）"""

    def __init__(self, sigma: float = 0.1):
        """
        Args:
            sigma: スケーリング係数の標準偏差
        """
        self.sigma = sigma

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            x: (channels, time_steps)

        Returns:
            拡張されたデータ
        """
        if isinstance(x, torch.Tensor):
            # チャンネルごとに異なるスケーリング係数を生成
            scale = torch.randn(x.shape[0], 1, device=x.device) * self.sigma + 1.0
            return x * scale
        else:
            # チャンネルごとに異なるスケーリング係数を生成
            scale = np.random.normal(1.0, self.sigma, size=(x.shape[0], 1))
            return x * scale


class Rotation:
    """3軸センサーデータの回転（PyTorch GPU対応）"""

    def __init__(self, max_angle: float = 15.0):
        """
        Args:
            max_angle: 最大回転角度（度）
        """
        self.max_angle = max_angle

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            x: (channels, time_steps) - channels must be multiple of 3

        Returns:
            拡張されたデータ
        """
        if isinstance(x, torch.Tensor):
            return self._rotation_torch(x)
        else:
            return self._rotation_numpy(x)

    def _rotation_torch(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorchベースの回転（GPU対応）"""
        if x.shape[0] % 3 != 0:
            return x

        x_aug = x.clone()
        num_sensors = x.shape[0] // 3

        for i in range(num_sensors):
            start_idx = i * 3
            end_idx = start_idx + 3
            xyz = x[start_idx:end_idx, :]  # (3, time_steps)

            # Random rotation angles (radians)
            angle_x = (torch.rand(1, device=x.device) * 2 - 1) * self.max_angle * math.pi / 180
            angle_y = (torch.rand(1, device=x.device) * 2 - 1) * self.max_angle * math.pi / 180
            angle_z = (torch.rand(1, device=x.device) * 2 - 1) * self.max_angle * math.pi / 180

            # Rotation matrices
            cos_x, sin_x = torch.cos(angle_x), torch.sin(angle_x)
            cos_y, sin_y = torch.cos(angle_y), torch.sin(angle_y)
            cos_z, sin_z = torch.cos(angle_z), torch.sin(angle_z)

            Rx = torch.tensor([
                [1, 0, 0],
                [0, cos_x, -sin_x],
                [0, sin_x, cos_x]
            ], device=x.device, dtype=x.dtype).squeeze()

            Ry = torch.tensor([
                [cos_y, 0, sin_y],
                [0, 1, 0],
                [-sin_y, 0, cos_y]
            ], device=x.device, dtype=x.dtype).squeeze()

            Rz = torch.tensor([
                [cos_z, -sin_z, 0],
                [sin_z, cos_z, 0],
                [0, 0, 1]
            ], device=x.device, dtype=x.dtype).squeeze()

            # Combined rotation
            R = torch.mm(torch.mm(Rz, Ry), Rx)

            # Apply rotation: (3, 3) @ (3, time_steps) = (3, time_steps)
            x_aug[start_idx:end_idx, :] = torch.mm(R, xyz)

        return x_aug

    def _rotation_numpy(self, x: np.ndarray) -> np.ndarray:
        """NumPyベースの回転（後方互換性）"""
        if x.shape[0] % 3 != 0:
            return x

        x_aug = x.copy()
        num_sensors = x.shape[0] // 3

        for i in range(num_sensors):
            start_idx = i * 3
            end_idx = start_idx + 3
            xyz = x[start_idx:end_idx, :]

            # Random rotation angles
            angle_x = np.random.uniform(-self.max_angle, self.max_angle) * np.pi / 180
            angle_y = np.random.uniform(-self.max_angle, self.max_angle) * np.pi / 180
            angle_z = np.random.uniform(-self.max_angle, self.max_angle) * np.pi / 180

            # Rotation matrices
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(angle_x), -np.sin(angle_x)],
                [0, np.sin(angle_x), np.cos(angle_x)]
            ])

            Ry = np.array([
                [np.cos(angle_y), 0, np.sin(angle_y)],
                [0, 1, 0],
                [-np.sin(angle_y), 0, np.cos(angle_y)]
            ])

            Rz = np.array([
                [np.cos(angle_z), -np.sin(angle_z), 0],
                [np.sin(angle_z), np.cos(angle_z), 0],
                [0, 0, 1]
            ])

            R = Rz @ Ry @ Rx
            x_aug[start_idx:end_idx, :] = R @ xyz

        return x_aug


class TimeWarping:
    """時間軸の変形（伸縮）（Torch tensor対応、SSLタスク用）"""

    def __init__(self, max_warp_factor: float = 1.5):
        """
        Args:
            max_warp_factor: 最大ワープ係数（例: 1.5 = ±50%の伸縮）
        """
        self.max_warp_factor = max_warp_factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (channels, time_steps) - Torch tensor

        Returns:
            拡張されたデータ
        """
        device = x.device
        channels, time_steps = x.shape

        # ランダムなカーブ（滑らかなノイズ）を作る
        warp_curve = torch.randn(time_steps, device=device).cumsum(0)
        warp_curve = (warp_curve - warp_curve.min()) / (warp_curve.max() - warp_curve.min())  # 0〜1に正規化
        warp_curve = warp_curve * (time_steps - 1)  # 0〜time_steps-1にスケーリング

        # ワープ量を最大ワープ係数で調整（例えば1.5なら最大50%の伸縮）
        center = torch.linspace(0, time_steps - 1, time_steps, device=device)
        indices = center + (warp_curve - center) * (self.max_warp_factor - 1)
        indices = indices.clamp(0, time_steps - 1)

        # 線形補間（非整数インデックスに対応）
        idx_low = indices.floor().long()
        idx_high = (idx_low + 1).clamp(max=time_steps - 1)
        weight = indices - idx_low

        # 各チャネルに対して補間
        x_warped = (1 - weight).unsqueeze(0) * x[:, idx_low] + weight.unsqueeze(0) * x[:, idx_high]

        return x_warped


class Permutation:
    """時系列を複数セグメントに分割して並び替え（Torch tensor対応）"""

    def __init__(self, n_segments: int = 5):
        """
        Args:
            n_segments: 分割するセグメント数
        """
        self.n_segments = n_segments

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (channels, time_steps) - Torch tensor

        Returns:
            拡張されたデータ
        """
        channels, time_steps = x.shape
        segment_length = time_steps // self.n_segments

        # セグメントに分割
        segments = []
        for i in range(self.n_segments):
            start = i * segment_length
            end = start + segment_length if i < self.n_segments - 1 else time_steps
            segments.append(x[:, start:end])

        # ランダムに並び替え
        indices = torch.randperm(self.n_segments)
        segments = [segments[i] for i in indices]

        # 連結
        x_permuted = torch.cat(segments, dim=1)

        return x_permuted


class Reverse:
    """時系列を時間軸に沿って反転（Torch tensor対応）"""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (channels, time_steps) - Torch tensor

        Returns:
            拡張されたデータ
        """
        # 時間軸（dim=1）に沿って反転
        return torch.flip(x, dims=[1])


class MagnitudeWarping:
    """振幅の変形（PyTorch対応）"""

    def __init__(self, sigma: float = 0.2, knot: int = 4):
        """
        Args:
            sigma: 変形の強さ
            knot: 制御点の数
        """
        self.sigma = sigma
        self.knot = knot

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            x: (channels, time_steps)

        Returns:
            拡張されたデータ
        """
        if isinstance(x, torch.Tensor):
            return self._warp_torch(x)
        else:
            return self._warp_numpy(x)

    def _warp_torch(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorchベースの振幅変形"""
        time_steps = x.shape[1]

        # ランダムな変形曲線を線形補間で生成
        warp = torch.randn(self.knot + 2, device=x.device) * self.sigma + 1.0
        warp_indices = torch.linspace(0, time_steps - 1, self.knot + 2, device=x.device)

        # 線形補間
        t = torch.arange(time_steps, dtype=torch.float32, device=x.device)
        warp_curve = torch.nn.functional.interpolate(
            warp.unsqueeze(0).unsqueeze(0),
            size=time_steps,
            mode="linear",
            align_corners=True
        ).squeeze()

        # 各チャンネルに適用
        return x * warp_curve.unsqueeze(0)

    def _warp_numpy(self, x: np.ndarray) -> np.ndarray:
        """NumPyベースの振幅変形"""
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
        return x * warp_curve[np.newaxis, :]


class RandomChoice:
    """複数の拡張手法からランダムに選択（PyTorch対応）"""

    def __init__(self, transforms: list, p: Optional[list] = None):
        """
        Args:
            transforms: 拡張手法のリスト
            p: 各手法を選択する確率（Noneの場合は均等）
        """
        self.transforms = transforms
        self.p = p

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            x: (channels, time_steps)

        Returns:
            拡張されたデータ
        """
        transform = np.random.choice(self.transforms, p=self.p)
        # NumPy専用の拡張(TimeWarping, Permutation)がtorch.Tensorを受け取った場合はCPU変換
        if isinstance(x, torch.Tensor) and isinstance(transform, (TimeWarping, Permutation)):
            x_np = x.cpu().numpy()
            y_np = transform(x_np)
            return torch.from_numpy(y_np).to(x.device)
        else:
            return transform(x)


class Compose:
    """複数の拡張手法を順次適用（PyTorch対応）"""

    def __init__(self, transforms: list):
        """
        Args:
            transforms: 拡張手法のリスト
        """
        self.transforms = transforms

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
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
    """指定された確率で拡張を適用（PyTorch対応）"""

    def __init__(self, transform, p: float = 0.5):
        """
        Args:
            transform: 拡張手法
            p: 適用確率
        """
        self.transform = transform
        self.p = p

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            x: (channels, time_steps)

        Returns:
            拡張されたデータ
        """
        if np.random.random() < self.p:
            return self.transform(x)
        return x


def get_augmentation_pipeline(mode: str = "light", max_epochs: int = 100):
    """
    データ拡張パイプラインを取得（改良版）

    Args:
        mode: 'light', 'medium', 'heavy', 'mtl'
        max_epochs: 最大エポック数（動的拡張用）

    Returns:
        拡張パイプライン
    """
    if mode == "light":
        # 軽度の拡張（Fine-tuning用）
        return Compose(
            [
                RandomApply(Jittering(sigma=0.05), p=0.5),
                RandomApply(Scaling(sigma=0.1), p=0.5),
                RandomApply(Amplitude(sigma=0.1), p=0.3),
            ]
        )

    elif mode == "medium":
        # 中程度の拡張
        return Compose(
            [
                RandomApply(Jittering(sigma=0.05), p=0.5),
                RandomApply(Scaling(sigma=0.1), p=0.5),
                RandomApply(Rotation(max_angle=15.0), p=0.4),
                RandomApply(Amplitude(sigma=0.15), p=0.4),
                RandomApply(PhaseShift(max_shift_ratio=0.1), p=0.3),
            ]
        )

    elif mode == "heavy":
        # 強い拡張（SSL Pre-training用）
        return Compose(
            [
                RandomApply(Jittering(sigma=0.1), p=0.8),
                RandomApply(Scaling(sigma=0.2), p=0.8),
                RandomApply(Rotation(max_angle=30.0), p=0.6),
                RandomApply(Amplitude(sigma=0.2), p=0.6),
                RandomChoice(
                    [
                        TimeWarping(sigma=0.2, knot=4),
                        Permutation(n_segments=4),
                        MagnitudeWarping(sigma=0.2, knot=4),
                    ],
                    p=[0.33, 0.33, 0.34],
                ),
                RandomApply(PhaseShift(max_shift_ratio=0.15), p=0.4),
            ]
        )

    elif mode == "mtl":
        # マルチタスク学習用の強力な拡張（PyTorch GPU対応）
        return Compose(
            [
                # 基本拡張
                RandomApply(Jittering(sigma=0.12), p=0.85),
                RandomApply(Scaling(sigma=0.25), p=0.85),
                RandomApply(Rotation(max_angle=35.0), p=0.7),

                # 高度な拡張
                RandomApply(Amplitude(sigma=0.25), p=0.7),
                RandomApply(ChannelMixing(alpha=0.5), p=0.5),
                RandomApply(ChannelArithmetic(alpha=0.3), p=0.4),

                # 時間軸拡張（PyTorch対応のみ）
                RandomApply(MagnitudeWarping(sigma=0.25, knot=4), p=0.6),
                RandomApply(PhaseShift(max_shift_ratio=0.2), p=0.5),
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


# ============================================================================
# 高度な拡張手法（リファレンスコードより）
# ============================================================================


class Amplitude:
    """振幅変調（PyTorch対応）"""

    def __init__(self, sigma: float = 0.2):
        """
        Args:
            sigma: 変調の強さ
        """
        self.sigma = sigma

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            x: (channels, time_steps)

        Returns:
            拡張されたデータ
        """
        if isinstance(x, torch.Tensor):
            alpha = 1.0 + (torch.rand(1, device=x.device) * 2 - 1) * self.sigma
            return x * alpha
        else:
            alpha = 1.0 + (np.random.random() * 2 - 1) * self.sigma
            return x * alpha


class ChannelMixing:
    """チャンネル間のミキシング（PyTorch対応）"""

    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: ミキシングの強さ
        """
        self.alpha = alpha

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            x: (channels, time_steps)

        Returns:
            拡張されたデータ
        """
        if isinstance(x, torch.Tensor):
            n_channels = x.shape[0]
            if n_channels < 2:
                return x

            # ランダムに2つのチャンネルを選択
            idx1, idx2 = torch.randperm(n_channels, device=x.device)[:2]

            # ミキシング
            mixed = x.clone()
            mixed[idx1] = self.alpha * x[idx1] + (1 - self.alpha) * x[idx2]
            mixed[idx2] = (1 - self.alpha) * x[idx1] + self.alpha * x[idx2]

            return mixed
        else:
            n_channels = x.shape[0]
            if n_channels < 2:
                return x

            # ランダムに2つのチャンネルを選択
            idx1, idx2 = np.random.choice(n_channels, 2, replace=False)

            # ミキシング
            mixed = x.copy()
            mixed[idx1] = self.alpha * x[idx1] + (1 - self.alpha) * x[idx2]
            mixed[idx2] = (1 - self.alpha) * x[idx1] + self.alpha * x[idx2]

            return mixed


class ChannelArithmetic:
    """チャンネル間の算術演算（加算/減算）（PyTorch対応）"""

    def __init__(self, alpha: float = 0.3):
        """
        Args:
            alpha: 演算の強さ
        """
        self.alpha = alpha

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            x: (channels, time_steps)

        Returns:
            拡張されたデータ
        """
        if isinstance(x, torch.Tensor):
            n_channels = x.shape[0]
            if n_channels < 2:
                return x

            # ランダムに2つのチャンネルを選択
            idx1, idx2 = torch.randperm(n_channels, device=x.device)[:2]

            # 加算または減算をランダムに選択
            result = x.clone()
            if torch.rand(1, device=x.device) > 0.5:
                result[idx1] = x[idx1] + self.alpha * x[idx2]
            else:
                result[idx1] = x[idx1] - self.alpha * x[idx2]

            return result
        else:
            n_channels = x.shape[0]
            if n_channels < 2:
                return x

            # ランダムに2つのチャンネルを選択
            idx1, idx2 = np.random.choice(n_channels, 2, replace=False)

            # 加算または減算をランダムに選択
            result = x.copy()
            if np.random.random() > 0.5:
                result[idx1] = x[idx1] + self.alpha * x[idx2]
            else:
                result[idx1] = x[idx1] - self.alpha * x[idx2]

            return result


class PhaseShift:
    """位相シフト（時間軸の巡回シフト）（PyTorch対応）"""

    def __init__(self, max_shift_ratio: float = 0.2):
        """
        Args:
            max_shift_ratio: 最大シフト量（時間ステップ数に対する割合）
        """
        self.max_shift_ratio = max_shift_ratio

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            x: (channels, time_steps)

        Returns:
            拡張されたデータ
        """
        time_steps = x.shape[1]
        max_shift = int(time_steps * self.max_shift_ratio)

        if isinstance(x, torch.Tensor):
            shift = torch.randint(-max_shift, max_shift + 1, (1,), device=x.device).item()
            return torch.roll(x, shifts=shift, dims=1)
        else:
            shift = np.random.randint(-max_shift, max_shift + 1)
            return np.roll(x, shift=shift, axis=1)


class DynamicAugmentation:
    """動的拡張 - エポックに応じて拡張強度を調整"""

    def __init__(self, transform, max_epochs: int = 100, initial_prob: float = 0.3, final_prob: float = 0.8):
        """
        Args:
            transform: 基本拡張手法
            max_epochs: 最大エポック数
            initial_prob: 初期適用確率
            final_prob: 最終適用確率
        """
        self.transform = transform
        self.max_epochs = max_epochs
        self.initial_prob = initial_prob
        self.final_prob = final_prob
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """現在のエポックを設定"""
        self.current_epoch = epoch

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            x: (channels, time_steps)

        Returns:
            拡張されたデータ
        """
        # エポックに応じた適用確率を計算
        progress = min(1.0, self.current_epoch / self.max_epochs)
        current_prob = self.initial_prob + (self.final_prob - self.initial_prob) * progress

        if np.random.random() < current_prob:
            return self.transform(x)
        return x
