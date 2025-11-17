"""
時系列センサーデータ用のモデル

1D CNNベースのHARモデルを提供します。
"""

from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Residual Block for 1D time-series data

    Architecture:
    bn-relu-conv-bn-relu-conv
      /                         \
    x --------------------------(+)->
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 2,
    ):
        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.relu(self.bn1(x))
        x = self.conv1(x)
        x = self.relu(self.bn2(x))
        x = self.conv2(x)

        x = x + identity

        return x


class Downsample(nn.Module):
    """
    Downsampling layer that applies anti-aliasing filters.

    For example, order=0 corresponds to a box filter (or average downsampling
    -- this is the same as AvgPool in PyTorch), order=1 to a triangle filter
    (or linear downsampling), order=2 to cubic downsampling, and so on.

    See https://richzhang.github.io/antialiased-cnns/ for more details.

    Args:
        channels: Number of input channels
        factor: Downsampling factor (must be > 1)
        order: Filter order (0=box, 1=triangle, 2=cubic, etc.)
    """

    def __init__(self, channels: int = None, factor: int = 2, order: int = 1):
        super(Downsample, self).__init__()

        if factor == 1:
            # No downsampling, use identity
            self.downsample = nn.Identity()
            self.channels = channels
            self.stride = 1
            self.order = 0
            self.padding = 0
            self.extra_padding = 0
        else:
            assert factor > 1, "Downsampling factor must be > 1"
            assert channels is not None, "channels must be specified when factor > 1"

            self.stride = factor
            self.channels = channels
            self.order = order

            # Figure out padding
            # The padding is given by order*(factor-1)/2
            # For odd values, we use asymmetric padding
            total_padding = order * (factor - 1)
            self.padding = total_padding // 2  # Integer division
            self.extra_padding = total_padding % 2  # Extra padding for odd cases

            # Create anti-aliasing kernel
            box_kernel = np.ones(factor)
            kernel = np.ones(factor)
            for _ in range(order):
                kernel = np.convolve(kernel, box_kernel)
            kernel /= np.sum(kernel)
            kernel = torch.Tensor(kernel)

            # Register as buffer (not a learnable parameter)
            self.register_buffer("kernel", kernel[None, None, :].repeat((channels, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x

        # Apply extra padding if needed (for odd total_padding)
        if self.extra_padding > 0:
            x = F.pad(x, (self.extra_padding, 0), mode="constant", value=0)

        return F.conv1d(
            x,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.shape[1],
        )


class Resnet(nn.Module):
    """
    Custom ResNet for time-series data

    Architecture: x->[Conv-[ResBlock]^m-BN-ReLU-Down]^n->y

    In other words:
            bn-relu-conv-bn-relu-conv                        bn-
           /                         \                      /
    x->conv --------------------------(+)-bn-relu-down-> conv ----

    Reference: https://github.com/anonymous/har-foundation
    """

    def __init__(self, n_channels: int = 3, foundationUK: bool = False, micro_window: bool = False, nano_window: bool = False):
        super(Resnet, self).__init__()

        # Architecture definition. Each tuple defines
        # a basic Resnet layer Conv-[ResBlock]^m]-BN-ReLU-Down
        # For example, (64, 5, 1, 5, 3, 1) means:
        # - 64 convolution filters
        # - kernel size of 5
        # - 1 residual block (ResBlock)
        # - ResBlock's kernel size of 5
        # - downsampling factor of 3
        # - downsampling filter order of 1

        if nano_window:
            # 超小ウィンドウ用設定（15サンプル: 15→8→4→2）
            # より浅い3層構造、ResBlockなし
            cgf = [
                (128, 3, 0, 3, 2, 1),  # kernel=3, down=2: 15→8
                (256, 3, 0, 3, 2, 1),  # kernel=3, down=2: 8→4
                (512, 3, 0, 3, 2, 1),  # kernel=3, down=2: 4→2
            ]
        elif micro_window:
            # 小ウィンドウ用設定（30サンプル: 30→15→8→4、60サンプル: 60→30→15→8）
            cgf = [
                (64, 3, 1, 3, 2, 1),   # kernel=3, down=2: 30→15 or 60→30
                (128, 3, 1, 3, 2, 1),  # kernel=3, down=2: 15→8 or 30→15
                (256, 3, 1, 3, 2, 1),  # kernel=3, down=2: 8→4 or 15→8
                (512, 3, 0, 3, 2, 1),  # kernel=3, down=2: 4→2 or 8→4
            ]
        elif foundationUK:
            cgf = [
                (64, 5, 2, 5, 2, 2),
                (128, 5, 2, 5, 2, 2),
                (256, 5, 2, 5, 5, 1),
                (512, 5, 2, 5, 5, 1),
            ]
        else:
            # デフォルト設定（150サンプル用）
            cgf = [
                (64, 5, 2, 5, 2, 2),
                (128, 5, 2, 5, 2, 2),
                (256, 5, 2, 5, 3, 1),
                (256, 5, 2, 5, 3, 1),
                (512, 5, 0, 5, 3, 1),
            ]

        self.output_dim = 512

        in_channels = n_channels
        feature_extractor = nn.Sequential()
        for i, layer_params in enumerate(cgf):
            (
                out_channels,
                conv_kernel_size,
                n_resblocks,
                resblock_kernel_size,
                downfactor,
                downorder,
            ) = layer_params
            feature_extractor.add_module(
                f"layer{i+1}",
                Resnet.make_layer(
                    in_channels,
                    out_channels,
                    conv_kernel_size,
                    n_resblocks,
                    resblock_kernel_size,
                    downfactor,
                    downorder,
                ),
            )
            in_channels = out_channels

        self.feature_extractor = feature_extractor

    @staticmethod
    def make_layer(
        in_channels: int,
        out_channels: int,
        conv_kernel_size: int,
        n_resblocks: int,
        resblock_kernel_size: int,
        downfactor: int,
        downorder: int = 1,
    ) -> nn.Sequential:
        """
        Basic layer in Resnets:

        x->[Conv-[ResBlock]^m-BN-ReLU-Down]->

        In other words:

                bn-relu-conv-bn-relu-conv
               /                         \
        x->conv --------------------------(+)-bn-relu-down->
        """

        # Check kernel sizes make sense (only odd numbers are supported)
        assert conv_kernel_size % 2, "Only odd number for conv_kernel_size supported"
        assert resblock_kernel_size % 2, "Only odd number for resblock_kernel_size supported"

        # Figure out correct paddings
        conv_padding = int((conv_kernel_size - 1) / 2)
        resblock_padding = int((resblock_kernel_size - 1) / 2)

        modules = [
            nn.Conv1d(
                in_channels,
                out_channels,
                conv_kernel_size,
                1,
                conv_padding,
                bias=False,
                padding_mode="circular",
            )
        ]

        for i in range(n_resblocks):
            modules.append(
                ResBlock(
                    out_channels,
                    out_channels,
                    resblock_kernel_size,
                    1,
                    resblock_padding,
                )
            )

        modules.append(nn.BatchNorm1d(out_channels))
        modules.append(nn.ReLU(True))
        modules.append(Downsample(out_channels, downfactor, downorder))

        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Resnet model.

        Expected input shape: (batch_size, n_channels, sequence_length)
        Example input shape: (32, 3, 100) where:
        - batch_size: 32 (number of samples)
        - n_channels: 3 (e.g., 3-axis accelerometer data)
        - sequence_length: 100 (number of time steps in the sequence)

        Output shape will vary depending on downsampling and convolution layers.
        """
        x = self.feature_extractor(x)
        return x


class Conv1DBlock(nn.Module):
    """1D Convolutional Block with BatchNorm and ReLU"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class DeepConvLSTM(nn.Module):
    """
    DeepConvLSTM: 1D CNN + LSTM for time-series classification

    Reference: Ordóñez and Roggen (2016)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_channels: Tuple[int, ...] = (64, 128, 256),
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        # Convolutional layers
        conv_layers = []
        prev_channels = in_channels
        for channels in conv_channels:
            conv_layers.append(Conv1DBlock(prev_channels, channels, kernel_size=5, padding=2))
            prev_channels = channels

        self.conv = nn.Sequential(*conv_layers)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=conv_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, time_steps)

        Returns:
            logits: (batch_size, num_classes)
        """
        # Conv layers
        x = self.conv(x)  # (batch, channels, time_steps)

        # Reshape for LSTM: (batch, time_steps, channels)
        x = x.permute(0, 2, 1)

        # LSTM
        x, _ = self.lstm(x)  # (batch, time_steps, hidden)

        # Take last timestep
        x = x[:, -1, :]  # (batch, hidden)

        # Classification
        x = self.dropout(x)
        x = self.fc(x)

        return x


class SimpleCNN(nn.Module):
    """
    Simple 1D CNN for time-series classification

    軽量で高速なベースラインモデル
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 64,
        num_blocks: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()

        # Convolutional blocks
        blocks = []
        prev_channels = in_channels
        for i in range(num_blocks):
            out_channels = base_channels * (2**i)
            blocks.append(
                Conv1DBlock(prev_channels, out_channels, kernel_size=5, padding=2, dropout=dropout)
            )
            blocks.append(nn.MaxPool1d(kernel_size=2, stride=2))
            prev_channels = out_channels

        self.conv_blocks = nn.Sequential(*blocks)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.fc = nn.Linear(prev_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, time_steps)

        Returns:
            logits: (batch_size, num_classes)
        """
        x = self.conv_blocks(x)  # (batch, channels, reduced_time)
        x = self.gap(x)  # (batch, channels, 1)
        x = x.squeeze(-1)  # (batch, channels)
        x = self.fc(x)  # (batch, num_classes)
        return x


class ResidualBlock1D(nn.Module):
    """1D Residual Block"""

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = Conv1DBlock(channels, channels, kernel_size=3, padding=1, dropout=dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(out)
        out += identity
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    """
    1D ResNet for time-series classification

    より深いモデルで複雑なパターンを学習
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 64,
        num_res_blocks: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()

        # Initial convolution
        self.init_conv = Conv1DBlock(in_channels, base_channels, kernel_size=7, padding=3)

        # Residual blocks
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResidualBlock1D(base_channels, dropout=dropout))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Downsample
        self.downsample = nn.Sequential(
            Conv1DBlock(
                base_channels,
                base_channels * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                dropout=dropout,
            ),
            Conv1DBlock(
                base_channels * 2,
                base_channels * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                dropout=dropout,
            ),
        )

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.fc = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, time_steps)

        Returns:
            logits: (batch_size, num_classes)
        """
        x = self.init_conv(x)
        x = self.res_blocks(x)
        x = self.downsample(x)
        x = self.gap(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x


class SensorSSLModel(nn.Module):
    """
    Self-Supervised Learning用のセンサーモデル

    エンコーダー + プロジェクションヘッド
    """

    def __init__(
        self,
        in_channels: int,
        backbone: str = "simple_cnn",
        projection_dim: int = 128,
        hidden_dim: int = 256,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.device = device if device is not None else torch.device("cpu")

        # Encoder (backbone)
        if backbone == "simple_cnn":
            self.encoder = SimpleCNN(in_channels, num_classes=hidden_dim, dropout=0.0)
            # Remove final FC layer
            self.encoder.fc = nn.Identity()
            encoder_dim = self.encoder.conv_blocks[-2].conv.out_channels  # Get last conv channels
        elif backbone == "resnet1d":
            self.encoder = ResNet1D(in_channels, num_classes=hidden_dim, dropout=0.0)
            self.encoder.fc = nn.Identity()
            encoder_dim = self.encoder.downsample[-1].conv.out_channels
        elif backbone == "deepconvlstm":
            self.encoder = DeepConvLSTM(in_channels, num_classes=hidden_dim, dropout=0.0)
            self.encoder.fc = nn.Identity()
            encoder_dim = self.encoder.lstm.hidden_size
        elif backbone == "resnet":
            self.encoder = Resnet(n_channels=in_channels, foundationUK=False)
            encoder_dim = self.encoder.output_dim
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # エンコーダーを指定されたデバイスに配置
        self.encoder = self.encoder.to(self.device)
        self.encoder_dim = encoder_dim

        # Projection head（指定されたデバイス上で作成）
        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        ).to(self.device)

    def forward(
        self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x1: First view (batch_size, channels, time_steps)
            x2: Second view (optional)

        Returns:
            (z1, z2): Projected representations
        """
        # Encode
        h1 = self.encoder(x1)

        # Global pooling if needed (encoder might return feature maps)
        if len(h1.shape) > 2:
            h1 = torch.mean(h1, dim=-1)

        # Project
        z1 = self.projection(h1)

        if x2 is not None:
            h2 = self.encoder(x2)
            if len(h2.shape) > 2:
                h2 = torch.mean(h2, dim=-1)
            z2 = self.projection(h2)
            return z1, z2

        return z1, None


class IntegratedSSLModel(nn.Module):
    """
    統合型SSLモデル - 複数のSSLタスクを1つのバックボーンで処理

    Integrated_SSLModelの方針を参考に、よりクリーンに実装
    各SSLタスクに対して専用のヘッドを持ち、タスク名で切り替え

    サポートするタスクタイプ:
    - binary_*: 変換予測（binary_permute, binary_reverse, binary_timewarp等）
    - masking_*: マスク再構成（masking_channel, masking_time, masking_time_channel等）
    - contrastive_*: 対照学習（contrastive_simclr等）
    """

    def __init__(
        self,
        backbone: nn.Module,
        ssl_tasks: List[str],
        hidden_dim: int = 256,
        n_channels: int = 3,
        sequence_length: int = 150,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            backbone: 共有バックボーン（ResNet等）
            ssl_tasks: SSLタスクのリスト
            hidden_dim: 中間層の次元数
            n_channels: 入力チャネル数（マスク再構成用）
            sequence_length: 入力系列長（マスク再構成用）
            device: モデルを配置するデバイス
        """
        super().__init__()

        self.device = device if device is not None else torch.device("cpu")

        # バックボーンを指定されたデバイスに配置
        self.backbone = backbone.to(self.device)
        self.input_dim = backbone.output_dim
        self.ssl_tasks = ssl_tasks
        self.hidden_dim = hidden_dim
        self.n_channels = n_channels
        self.sequence_length = sequence_length

        # タスクごとのヘッドを作成（指定されたデバイス上で）
        self.task_heads = nn.ModuleDict()
        for task in ssl_tasks:
            self.task_heads[task] = self._create_task_head(task).to(self.device)

    def _create_task_head(self, task: str) -> nn.Module:
        """タスクに応じたヘッドを作成（プレフィックスで判定）"""
        if task.startswith("binary_"):
            # Binary分類タスク（binary_permute, binary_reverse, binary_timewarp等）
            return nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(self.hidden_dim, 2),  # 2クラス分類
            )
        elif task.startswith("contrastive_"):
            # 対照学習タスク（contrastive_simclr等）
            return nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, 128),  # 128次元埋め込み
            )
        elif task.startswith("masking_"):
            # マスク再構成タスク（masking_channel, masking_time, masking_time_channel等）
            # デコーダー: 特徴マップから元の時系列を再構成
            return self._create_reconstruction_head()
        elif task.startswith("invariant_"):
            # Invariance学習タスク（invariant_orientation等）
            # Contrastive learningのためのプロジェクションヘッド
            return self._create_invariance_head(task)
        else:
            raise ValueError(
                f"Unknown task type: {task}. Use prefix like 'binary_', 'contrastive_', 'masking_', or 'invariant_'"
            )

    def _create_reconstruction_head(self) -> nn.Module:
        """マスク再構成用のデコーダーヘッドを作成"""
        # バックボーンの出力: (batch, 512, 1) または (batch, 512)
        # 目標: (batch, n_channels, sequence_length) に復元

        return nn.Sequential(
            # 1. 特徴次元を拡張
            nn.Linear(self.input_dim, self.hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            # 2. さらに拡張
            nn.Linear(self.hidden_dim * 4, self.n_channels * self.sequence_length),
            # 3. 出力はreshapeで (batch, n_channels, sequence_length) にする
        )

    def _create_invariance_head(self, task: str) -> nn.Module:
        """
        Invariance学習用のプロジェクションヘッドを作成

        Args:
            task: タスク名（invariant_orientation, invariant_scale等）

        Returns:
            プロジェクションヘッド（特徴を低次元埋め込み空間に写像）

        Note:
            SimCLRのプロジェクションヘッドと同じ構造
            出力はL2正規化前提（loss側で正規化される）
        """
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, 128),  # 128次元埋め込み
        )

    def forward(self, x: torch.Tensor, task: str) -> torch.Tensor:
        """
        Args:
            x: 入力 (batch_size, channels, time_steps)
            task: 実行するタスク名

        Returns:
            output: タスク固有の出力
        """
        # バックボーンで特徴抽出
        features = self.backbone(x)

        # タスクタイプに応じて処理を分岐
        if task.startswith("masking_"):
            # マスク再構成タスク: 時系列を復元
            if len(features.shape) > 2:
                # Global average pooling
                features = torch.mean(features, dim=-1)

            # デコーダーで復元
            output = self.task_heads[task](features)  # (batch, n_channels * sequence_length)

            # Reshape to (batch, n_channels, sequence_length)
            batch_size = output.size(0)
            output = output.view(batch_size, self.n_channels, self.sequence_length)

            return output
        else:
            # Binary分類 or 対照学習: Global pooling後に分類/埋め込み
            if len(features.shape) > 2:
                features = torch.mean(features, dim=-1)

            # タスクヘッドを適用
            output = self.task_heads[task](features)
            return output


# 後方互換性のためのエイリアス
MultiTaskSSLModel = IntegratedSSLModel


class SensorClassificationModel(nn.Module):
    """
    Fine-tuning用のセンサー分類モデル

    事前学習済みエンコーダー + 分類ヘッド
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        backbone: str = "simple_cnn",
        pretrained_path: Optional[str] = None,
        freeze_backbone: bool = False,
        foundationUK: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.device = device if device is not None else torch.device("cpu")

        # Encoder (backbone)
        if backbone == "simple_cnn":
            self.encoder = SimpleCNN(in_channels, num_classes, dropout=0.5)
            encoder_dim = self.encoder.conv_blocks[-2].conv.out_channels
            self.encoder.fc = nn.Identity()  # Remove classification head
        elif backbone == "resnet1d":
            self.encoder = ResNet1D(in_channels, num_classes, dropout=0.5)
            encoder_dim = self.encoder.downsample[-1].conv.out_channels
            self.encoder.fc = nn.Identity()
        elif backbone == "deepconvlstm":
            self.encoder = DeepConvLSTM(in_channels, num_classes, dropout=0.5)
            encoder_dim = self.encoder.lstm.hidden_size
            self.encoder.fc = nn.Identity()
        elif backbone == "resnet":
            self.encoder = Resnet(n_channels=in_channels, foundationUK=foundationUK)
            encoder_dim = self.encoder.output_dim
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # エンコーダーを指定されたデバイスに配置
        self.encoder = self.encoder.to(self.device)

        # Load pretrained weights if provided
        if pretrained_path:
            self._load_pretrained(pretrained_path)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Classification head
        # EvaClassifier_2layers スタイル: シンプルな2層構造、bias=False
        self.fc = nn.Sequential(
            nn.Linear(encoder_dim, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes, bias=False),
        ).to(self.device)

    def _load_pretrained(self, path: str):
        """事前学習済みの重みをロード"""
        checkpoint = torch.load(path, map_location=self.device)

        # SSL model の encoder の重みを抽出
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # 現在のエンコーダーのキーを取得
        encoder_keys = set(self.encoder.state_dict().keys())

        print(f"Loading pretrained weights from {path}")
        print(f"Target encoder has {len(encoder_keys)} parameters")

        # backbone.* の重みを抽出（IntegratedSSLModelの場合）
        encoder_state = {}
        for key, value in state_dict.items():
            if key.startswith("backbone."):
                # backbone.feature_extractor.* -> feature_extractor.*
                new_key = key.replace("backbone.", "")
                encoder_state[new_key] = value
            elif key.startswith("encoder."):
                # encoder.* -> * (旧形式との互換性)
                new_key = key.replace("encoder.", "")
                encoder_state[new_key] = value

        if not encoder_state:
            raise ValueError(
                f"No compatible weights found in {path}. "
                f"Expected keys starting with 'backbone.' or 'encoder.', "
                f"but found: {list(state_dict.keys())[:5]}..."
            )

        print(f"Pretrained checkpoint has {len(encoder_state)} encoder parameters")

        # キーの完全一致をチェック
        loaded_keys = set(encoder_state.keys())
        missing_keys = encoder_keys - loaded_keys
        unexpected_keys = loaded_keys - encoder_keys
        matched_keys = encoder_keys & loaded_keys

        print(f"\nKey matching report:")
        print(f"  Matched keys: {len(matched_keys)}/{len(encoder_keys)} ({100*len(matched_keys)/len(encoder_keys):.1f}%)")

        if missing_keys:
            print(f"  Missing keys in checkpoint: {len(missing_keys)}")
            print(f"    First 10: {list(missing_keys)[:10]}")

        if unexpected_keys:
            print(f"  Unexpected keys in checkpoint: {len(unexpected_keys)}")
            print(f"    First 10: {list(unexpected_keys)[:10]}")

        # 完全一致しない場合は警告
        if missing_keys or unexpected_keys:
            print(f"\n⚠️  WARNING: Weight keys do not match perfectly!")
            print(f"   This may result in suboptimal transfer learning performance.")
            print(f"   Consider checking if the pretrained model architecture matches.")
        else:
            print(f"\n✓ Perfect key match! All weights will be loaded correctly.")

        # Load into encoder
        incompatible_keys = self.encoder.load_state_dict(encoder_state, strict=False)

        print(f"\n✓ Loaded pretrained weights from {path}")
        print(f"  Successfully loaded {len(matched_keys)} parameter tensors")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, channels, time_steps)

        Returns:
            logits: (batch_size, num_classes)
        """
        # Encode
        h = self.encoder(x)

        # Global pooling if needed (for feature maps)
        if len(h.shape) > 2:
            # Global average pooling over the time dimension
            h = torch.mean(h, dim=-1)

        # Classify
        logits = self.fc(h)

        return logits


def get_sensor_model(model_name: str, in_channels: int, num_classes: int, **kwargs) -> nn.Module:
    """
    センサーモデルのファクトリー関数

    Args:
        model_name: "simple_cnn", "resnet1d", "deepconvlstm", "resnet"
        in_channels: 入力チャンネル数
        num_classes: クラス数
        **kwargs: モデル固有のパラメータ

    Returns:
        モデルインスタンス
    """
    if model_name == "simple_cnn":
        return SimpleCNN(in_channels, num_classes, **kwargs)
    elif model_name == "resnet1d":
        return ResNet1D(in_channels, num_classes, **kwargs)
    elif model_name == "deepconvlstm":
        return DeepConvLSTM(in_channels, num_classes, **kwargs)
    elif model_name == "resnet":
        # Resnetは独自のインターフェース
        foundationUK = kwargs.get("foundationUK", False)
        model = Resnet(n_channels=in_channels, foundationUK=foundationUK)
        # 分類ヘッドを追加
        model.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(model.output_dim, num_classes)
        )
        # forward メソッドをオーバーライド
        original_forward = model.forward

        def forward_with_fc(x):
            x = original_forward(x)
            x = model.fc(x)
            return x

        model.forward = forward_with_fc
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")
"""
マルチデバイス対応エンコーダーモデル
backbones.pyに追加するコード
"""

import torch
import torch.nn as nn
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class MultiDeviceSensorClassificationModel(nn.Module):
    """
    マルチデバイス対応のセンサー分類モデル

    各デバイス（装着部位）ごとに独立したエンコーダーを持ち、
    それぞれの特徴量を結合して分類する。
    任意の数のデバイスに対応（1つ、3つ、5つなど）。

    Architecture:
        - Device 1 → Encoder 1 → features_1
        - Device 2 → Encoder 2 → features_2
        - ...
        - Device N → Encoder N → features_N
        - Concat([features_1, features_2, ..., features_N]) → Classification Head → Output

    Args:
        num_devices: デバイス数（1, 3, 5など）
        in_channels: 各デバイスの入力チャネル数（通常3: x, y, z軸）
        num_classes: 分類クラス数
        backbone: バックボーンアーキテクチャ（'resnet', 'simple_cnn'など）
        pretrained_path: 事前学習済みモデルのパス（全エンコーダーに同じ重みを適用）
        freeze_backbone: エンコーダーを凍結するか
        device_names: デバイス名のリスト（ログ用、オプション）
    """

    def __init__(
        self,
        num_devices: int,
        in_channels: int,
        num_classes: int,
        backbone: str = "resnet",
        pretrained_path: Optional[str] = None,
        freeze_backbone: bool = False,
        device_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if num_devices < 1:
            raise ValueError(f"num_devices must be >= 1, got {num_devices}")

        self.num_devices = num_devices
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.device_names = device_names or [f"Device{i}" for i in range(num_devices)]
        self.device = device if device is not None else torch.device("cpu")

        # 各デバイス用のエンコーダーを作成（指定されたデバイス上で）
        # 事前学習済みモデルと同じ命名規則（feature_extractor）を使用
        self.encoders = nn.ModuleDict()
        for i in range(num_devices):
            encoder = self._create_encoder(in_channels, backbone)
            # エンコーダーを指定されたデバイスに配置
            encoder = encoder.to(self.device)
            # ModuleDictに直接エンコーダーを格納
            self.encoders[str(i)] = encoder

        # エンコーダー出力の次元数を取得（フラット化後の次元数）
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, 150).to(self.device)  # 指定されたデバイスで作成
            encoder_output = self.encoders["0"](dummy_input)
            # フラット化後の次元数を取得
            encoder_output_flat = encoder_output.view(encoder_output.size(0), -1)
            self.encoder_dim = encoder_output_flat.shape[1]

        # 結合後の特徴量次元数
        combined_dim = self.encoder_dim * num_devices

        # 分類ヘッド（指定されたデバイス上で作成）
        # EvaClassifier_2layers スタイル: シンプルな2層構造、bias=False
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes, bias=False),
        ).to(self.device)

        # 事前学習済み重みのロード（全エンコーダーに同じ重みを適用）
        # この時点でモデルは既に指定されたデバイス上にある
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)

        # バックボーン凍結
        if freeze_backbone:
            for encoder in self.encoders.values():
                for param in encoder.parameters():
                    param.requires_grad = False

        logger.info(
            f"Created MultiDeviceSensorClassificationModel: "
            f"{num_devices} devices, {backbone} backbone, "
            f"encoder_dim={self.encoder_dim}, combined_dim={combined_dim}, "
            f"device={self.device}"
        )

    def _create_encoder(self, in_channels: int, backbone: str) -> nn.Module:
        """エンコーダーを作成

        Resnetの場合、feature_extractorを直接返す（layer1, layer2等の命名を保持）
        """
        if backbone == "resnet":
            from src.models.backbones import Resnet
            resnet = Resnet(n_channels=in_channels)
            # feature_extractorを直接返す（事前学習済みモデルと同じキー構造）
            encoder = resnet.feature_extractor
        elif backbone == "simple_cnn":
            from src.models.backbones import SimpleCNN
            encoder = SimpleCNN(in_channels=in_channels, num_classes=10)
            # 最後の分類層を削除
            encoder = nn.Sequential(*list(encoder.children())[:-1])
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        return encoder

    def _load_pretrained_weights(self, pretrained_path: str):
        """事前学習済み重みを全エンコーダーにロード"""
        logger.info(f"Loading pretrained weights from {pretrained_path} to {self.device}")

        try:
            # チェックポイントを直接指定されたデバイスにロード
            # __init__で設定されたself.deviceを使用（モデルは既にそのデバイス上にある）
            checkpoint = torch.load(pretrained_path, map_location=self.device)

            # チェックポイントの形式を確認
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # backbone.プレフィックスを削除し、feature_extractorの重みのみを抽出
            # さらにfeature_extractor.プレフィックスも削除してエンコーダーに直接ロード
            encoder_state_dict = {}
            for key, value in state_dict.items():
                # backbone.プレフィックスを削除
                if key.startswith("backbone."):
                    key = key.replace("backbone.", "", 1)

                # feature_extractorの重みを抽出し、プレフィックスを削除
                if key.startswith("feature_extractor."):
                    new_key = key.replace("feature_extractor.", "", 1)
                    encoder_state_dict[new_key] = value

            # 元のcheckpointとstate_dictを削除してメモリ解放
            del checkpoint
            del state_dict

            # 全エンコーダーに同じ重みをロード
            loaded_count = 0
            for i, encoder in enumerate(self.encoders.values()):
                result = encoder.load_state_dict(encoder_state_dict, strict=False)
                if result.missing_keys or result.unexpected_keys:
                    logger.warning(
                        f"Encoder {i} ({self.device_names[i]}): "
                        f"missing_keys={result.missing_keys}, "
                        f"unexpected_keys={result.unexpected_keys}"
                    )
                loaded_count += 1

            # encoder_state_dictも削除してメモリ解放
            del encoder_state_dict

            logger.info(
                f"Successfully loaded pretrained weights to {loaded_count}/{self.num_devices} encoders"
            )

        except Exception as e:
            logger.error(f"Failed to load pretrained weights: {e}")
            raise

    def forward(self, device_data_list: List[torch.Tensor]) -> torch.Tensor:
        """
        順伝播

        Args:
            device_data_list: [torch.Tensor, ...] 各デバイスのデータ
                各Tensor形状: [batch_size, channels, time_steps]

        Returns:
            output: [batch_size, num_classes] 分類スコア
        """
        if len(device_data_list) != self.num_devices:
            raise ValueError(
                f"Expected {self.num_devices} device inputs, "
                f"got {len(device_data_list)}"
            )

        # 各デバイスのデータをエンコード
        features_list = []
        for i, (encoder, device_data) in enumerate(zip(self.encoders.values(), device_data_list)):
            features = encoder(device_data)
            # 特徴量をフラット化
            features = features.view(features.size(0), -1)
            features_list.append(features)

        # 全デバイスの特徴量を結合
        combined_features = torch.cat(features_list, dim=1)

        # 分類
        output = self.classifier(combined_features)

        return output

    def get_device_features(
        self, device_data_list: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        各デバイスの特徴量を個別に取得（解析用）

        Args:
            device_data_list: [torch.Tensor, ...] 各デバイスのデータ

        Returns:
            features_list: [torch.Tensor, ...] 各デバイスの特徴量
        """
        features_list = []
        for encoder, device_data in zip(self.encoders.values(), device_data_list):
            with torch.no_grad():
                features = encoder(device_data)
                features = features.view(features.size(0), -1)
                features_list.append(features)

        return features_list
