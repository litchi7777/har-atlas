#!/usr/bin/env python3
"""
センサーデータのリコンストラクション可視化

masking_time_channelタスクで学習したモデルを使用して、
マスクされたセンサーデータからの再構成を可視化します。

使用例:
    python analysis/main.py reconstruct \
        --model experiments/pretrain/run_*/ssl_tasks=*/models/checkpoint_epoch_99.pth \
        --num-samples 5

    # 直接実行
    python analysis/scripts/visualize_reconstruction.py \
        --model experiments/pretrain/run_*/ssl_tasks=*/models/checkpoint_epoch_99.pth
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.data.augmentations import TimeChannelMasking
from src.models.backbones import IntegratedSSLModel, Resnet


def load_reconstruction_model(
    model_path: str,
    device: str = "cuda",
) -> tuple:
    """
    リコンストラクションモデルをロード

    Args:
        model_path: チェックポイントのパス
        device: デバイス

    Returns:
        (model, config, window_size): モデル、設定、ウィンドウサイズ
    """
    model_path = Path(model_path)
    print(f"Loading model from: {model_path}")

    # 設定ファイルを読み込み
    config_path = model_path.parent.parent / "config.yaml"
    if not config_path.exists():
        # experiments/pretrain/run_*/ssl_tasks=*/models/ の場合
        config_path = model_path.parent.parent / "config.yaml"

    config = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"  Loaded config from: {config_path}")

    # ウィンドウサイズの検出
    window_size = 150  # デフォルト
    if "sensor_data" in config:
        window_size = config.get("sensor_data", {}).get("window_size", 150)

    # SSLタスクの取得
    ssl_tasks = config.get("multitask", {}).get("ssl_tasks", [])
    if not ssl_tasks:
        # デフォルト（モデルパスから推測）
        ssl_tasks = ["binary_permute", "binary_reverse", "binary_timewarp", "masking_time_channel"]

    # masking_time_channelタスクが含まれているか確認
    masking_tasks = [t for t in ssl_tasks if t.startswith("masking_")]
    if not masking_tasks:
        raise ValueError(
            f"No masking tasks found in config. Available tasks: {ssl_tasks}\n"
            "This model cannot perform reconstruction."
        )

    print(f"  SSL tasks: {ssl_tasks}")
    print(f"  Masking tasks: {masking_tasks}")
    print(f"  Window size: {window_size}")

    # アーキテクチャ選択
    nano_window = window_size < 20
    micro_window = 20 <= window_size < 100

    # チェックポイントロード
    checkpoint = torch.load(model_path, map_location=device)

    # hidden_dimの検出
    # 優先順位: 1. config, 2. binary_タスクのヘッド, 3. デフォルト
    hidden_dim = config.get("model", {}).get("feature_dim", None)

    if hidden_dim is None:
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        # binary_タスクのヘッドから検出（masking_タスクはhidden_dim*4を使用するので避ける）
        for key in state_dict:
            if "task_heads.binary_" in key and ".0.weight" in key:
                hidden_dim = state_dict[key].shape[0]
                break

    if hidden_dim is None:
        hidden_dim = 256  # デフォルト

    print(f"  Hidden dim: {hidden_dim}")

    # バックボーン作成
    backbone = Resnet(
        n_channels=3,
        foundationUK=False,
        micro_window=micro_window,
        nano_window=nano_window,
    )

    # モデル作成
    model = IntegratedSSLModel(
        backbone=backbone,
        ssl_tasks=ssl_tasks,
        hidden_dim=hidden_dim,
        n_channels=3,
        sequence_length=window_size,
        device=torch.device(device),
    )

    # 重みロード
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()

    print(f"  Model loaded successfully")

    return model, config, window_size, masking_tasks[0]


def load_sample_data(
    data_root: str = "har-unified-dataset/data/processed",
    datasets: list = None,
    num_samples: int = 5,
    window_size: int = 150,
) -> np.ndarray:
    """
    サンプルデータをロード

    Args:
        data_root: データルート
        datasets: 使用するデータセット
        num_samples: サンプル数
        window_size: ウィンドウサイズ

    Returns:
        samples: (num_samples, 3, window_size)
    """
    data_root = Path(data_root)

    if datasets is None:
        # 利用可能なデータセットを探す
        available = [d.name for d in data_root.iterdir() if d.is_dir()]
        datasets = available[:1] if available else []

    if not datasets:
        raise ValueError(f"No datasets found in {data_root}")

    samples = []
    for dataset in datasets:
        dataset_path = data_root / dataset
        if not dataset_path.exists():
            continue

        # 方法1: train/*.npzからサンプルをロード
        train_path = dataset_path / "train"
        if not train_path.exists():
            train_path = dataset_path

        npz_files = list(train_path.glob("*.npz"))
        for npz_file in npz_files[:num_samples]:
            try:
                data = np.load(npz_file)
                if "X" in data:
                    X = data["X"]  # shape: (n_windows, channels, time_steps)
                    if len(X) > 0:
                        # ランダムに1ウィンドウを選択
                        idx = np.random.randint(len(X))
                        sample = X[idx]

                        # 3チャンネルに調整
                        if sample.shape[0] > 3:
                            sample = sample[:3]
                        elif sample.shape[0] < 3:
                            # パディング
                            pad = np.zeros((3 - sample.shape[0], sample.shape[1]))
                            sample = np.concatenate([sample, pad], axis=0)

                        # ウィンドウサイズ調整
                        if sample.shape[1] != window_size:
                            # リサンプリング
                            from scipy.ndimage import zoom
                            zoom_factor = window_size / sample.shape[1]
                            sample = zoom(sample, (1, zoom_factor), order=1)

                        samples.append(sample)

                        if len(samples) >= num_samples:
                            break
            except Exception as e:
                print(f"  Warning: Could not load {npz_file}: {e}")
                continue

        if len(samples) >= num_samples:
            break

        # 方法2: USER*/LOCATION/ACC/X.npy形式（NHANES等）
        # データは既にウィンドウ化済み: (n_windows, channels, time_steps)
        if not samples:
            user_dirs = list(dataset_path.glob("USER*"))
            for user_dir in user_dirs[:num_samples * 2]:
                # ACCデータを探す
                x_files = list(user_dir.glob("**/ACC/X.npy"))
                if not x_files:
                    continue

                for x_file in x_files:
                    try:
                        X = np.load(x_file)  # shape: (n_windows, channels, time_steps)
                        if X.ndim < 2:
                            continue
                        if len(X) == 0:
                            continue

                        # 3次元の場合: (n_windows, channels, time_steps) - ウィンドウ化済み
                        if X.ndim == 3:
                            # ランダムに1ウィンドウを選択
                            idx = np.random.randint(len(X))
                            sample = X[idx]  # (channels, time_steps)

                            # 3チャンネルに調整
                            if sample.shape[0] > 3:
                                sample = sample[:3]
                            elif sample.shape[0] < 3:
                                pad = np.zeros((3 - sample.shape[0], sample.shape[1]))
                                sample = np.concatenate([sample, pad], axis=0)

                            # ウィンドウサイズ調整
                            if sample.shape[1] != window_size:
                                from scipy.ndimage import zoom
                                zoom_factor = window_size / sample.shape[1]
                                sample = zoom(sample, (1, zoom_factor), order=1)

                            samples.append(sample.astype(np.float32))

                        # 2次元の場合: (time_steps, channels) - 連続データ
                        elif X.ndim == 2:
                            if X.shape[0] < window_size:
                                continue

                            # (time_steps, channels) -> (channels, time_steps)
                            if X.shape[1] <= 6:
                                X = X.T

                            # ランダムな開始位置からウィンドウを抽出
                            start_idx = np.random.randint(0, X.shape[1] - window_size + 1)
                            sample = X[:3, start_idx:start_idx + window_size]

                            if sample.shape[0] < 3:
                                pad = np.zeros((3 - sample.shape[0], window_size))
                                sample = np.concatenate([sample, pad], axis=0)

                            samples.append(sample.astype(np.float32))

                        if len(samples) >= num_samples:
                            break
                    except Exception as e:
                        print(f"  Warning: Could not load {x_file}: {e}")
                        continue

                if len(samples) >= num_samples:
                    break

        if len(samples) >= num_samples:
            break

    if not samples:
        # フォールバック: 合成データ
        print("  No data found, generating synthetic samples...")
        for _ in range(num_samples):
            t = np.linspace(0, 4 * np.pi, window_size)
            sample = np.stack([
                np.sin(t) + np.random.randn(window_size) * 0.1,
                np.cos(t) + np.random.randn(window_size) * 0.1,
                np.sin(2 * t) * 0.5 + np.random.randn(window_size) * 0.1,
            ])
            samples.append(sample)

    return np.array(samples[:num_samples], dtype=np.float32)


def visualize_reconstruction(
    model: torch.nn.Module,
    samples: np.ndarray,
    masking_task: str,
    mask_ratio: float = 0.15,
    device: str = "cuda",
    output_path: str = None,
    show_plot: bool = True,
):
    """
    リコンストラクション結果を可視化

    Args:
        model: リコンストラクションモデル
        samples: サンプルデータ (N, 3, T)
        masking_task: マスキングタスク名
        mask_ratio: マスク比率
        device: デバイス
        output_path: 出力パス（Noneの場合は保存しない）
        show_plot: プロットを表示するか
    """
    model.eval()
    num_samples = len(samples)

    # マスキング変換
    masking = TimeChannelMasking(
        time_mask_ratio=mask_ratio,
        channel_mask_ratio=mask_ratio,
    )

    # Figure作成
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 3 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    channel_names = ["X", "Y", "Z"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, sample in enumerate(samples):
        # マスク適用
        masked_sample, (time_mask, channel_mask) = masking(sample)

        # Tensorに変換
        original_tensor = torch.FloatTensor(sample).unsqueeze(0).to(device)
        masked_tensor = torch.FloatTensor(masked_sample).unsqueeze(0).to(device)

        # リコンストラクション
        with torch.no_grad():
            reconstructed = model(masked_tensor, masking_task)
            reconstructed = reconstructed.cpu().numpy()[0]

        # プロット
        time_steps = sample.shape[1]
        t = np.arange(time_steps)

        # 1. Original
        ax = axes[i, 0]
        for ch in range(3):
            ax.plot(t, sample[ch], color=colors[ch], label=channel_names[ch], alpha=0.8)
        ax.set_title(f"Sample {i+1}: Original")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # 2. Masked
        ax = axes[i, 1]
        for ch in range(3):
            ax.plot(t, masked_sample[ch], color=colors[ch], label=channel_names[ch], alpha=0.8)
        # マスク領域をハイライト
        for tm in time_mask:
            ax.axvline(x=tm, color="red", alpha=0.1, linewidth=1)
        ax.set_title(f"Masked (time={len(time_mask)}, ch={len(channel_mask)})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

        # 3. Reconstructed
        ax = axes[i, 2]
        for ch in range(3):
            ax.plot(t, reconstructed[ch], color=colors[ch], label=channel_names[ch], alpha=0.8)
        ax.set_title("Reconstructed")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

        # 4. Comparison (overlay)
        ax = axes[i, 3]
        for ch in range(3):
            ax.plot(t, sample[ch], color=colors[ch], alpha=0.5, linestyle="-", label=f"{channel_names[ch]} (orig)")
            ax.plot(t, reconstructed[ch], color=colors[ch], alpha=0.8, linestyle="--", label=f"{channel_names[ch]} (rec)")
        ax.set_title("Comparison")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

        # MSE計算
        mse = np.mean((sample - reconstructed) ** 2)
        axes[i, 3].text(
            0.02, 0.98, f"MSE: {mse:.4f}",
            transform=axes[i, 3].transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved figure to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig


def main():
    parser = argparse.ArgumentParser(description="センサーデータのリコンストラクション可視化")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="事前学習済みモデルのパス",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="har-unified-dataset/data/processed",
        help="データルート",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="使用するデータセット",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="可視化するサンプル数",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.15,
        help="マスク比率",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="デバイス",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/figures",
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="プロットを表示しない",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("センサーデータ リコンストラクション可視化")
    print("=" * 80)

    # モデルロード
    model, config, window_size, masking_task = load_reconstruction_model(
        args.model,
        device=args.device,
    )

    # データセット設定を設定から取得
    datasets = args.datasets
    if datasets is None:
        datasets = config.get("sensor_data", {}).get("datasets", None)

    # サンプルデータロード
    print("\nLoading sample data...")
    samples = load_sample_data(
        data_root=args.data_root,
        datasets=datasets,
        num_samples=args.num_samples,
        window_size=window_size,
    )
    print(f"  Loaded {len(samples)} samples, shape: {samples.shape}")

    # 可視化
    print("\nGenerating reconstruction visualization...")
    output_path = Path(args.output_dir) / "reconstruction.png"

    visualize_reconstruction(
        model=model,
        samples=samples,
        masking_task=masking_task,
        mask_ratio=args.mask_ratio,
        device=args.device,
        output_path=str(output_path),
        show_plot=not args.no_show,
    )

    print("\n✓ 完了")


if __name__ == "__main__":
    main()
