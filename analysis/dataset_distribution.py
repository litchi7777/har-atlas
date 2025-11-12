"""
データセット分布の可視化と分析

このスクリプトは、各データセット・身体部位のセンサーデータ分布を可視化し、
データセット間の特性の違いを分析します。

使用方法:
    python analysis/dataset_distribution.py --dataset dsads --location Torso
    python analysis/dataset_distribution.py --all  # 全データセット・部位を分析

出力:
    analysis/figures/ 以下に可視化結果を保存
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import json

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "har-unified-dataset" / "src"))

from dataset_info import DATASETS

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# スタイル設定
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def load_sensor_data(dataset_name, location):
    """
    指定されたデータセット・部位のセンサーデータを読み込む

    データ構造: data_root/dataset/USER*/location/ACC/X.npy (and Y.npy)

    Args:
        dataset_name: データセット名（例: 'dsads'）
        location: 身体部位（例: 'Torso'）

    Returns:
        X: センサーデータ (N, 3, T)
        y: ラベル (N,)
        metadata: メタデータ辞書
    """
    import glob

    data_root = project_root / "har-unified-dataset" / "data" / "processed"
    dataset_path = data_root / dataset_name

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # パターン: data_root/dataset/USER*/location/ACC/X.npy
    pattern = str(dataset_path / f"*/{location}/ACC/X.npy")
    x_paths = sorted(glob.glob(pattern))

    if not x_paths:
        raise FileNotFoundError(f"No X.npy files found for pattern: {pattern}")

    # 全ユーザーのデータを統合
    all_X = []
    all_y = []

    for x_path in x_paths:
        y_path = x_path.replace("/X.npy", "/Y.npy")

        if not Path(y_path).exists():
            print(f"Warning: Y.npy not found for {x_path}, skipping")
            continue

        # データ読み込み
        X_data = np.load(x_path, mmap_mode='r')  # メモリ効率化
        y_data = np.load(y_path, mmap_mode='r')

        # Y.npyはラベル（1次元）を想定
        if y_data.ndim != 1:
            print(f"Warning: Y.npy is not 1D at {y_path}, skipping")
            continue

        all_X.append(X_data)
        all_y.append(y_data)

    if not all_X:
        raise ValueError(f"No valid data found for {dataset_name}/{location}")

    # 結合
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # メタデータ読み込み
    metadata_path = dataset_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    return X, y, metadata


def plot_sensor_distribution(X, y, dataset_name, location, labels_dict, output_dir):
    """
    センサーデータの分布を可視化

    Args:
        X: センサーデータ (N, 3, T)
        y: ラベル (N,)
        dataset_name: データセット名
        location: 身体部位
        labels_dict: ラベル辞書 {0: 'walking', 1: 'sitting', ...}
        output_dir: 出力ディレクトリ
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{dataset_name.upper()} - {location}: Sensor Data Distribution',
                 fontsize=16, fontweight='bold')

    # 1. 各軸の振幅分布（ヒストグラム）
    ax = axes[0, 0]
    for i, axis_name in enumerate(['X', 'Y', 'Z']):
        data = X[:, i, :].flatten()
        ax.hist(data, bins=100, alpha=0.5, label=f'{axis_name}-axis', density=True)

    ax.set_xlabel('Acceleration (G)')
    ax.set_ylabel('Density')
    ax.set_title('Amplitude Distribution (All Axes)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 各軸の統計量（箱ひげ図）
    ax = axes[0, 1]
    box_data = [X[:, i, :].flatten() for i in range(3)]
    bp = ax.boxplot(box_data, labels=['X', 'Y', 'Z'], patch_artist=True)

    colors = ['#ff9999', '#66b3ff', '#99ff99']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Acceleration (G)')
    ax.set_title('Statistical Summary by Axis')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. 活動クラス別のサンプル時系列
    ax = axes[1, 0]
    unique_labels = np.unique(y)
    unique_labels = unique_labels[unique_labels >= 0]  # 負のラベルを除外

    n_samples_per_class = min(3, len(unique_labels))
    for i, label in enumerate(unique_labels[:n_samples_per_class]):
        indices = np.where(y == label)[0]
        if len(indices) > 0:
            sample_idx = indices[0]
            # X軸のみプロット（見やすさのため）
            time_series = X[sample_idx, 0, :]
            label_name = labels_dict.get(int(label), f'Class {label}')
            ax.plot(time_series, label=label_name, alpha=0.8, linewidth=1.5)

    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Acceleration (G) - X-axis')
    ax.set_title('Sample Time Series by Activity Class')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4. クラス分布（サンプル数）
    ax = axes[1, 1]
    unique_labels_all = np.unique(y)
    unique_labels_all = unique_labels_all[unique_labels_all >= 0]

    class_counts = [np.sum(y == label) for label in unique_labels_all]
    class_names = [labels_dict.get(int(label), f'Class {label}')[:20]  # 20文字に制限
                   for label in unique_labels_all]

    bars = ax.barh(range(len(class_counts)), class_counts)
    ax.set_yticks(range(len(class_counts)))
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel('Number of Samples')
    ax.set_title('Activity Class Distribution')
    ax.grid(True, alpha=0.3, axis='x')

    # カラーマップ
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    plt.tight_layout()

    # 保存
    output_path = output_dir / f'{dataset_name}_{location}_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')
    plt.close()


def plot_dataset_comparison(all_data, output_dir):
    """
    複数データセット間の比較可視化

    Args:
        all_data: {(dataset, location): {'X': X, 'y': y, 'metadata': metadata}}
        output_dir: 出力ディレクトリ
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cross-Dataset Comparison', fontsize=16, fontweight='bold')

    # 1. 各データセットの振幅範囲比較
    ax = axes[0, 0]
    dataset_stats = []
    labels = []

    for (dataset, location), data in sorted(all_data.items()):
        X = data['X']
        # 全軸の標準偏差の平均
        std_val = np.mean([X[:, i, :].std() for i in range(3)])
        dataset_stats.append(std_val)
        labels.append(f'{dataset}/{location}'[:25])

    bars = ax.barh(range(len(dataset_stats)), dataset_stats)
    ax.set_yticks(range(len(dataset_stats)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Average Standard Deviation (G)')
    ax.set_title('Signal Variability by Dataset')
    ax.grid(True, alpha=0.3, axis='x')

    colors = plt.cm.tab20(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # 2. クラス数比較
    ax = axes[0, 1]
    class_counts = []
    labels_2 = []

    for (dataset, location), data in sorted(all_data.items()):
        y = data['y']
        n_classes = len(np.unique(y[y >= 0]))
        class_counts.append(n_classes)
        labels_2.append(f'{dataset}/{location}'[:25])

    bars2 = ax.barh(range(len(class_counts)), class_counts)
    ax.set_yticks(range(len(class_counts)))
    ax.set_yticklabels(labels_2, fontsize=8)
    ax.set_xlabel('Number of Classes')
    ax.set_title('Activity Classes by Dataset')
    ax.grid(True, alpha=0.3, axis='x')

    for bar, color in zip(bars2, colors):
        bar.set_color(color)

    # 3. サンプル数比較
    ax = axes[1, 0]
    sample_counts = []
    labels_3 = []

    for (dataset, location), data in sorted(all_data.items()):
        X = data['X']
        sample_counts.append(len(X))
        labels_3.append(f'{dataset}/{location}'[:25])

    bars3 = ax.barh(range(len(sample_counts)), sample_counts)
    ax.set_yticks(range(len(sample_counts)))
    ax.set_yticklabels(labels_3, fontsize=8)
    ax.set_xlabel('Number of Samples')
    ax.set_title('Dataset Size Comparison')
    ax.grid(True, alpha=0.3, axis='x')

    for bar, color in zip(bars3, colors):
        bar.set_color(color)

    # 4. 振幅分布の比較（バイオリンプロット）
    ax = axes[1, 1]

    # 各データセットからランダムサンプリング
    violin_data = []
    violin_labels = []

    for (dataset, location), data in list(sorted(all_data.items()))[:8]:  # 最大8データセット
        X = data['X']
        # X軸のみ、ランダムに1000サンプル
        samples = X[:, 0, :].flatten()
        if len(samples) > 1000:
            samples = np.random.choice(samples, 1000, replace=False)
        violin_data.append(samples)
        violin_labels.append(f'{dataset}/{location}'[:15])

    if violin_data:
        parts = ax.violinplot(violin_data, positions=range(len(violin_data)),
                              showmeans=True, showmedians=True)
        ax.set_xticks(range(len(violin_data)))
        ax.set_xticklabels(violin_labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Acceleration (G) - X-axis')
        ax.set_title('Amplitude Distribution Comparison')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 保存
    output_path = output_dir / 'cross_dataset_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')
    plt.close()


def print_statistics(X, y, dataset_name, location, labels_dict):
    """データセットの統計情報を出力"""
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name.upper()} - {location}")
    print(f"{'='*80}")
    print(f"Data shape: {X.shape} (N_samples, N_channels, N_timesteps)")
    print(f"Label shape: {y.shape}")
    print(f"Number of samples: {len(X)}")
    print(f"Time steps: {X.shape[2]}")

    # ラベル情報
    unique_labels = np.unique(y)
    valid_labels = unique_labels[unique_labels >= 0]
    print(f"\nNumber of classes: {len(valid_labels)}")
    print(f"Class distribution:")
    for label in valid_labels:
        count = np.sum(y == label)
        percentage = count / len(y) * 100
        label_name = labels_dict.get(int(label), f'Class {label}')
        print(f"  {label_name:30s}: {count:6d} samples ({percentage:5.2f}%)")

    # センサーデータ統計
    print(f"\nSensor data statistics (all axes):")
    for i, axis_name in enumerate(['X-axis', 'Y-axis', 'Z-axis']):
        data = X[:, i, :].flatten()
        print(f"  {axis_name}:")
        print(f"    Mean:   {np.mean(data):8.4f} G")
        print(f"    Std:    {np.std(data):8.4f} G")
        print(f"    Min:    {np.min(data):8.4f} G")
        print(f"    Max:    {np.max(data):8.4f} G")
        print(f"    Range:  {np.max(data) - np.min(data):8.4f} G")


def main():
    parser = argparse.ArgumentParser(description='Visualize dataset distributions')
    parser.add_argument('--dataset', type=str, help='Dataset name (e.g., dsads)')
    parser.add_argument('--location', type=str, help='Body location (e.g., Torso)')
    parser.add_argument('--all', action='store_true', help='Analyze all datasets')
    parser.add_argument('--compare', action='store_true', help='Generate cross-dataset comparison')

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = project_root / "analysis" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all or args.compare:
        # 利用可能なデータセット・部位を検出
        import glob

        data_root = project_root / "har-unified-dataset" / "data" / "processed"
        all_data = {}

        for dataset_path in sorted(data_root.iterdir()):
            if not dataset_path.is_dir():
                continue

            dataset_name = dataset_path.name

            # メタデータからラベル情報取得
            dataset_upper = dataset_name.upper()
            labels_dict = DATASETS.get(dataset_upper, {}).get('labels', {})

            # USER*/location/ACC/X.npy のパターンで検索
            pattern = str(dataset_path / "*/*/ACC/X.npy")
            x_paths = glob.glob(pattern)

            # 身体部位のリストを抽出
            locations = set()
            for x_path in x_paths:
                parts = Path(x_path).parts
                # .../USER00001/Torso/ACC/X.npy から Torso を抽出
                if len(parts) >= 3:
                    location = parts[-3]  # ACC の2つ上
                    locations.add(location)

            # 各身体部位でデータ読み込み
            for location in sorted(locations):
                try:
                    X, y, metadata = load_sensor_data(dataset_name, location)
                    all_data[(dataset_name, location)] = {
                        'X': X, 'y': y, 'metadata': metadata
                    }

                    if args.all:
                        # 個別の可視化
                        print_statistics(X, y, dataset_name, location, labels_dict)
                        plot_sensor_distribution(X, y, dataset_name, location,
                                               labels_dict, output_dir)

                except Exception as e:
                    print(f"Error processing {dataset_name}/{location}: {e}")

        if args.compare and all_data:
            # 比較可視化
            plot_dataset_comparison(all_data, output_dir)

        print(f"\n{'='*80}")
        print(f"Total datasets analyzed: {len(all_data)}")
        print(f"Figures saved to: {output_dir}")
        print(f"{'='*80}\n")

    elif args.dataset and args.location:
        # 個別データセット分析
        dataset_upper = args.dataset.upper()
        labels_dict = DATASETS.get(dataset_upper, {}).get('labels', {})

        X, y, metadata = load_sensor_data(args.dataset, args.location)

        print_statistics(X, y, args.dataset, args.location, labels_dict)
        plot_sensor_distribution(X, y, args.dataset, args.location,
                               labels_dict, output_dir)

        print(f"\nFigure saved to: {output_dir}")

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python analysis/dataset_distribution.py --dataset dsads --location Torso")
        print("  python analysis/dataset_distribution.py --all")
        print("  python analysis/dataset_distribution.py --all --compare")


def main_with_args(args):
    """引数オブジェクトを受け取って実行（analyze.pyから呼ばれる用）"""
    # 出力ディレクトリ作成
    output_dir = project_root / "analysis" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all or args.compare:
        # 利用可能なデータセット・部位を検出
        import glob

        data_root = project_root / "har-unified-dataset" / "data" / "processed"
        all_data = {}

        for dataset_path in sorted(data_root.iterdir()):
            if not dataset_path.is_dir():
                continue

            dataset_name = dataset_path.name

            # メタデータからラベル情報取得
            dataset_upper = dataset_name.upper()
            labels_dict = DATASETS.get(dataset_upper, {}).get('labels', {})

            # USER*/location/ACC/X.npy のパターンで検索
            pattern = str(dataset_path / "*/*/ACC/X.npy")
            x_paths = glob.glob(pattern)

            # 身体部位のリストを抽出
            locations = set()
            for x_path in x_paths:
                parts = Path(x_path).parts
                # .../USER00001/Torso/ACC/X.npy から Torso を抽出
                if len(parts) >= 3:
                    location = parts[-3]  # ACC の2つ上
                    locations.add(location)

            # 各身体部位でデータ読み込み
            for location in sorted(locations):
                try:
                    X, y, metadata = load_sensor_data(dataset_name, location)
                    all_data[(dataset_name, location)] = {
                        'X': X, 'y': y, 'metadata': metadata
                    }

                    if args.all:
                        # 個別の可視化
                        print_statistics(X, y, dataset_name, location, labels_dict)
                        plot_sensor_distribution(X, y, dataset_name, location,
                                               labels_dict, output_dir)

                except Exception as e:
                    print(f"Error processing {dataset_name}/{location}: {e}")

        if args.compare and all_data:
            # 比較可視化
            plot_dataset_comparison(all_data, output_dir)

        print(f"\n{'='*80}")
        print(f"Total datasets analyzed: {len(all_data)}")
        print(f"Figures saved to: {output_dir}")
        print(f"{'='*80}\n")

    elif args.dataset and args.location:
        # 個別データセット分析
        dataset_upper = args.dataset.upper()
        labels_dict = DATASETS.get(dataset_upper, {}).get('labels', {})

        X, y, metadata = load_sensor_data(args.dataset, args.location)

        print_statistics(X, y, args.dataset, args.location, labels_dict)
        plot_sensor_distribution(X, y, args.dataset, args.location,
                               labels_dict, output_dir)

        print(f"\nFigure saved to: {output_dir}")


if __name__ == '__main__':
    main()
