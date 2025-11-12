"""
データセット品質の分析

このスクリプトは、データセットの品質を多角的に分析します：
- 欠損値・異常値の検出
- クラスバランスの評価
- 信号品質の評価（SNR、周波数特性等）
- ユーザー間のばらつき分析

使用方法:
    # 単一データセットの分析
    python analysis/data_quality.py --dataset dsads --location Torso

    # 全データセットの比較
    python analysis/data_quality.py --all --compare

    # 特定のデータセットのみ
    python analysis/data_quality.py --datasets dsads mhealth pamap2

出力:
    analysis/figures/data_quality/ 以下に可視化結果を保存
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import stats, signal
import glob

# 共通ユーティリティをインポート
from utils import (
    get_project_root,
    get_processed_data_root,
    find_dataset_location_pairs,
    load_sensor_data,
    get_label_dict,
    print_section_header,
    print_subsection_header
)

# スタイル設定
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.0)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def detect_outliers(X: np.ndarray, method: str = 'iqr', threshold: float = 3.0) -> np.ndarray:
    """
    異常値を検出

    Args:
        X: センサーデータ (N, 3, T)
        method: 'iqr' or 'zscore'
        threshold: 閾値（IQRの場合は倍率、z-scoreの場合は標準偏差数）

    Returns:
        outlier_mask: 異常値マスク (N,) True=異常値
    """
    # 各サンプルの統計量を計算
    sample_means = X.mean(axis=(1, 2))  # (N,)
    sample_stds = X.std(axis=(1, 2))    # (N,)
    sample_maxs = X.max(axis=(1, 2))    # (N,)
    sample_mins = X.min(axis=(1, 2))    # (N,)

    outlier_mask = np.zeros(len(X), dtype=bool)

    if method == 'iqr':
        # IQR法
        for stat in [sample_means, sample_stds, sample_maxs, sample_mins]:
            q1 = np.percentile(stat, 25)
            q3 = np.percentile(stat, 75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            outlier_mask |= (stat < lower) | (stat > upper)

    elif method == 'zscore':
        # Z-score法
        for stat in [sample_means, sample_stds, sample_maxs, sample_mins]:
            z_scores = np.abs(stats.zscore(stat))
            outlier_mask |= z_scores > threshold

    return outlier_mask


def calculate_snr(X: np.ndarray) -> float:
    """
    信号対雑音比（SNR）を計算

    Args:
        X: センサーデータ (N, 3, T)

    Returns:
        SNR (dB)
    """
    # 信号パワー: 各サンプルの分散の平均
    signal_power = X.var(axis=2).mean()

    # ノイズパワー: 高周波成分（簡易的にハイパスフィルタ適用後の分散）
    # ここでは差分の分散をノイズとみなす
    noise_power = np.diff(X, axis=2).var()

    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        snr = float('inf')

    return snr


def analyze_frequency_spectrum(X: np.ndarray, sampling_rate: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    周波数スペクトルを分析

    Args:
        X: センサーデータ (N, 3, T)
        sampling_rate: サンプリングレート (Hz)

    Returns:
        freqs: 周波数 (Hz)
        power_spectrum: パワースペクトル
    """
    # 各軸のFFTを計算して平均
    fft_results = []

    for i in range(X.shape[1]):  # 各軸
        # 複数サンプルの平均スペクトル
        sample_spectra = []
        for j in range(min(100, X.shape[0])):  # 最大100サンプル
            fft = np.fft.rfft(X[j, i, :])
            power = np.abs(fft) ** 2
            sample_spectra.append(power)

        avg_spectrum = np.mean(sample_spectra, axis=0)
        fft_results.append(avg_spectrum)

    # 3軸の平均
    power_spectrum = np.mean(fft_results, axis=0)
    freqs = np.fft.rfftfreq(X.shape[2], 1.0 / sampling_rate)

    return freqs, power_spectrum


def plot_data_quality_report(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    location: str,
    label_dict: Dict[int, str],
    output_path: Path
):
    """
    データ品質レポートをプロット

    Args:
        X: センサーデータ (N, 3, T)
        y: ラベル (N,)
        dataset_name: データセット名
        location: 身体部位
        label_dict: ラベル辞書
        output_path: 出力パス
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle(f'Data Quality Report: {dataset_name.upper()} - {location}',
                 fontsize=16, fontweight='bold')

    # 1. クラスバランス
    ax = fig.add_subplot(gs[0, 0])
    unique_labels, counts = np.unique(y, return_counts=True)
    class_names = [label_dict.get(int(l), f"Class {l}")[:15] for l in unique_labels]

    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    ax.barh(range(len(counts)), counts, color=colors, alpha=0.8)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel('Sample Count', fontsize=10)
    ax.set_title('Class Balance', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Imbalance ratio
    imbalance_ratio = counts.max() / counts.min() if counts.min() > 0 else float('inf')
    ax.text(0.95, 0.95, f'Imbalance: {imbalance_ratio:.2f}x',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. 振幅分布
    ax = fig.add_subplot(gs[0, 1])
    for i, axis_name in enumerate(['X', 'Y', 'Z']):
        data = X[:, i, :].flatten()
        ax.hist(data, bins=100, alpha=0.5, label=f'{axis_name}-axis', density=True)

    ax.set_xlabel('Acceleration (G)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('Amplitude Distribution', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. 異常値検出
    ax = fig.add_subplot(gs[0, 2])
    outlier_mask = detect_outliers(X, method='iqr')
    outlier_percentage = outlier_mask.sum() / len(X) * 100

    labels_outlier = ['Normal', 'Outlier']
    sizes = [len(X) - outlier_mask.sum(), outlier_mask.sum()]
    colors_pie = ['#66b3ff', '#ff6b6b']
    explode = (0, 0.1)

    ax.pie(sizes, explode=explode, labels=labels_outlier, colors=colors_pie,
           autopct='%1.1f%%', shadow=True, startangle=90)
    ax.set_title(f'Outlier Detection\n({outlier_percentage:.2f}% outliers)',
                 fontsize=11, fontweight='bold')

    # 4. 信号強度の箱ひげ図
    ax = fig.add_subplot(gs[1, 0])
    box_data = [X[:, i, :].flatten() for i in range(3)]
    bp = ax.boxplot(box_data, labels=['X', 'Y', 'Z'], patch_artist=True)

    colors_box = ['#ff9999', '#66b3ff', '#99ff99']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)

    ax.set_ylabel('Acceleration (G)', fontsize=10)
    ax.set_title('Signal Magnitude by Axis', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 5. 周波数スペクトル
    ax = fig.add_subplot(gs[1, 1])
    freqs, power_spectrum = analyze_frequency_spectrum(X)

    ax.semilogy(freqs, power_spectrum, linewidth=1.5, color='steelblue')
    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('Power', fontsize=10)
    ax.set_title('Average Power Spectrum', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 25])  # 0-25 Hz に制限（人間の動作の主要帯域）

    # 6. SNR
    ax = fig.add_subplot(gs[1, 2])
    snr = calculate_snr(X)

    ax.text(0.5, 0.5, f'SNR: {snr:.2f} dB',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=20, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.axis('off')
    ax.set_title('Signal-to-Noise Ratio', fontsize=11, fontweight='bold')

    # 7. 時系列サンプル（クラス別）
    ax = fig.add_subplot(gs[2, :])
    unique_labels_sample = unique_labels[:5]  # 最大5クラス

    for i, label in enumerate(unique_labels_sample):
        indices = np.where(y == label)[0]
        if len(indices) > 0:
            sample_idx = indices[0]
            time_series = X[sample_idx, 0, :]  # X軸のみ
            label_name = label_dict.get(int(label), f"Class {label}")
            ax.plot(time_series, label=label_name, alpha=0.8, linewidth=1.5)

    ax.set_xlabel('Time Steps', fontsize=10)
    ax.set_ylabel('Acceleration (G) - X-axis', fontsize=10)
    ax.set_title('Sample Time Series by Class', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_quality_metrics(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    location: str,
    label_dict: Dict[int, str]
):
    """
    データ品質メトリクスをターミナルに出力

    Args:
        X: センサーデータ (N, 3, T)
        y: ラベル (N,)
        dataset_name: データセット名
        location: 身体部位
        label_dict: ラベル辞書
    """
    print_section_header(f"Data Quality Metrics: {dataset_name.upper()} - {location}")

    # 基本統計
    print_subsection_header("Basic Statistics")
    print(f"  Samples:      {len(X)}")
    print(f"  Time steps:   {X.shape[2]}")
    print(f"  Classes:      {len(np.unique(y))}")
    print()

    # クラスバランス
    unique_labels, counts = np.unique(y, return_counts=True)
    imbalance_ratio = counts.max() / counts.min() if counts.min() > 0 else float('inf')

    print_subsection_header("Class Balance")
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")
    print(f"  Most common:     {counts.max()} samples")
    print(f"  Least common:    {counts.min()} samples")
    print()

    # 信号品質
    print_subsection_header("Signal Quality")
    snr = calculate_snr(X)
    print(f"  SNR:             {snr:.2f} dB")

    for i, axis_name in enumerate(['X', 'Y', 'Z']):
        data = X[:, i, :].flatten()
        print(f"  {axis_name}-axis mean:     {data.mean():.4f} G")
        print(f"  {axis_name}-axis std:      {data.std():.4f} G")
        print(f"  {axis_name}-axis range:    [{data.min():.4f}, {data.max():.4f}] G")

    print()

    # 異常値
    print_subsection_header("Outlier Detection")
    outlier_mask = detect_outliers(X, method='iqr')
    outlier_percentage = outlier_mask.sum() / len(X) * 100
    print(f"  Outliers (IQR): {outlier_mask.sum()} samples ({outlier_percentage:.2f}%)")
    print()


def compare_datasets_quality(all_metrics: List[Dict], output_dir: Path):
    """
    複数データセットの品質比較

    Args:
        all_metrics: [{'name': ..., 'snr': ..., 'imbalance': ..., ...}, ...]
        output_dir: 出力ディレクトリ
    """
    print_section_header("Cross-Dataset Quality Comparison")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Data Quality Comparison Across Datasets', fontsize=16, fontweight='bold')

    names = [m['name'] for m in all_metrics]
    colors = plt.cm.tab20(np.linspace(0, 1, len(names)))

    # 1. SNR比較
    ax = axes[0, 0]
    snrs = [m['snr'] for m in all_metrics]
    bars = ax.barh(range(len(names)), snrs, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n[:25] for n in names], fontsize=8)
    ax.set_xlabel('SNR (dB)', fontsize=10)
    ax.set_title('Signal-to-Noise Ratio', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # 2. クラスバランス
    ax = axes[0, 1]
    imbalances = [m['imbalance_ratio'] for m in all_metrics]
    bars = ax.barh(range(len(names)), imbalances, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n[:25] for n in names], fontsize=8)
    ax.set_xlabel('Imbalance Ratio', fontsize=10)
    ax.set_title('Class Imbalance', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # 3. サンプル数
    ax = axes[1, 0]
    sample_counts = [m['n_samples'] for m in all_metrics]
    bars = ax.barh(range(len(names)), sample_counts, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n[:25] for n in names], fontsize=8)
    ax.set_xlabel('Number of Samples', fontsize=10)
    ax.set_title('Dataset Size', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # 4. 異常値割合
    ax = axes[1, 1]
    outlier_percentages = [m['outlier_percentage'] for m in all_metrics]
    bars = ax.barh(range(len(names)), outlier_percentages, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n[:25] for n in names], fontsize=8)
    ax.set_xlabel('Outlier Percentage (%)', fontsize=10)
    ax.set_title('Data Outliers', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    output_path = output_dir / "quality_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze data quality')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--location', type=str, help='Body location')
    parser.add_argument('--datasets', nargs='+', help='Specific datasets to analyze')
    parser.add_argument('--all', action='store_true', help='Analyze all datasets')
    parser.add_argument('--compare', action='store_true', help='Generate comparison plots')

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = get_project_root() / "analysis" / "figures" / "data_quality"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all or args.compare:
        # 全データセットの分析
        dataset_location_pairs = find_dataset_location_pairs(dataset_filter=args.datasets)
        all_metrics = []

        for dataset_name, location in dataset_location_pairs:
            try:
                X, y, _ = load_sensor_data(dataset_name, location)
                label_dict = get_label_dict(dataset_name)

                # メトリクス計算
                snr = calculate_snr(X)
                outlier_mask = detect_outliers(X)
                outlier_percentage = outlier_mask.sum() / len(X) * 100
                unique_labels, counts = np.unique(y, return_counts=True)
                imbalance_ratio = counts.max() / counts.min() if counts.min() > 0 else float('inf')

                all_metrics.append({
                    'name': f"{dataset_name}/{location}",
                    'snr': snr,
                    'imbalance_ratio': imbalance_ratio,
                    'n_samples': len(X),
                    'outlier_percentage': outlier_percentage
                })

                if args.all:
                    # 個別レポート
                    print_quality_metrics(X, y, dataset_name, location, label_dict)
                    output_path = output_dir / f"{dataset_name}_{location}_quality.png"
                    plot_data_quality_report(X, y, dataset_name, location, label_dict, output_path)

            except Exception as e:
                print(f"Error processing {dataset_name}/{location}: {e}")

        if args.compare and all_metrics:
            compare_datasets_quality(all_metrics, output_dir)

    elif args.dataset and args.location:
        # 単一データセットの分析
        X, y, _ = load_sensor_data(args.dataset, args.location)
        label_dict = get_label_dict(args.dataset)

        print_quality_metrics(X, y, args.dataset, args.location, label_dict)
        output_path = output_dir / f"{args.dataset}_{args.location}_quality.png"
        plot_data_quality_report(X, y, args.dataset, args.location, label_dict, output_path)

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python analysis/data_quality.py --dataset dsads --location Torso")
        print("  python analysis/data_quality.py --all --compare")
        print("  python analysis/data_quality.py --datasets dsads mhealth --compare")

    print_section_header("Data Quality Analysis Complete")
    print(f"Output directory: {output_dir}")


def main_with_args(args):
    """引数オブジェクトを受け取って実行（analyze.pyから呼ばれる用）"""
    # 出力ディレクトリ作成
    output_dir = get_project_root() / "analysis" / "figures" / "data_quality"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all or args.compare:
        # 全データセットの分析
        dataset_location_pairs = find_dataset_location_pairs(dataset_filter=args.datasets)
        all_metrics = []

        for dataset_name, location in dataset_location_pairs:
            try:
                X, y, _ = load_sensor_data(dataset_name, location)
                label_dict = get_label_dict(dataset_name)

                # メトリクス計算
                snr = calculate_snr(X)
                outlier_mask = detect_outliers(X)
                outlier_percentage = outlier_mask.sum() / len(X) * 100
                unique_labels, counts = np.unique(y, return_counts=True)
                imbalance_ratio = counts.max() / counts.min() if counts.min() > 0 else float('inf')

                all_metrics.append({
                    'name': f"{dataset_name}/{location}",
                    'snr': snr,
                    'imbalance_ratio': imbalance_ratio,
                    'n_samples': len(X),
                    'outlier_percentage': outlier_percentage
                })

                if args.all:
                    # 個別レポート
                    print_quality_metrics(X, y, dataset_name, location, label_dict)
                    output_path = output_dir / f"{dataset_name}_{location}_quality.png"
                    plot_data_quality_report(X, y, dataset_name, location, label_dict, output_path)

            except Exception as e:
                print(f"Error processing {dataset_name}/{location}: {e}")

        if args.compare and all_metrics:
            compare_datasets_quality(all_metrics, output_dir)

    elif args.dataset and args.location:
        # 単一データセットの分析
        X, y, _ = load_sensor_data(args.dataset, args.location)
        label_dict = get_label_dict(args.dataset)

        print_quality_metrics(X, y, args.dataset, args.location, label_dict)
        output_path = output_dir / f"{args.dataset}_{args.location}_quality.png"
        plot_data_quality_report(X, y, args.dataset, args.location, label_dict, output_path)

    print_section_header("Data Quality Analysis Complete")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
