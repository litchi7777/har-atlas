"""
特徴量の詳細分析

このスクリプトは、エンコーダーが学習した特徴表現を詳細に分析します：
- 特徴量の活性化パターン
- 特徴量の重要度分析
- レイヤー別の特徴分布
- クラス別・データセット別の特徴統計

使用方法:
    # 事前学習モデルの特徴分析
    python analysis/feature_analysis.py \\
      --model experiments/pretrain/run_*/exp_0/models/best_model.pth \\
      --datasets dsads mhealth

    # 複数モデルの比較
    python analysis/feature_analysis.py \\
      --models experiments/pretrain/run_*/exp_0/models/best_model.pth \\
               experiments/pretrain/run_*/exp_1/models/best_model.pth \\
      --compare

出力:
    analysis/figures/features/ 以下に可視化結果を保存
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from tqdm import tqdm

# 共通ユーティリティをインポート
from utils import (
    get_project_root,
    find_dataset_location_pairs,
    load_sensor_data,
    load_pretrained_model,
    extract_features,
    get_label_dict,
    categorize_body_part,
    get_body_part_color,
    print_section_header,
    print_subsection_header
)

# スタイル設定
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.0)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_feature_activation_distribution(
    features: np.ndarray,
    output_path: Path,
    title: str = "Feature Activation Distribution"
):
    """
    特徴量の活性化分布をプロット

    Args:
        features: 特徴量 (N, D)
        output_path: 出力パス
        title: タイトル
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 1. 各特徴次元の平均活性化
    ax = axes[0, 0]
    mean_activations = features.mean(axis=0)
    ax.bar(range(len(mean_activations)), mean_activations, alpha=0.7, color='steelblue')
    ax.set_xlabel('Feature Dimension', fontsize=11)
    ax.set_ylabel('Mean Activation', fontsize=11)
    ax.set_title('Mean Activation per Feature', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # 2. 各特徴次元の標準偏差
    ax = axes[0, 1]
    std_activations = features.std(axis=0)
    ax.bar(range(len(std_activations)), std_activations, alpha=0.7, color='coral')
    ax.set_xlabel('Feature Dimension', fontsize=11)
    ax.set_ylabel('Standard Deviation', fontsize=11)
    ax.set_title('Variability per Feature', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # 3. 特徴量のヒストグラム（全次元統合）
    ax = axes[1, 0]
    ax.hist(features.flatten(), bins=100, alpha=0.7, color='seagreen', edgecolor='black')
    ax.set_xlabel('Activation Value', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Overall Activation Distribution', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # 4. 特徴相関行列（サンプリング）
    ax = axes[1, 1]
    # 相関計算はコストが高いので、特徴次元をサンプリング
    n_features = min(50, features.shape[1])
    sampled_indices = np.linspace(0, features.shape[1]-1, n_features, dtype=int)
    sampled_features = features[:, sampled_indices]

    corr = np.corrcoef(sampled_features.T)
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_xlabel('Feature Index', fontsize=11)
    ax.set_ylabel('Feature Index', fontsize=11)
    ax.set_title(f'Feature Correlation (sampled {n_features})', fontsize=12)
    plt.colorbar(im, ax=ax, label='Correlation')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_feature_importance(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str = "Feature Importance",
    top_k: int = 30
):
    """
    特徴量の重要度を分析（クラス分離能力に基づく）

    Args:
        features: 特徴量 (N, D)
        labels: ラベル (N,)
        output_path: 出力パス
        title: タイトル
        top_k: 表示する上位k個
    """
    # Fisher判別比を計算（クラス間分散 / クラス内分散）
    unique_labels = np.unique(labels)
    n_features = features.shape[1]

    fisher_scores = []

    for i in range(n_features):
        feature_values = features[:, i]

        # クラス間分散
        overall_mean = feature_values.mean()
        between_var = 0
        for label in unique_labels:
            label_mask = labels == label
            class_mean = feature_values[label_mask].mean()
            class_count = label_mask.sum()
            between_var += class_count * (class_mean - overall_mean) ** 2

        # クラス内分散
        within_var = 0
        for label in unique_labels:
            label_mask = labels == label
            class_values = feature_values[label_mask]
            within_var += ((class_values - class_values.mean()) ** 2).sum()

        # Fisher判別比
        if within_var > 0:
            fisher_score = between_var / within_var
        else:
            fisher_score = 0

        fisher_scores.append(fisher_score)

    fisher_scores = np.array(fisher_scores)

    # 上位k個を抽出
    top_indices = np.argsort(fisher_scores)[::-1][:top_k]
    top_scores = fisher_scores[top_indices]

    # プロット
    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.25)))

    colors = plt.cm.viridis(np.linspace(0, 1, len(top_indices)))
    ax.barh(range(len(top_indices)), top_scores, color=colors, alpha=0.8)

    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels([f"Feature {i}" for i in top_indices], fontsize=9)
    ax.set_xlabel('Fisher Discriminant Ratio', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    return fisher_scores, top_indices


def plot_class_feature_statistics(
    features: np.ndarray,
    labels: np.ndarray,
    label_names: Dict[int, str],
    output_path: Path,
    title: str = "Class-wise Feature Statistics"
):
    """
    クラス別の特徴統計をプロット

    Args:
        features: 特徴量 (N, D)
        labels: ラベル (N,)
        label_names: ラベル名辞書
        output_path: 出力パス
        title: タイトル
    """
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, n_classes * 0.4)))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 1. クラス別の平均特徴ノルム
    ax = axes[0]
    class_norms = []
    class_names_list = []

    for label in unique_labels:
        label_mask = labels == label
        class_features = features[label_mask]
        # L2ノルム
        norms = np.linalg.norm(class_features, axis=1)
        class_norms.append(norms.mean())
        class_names_list.append(label_names.get(int(label), f"Class {label}")[:25])

    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    bars = ax.barh(range(n_classes), class_norms, color=colors, alpha=0.8)

    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(class_names_list, fontsize=9)
    ax.set_xlabel('Mean L2 Norm', fontsize=11)
    ax.set_title('Average Feature Magnitude by Class', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')

    # 2. クラス別の特徴分散
    ax = axes[1]
    class_vars = []

    for label in unique_labels:
        label_mask = labels == label
        class_features = features[label_mask]
        # 各次元の分散の平均
        var = class_features.var(axis=0).mean()
        class_vars.append(var)

    bars = ax.barh(range(n_classes), class_vars, color=colors, alpha=0.8)

    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(class_names_list, fontsize=9)
    ax.set_xlabel('Mean Variance', fontsize=11)
    ax.set_title('Average Feature Variance by Class', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_dataset_feature_comparison(
    all_features: List[np.ndarray],
    dataset_names: List[str],
    output_path: Path,
    title: str = "Dataset Feature Comparison"
):
    """
    データセット間の特徴比較

    Args:
        all_features: [features1, features2, ...]
        dataset_names: ['dataset1', 'dataset2', ...]
        output_path: 出力パス
        title: タイトル
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 1. データセット別の平均特徴ノルム
    ax = axes[0]
    mean_norms = []

    for features in all_features:
        norms = np.linalg.norm(features, axis=1)
        mean_norms.append(norms.mean())

    colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_names)))
    bars = ax.bar(range(len(dataset_names)), mean_norms, color=colors, alpha=0.8)

    ax.set_xticks(range(len(dataset_names)))
    ax.set_xticklabels(dataset_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Mean L2 Norm', fontsize=11)
    ax.set_title('Average Feature Magnitude by Dataset', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # 2. データセット別の特徴分散
    ax = axes[1]
    mean_vars = []

    for features in all_features:
        var = features.var(axis=0).mean()
        mean_vars.append(var)

    bars = ax.bar(range(len(dataset_names)), mean_vars, color=colors, alpha=0.8)

    ax.set_xticks(range(len(dataset_names)))
    ax.set_xticklabels(dataset_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Mean Variance', fontsize=11)
    ax.set_title('Average Feature Variance by Dataset', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def analyze_model_features(
    model_path: str,
    dataset_location_pairs: List[Tuple[str, str]],
    output_dir: Path,
    device: str = 'cuda',
    max_samples: int = 1000
):
    """
    モデルの特徴量を分析

    Args:
        model_path: モデルファイルのパス
        dataset_location_pairs: [(dataset, location), ...]
        output_dir: 出力ディレクトリ
        device: デバイス
        max_samples: 各データセット・部位から取得する最大サンプル数
    """
    model_name = Path(model_path).parent.parent.name
    print_section_header(f"Feature Analysis: {model_name}")

    # モデル読み込み
    encoder = load_pretrained_model(model_path, device=device)

    # データ収集と特徴抽出
    all_features = []
    all_labels = []
    all_dataset_names = []

    dataset_features_dict = {}

    for dataset_name, location in tqdm(dataset_location_pairs, desc="Extracting features"):
        try:
            X, y, _ = load_sensor_data(
                dataset_name,
                location,
                max_samples_per_class=max_samples // len(np.unique(y)) if len(y) > 0 else max_samples
            )

            # 特徴抽出
            features = extract_features(encoder, X, device=device, show_progress=False)

            all_features.append(features)
            all_labels.extend(y.tolist())
            all_dataset_names.extend([dataset_name] * len(features))

            # データセット別に保存
            key = f"{dataset_name}/{location}"
            dataset_features_dict[key] = {
                'features': features,
                'labels': y
            }

        except Exception as e:
            print(f"Error processing {dataset_name}/{location}: {e}")

    # 全特徴を統合
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.array(all_labels)

    print_subsection_header("Feature Statistics")
    print(f"Total samples: {len(all_features)}")
    print(f"Feature dimension: {all_features.shape[1]}")
    print(f"Datasets: {len(set(all_dataset_names))}")
    print()

    # 1. 特徴活性化分布
    output_path = output_dir / f"{model_name}_activation_distribution.png"
    plot_feature_activation_distribution(
        all_features,
        output_path,
        title=f"Feature Activation Distribution - {model_name}"
    )

    # 2. 特徴重要度
    output_path = output_dir / f"{model_name}_feature_importance.png"
    fisher_scores, top_indices = plot_feature_importance(
        all_features,
        all_labels,
        output_path,
        title=f"Feature Importance - {model_name}"
    )

    # 上位特徴を出力
    print_subsection_header("Top 10 Important Features")
    for rank, idx in enumerate(top_indices[:10], 1):
        print(f"  {rank:2d}. Feature {idx:3d}: Fisher Score = {fisher_scores[idx]:.4f}")
    print()

    # 3. データセット別比較
    if len(dataset_features_dict) > 1:
        # 各データセットから同数サンプリング
        min_samples = min(len(data['features']) for data in dataset_features_dict.values())
        sampled_features = []
        sampled_names = []

        for key, data in list(dataset_features_dict.items())[:10]:  # 最大10データセット
            features = data['features']
            if len(features) > min_samples:
                indices = np.random.choice(len(features), min_samples, replace=False)
                features = features[indices]
            sampled_features.append(features)
            sampled_names.append(key)

        output_path = output_dir / f"{model_name}_dataset_comparison.png"
        plot_dataset_feature_comparison(
            sampled_features,
            sampled_names,
            output_path,
            title=f"Dataset Feature Comparison - {model_name}"
        )

    print()


def main():
    parser = argparse.ArgumentParser(description='Analyze learned features')
    parser.add_argument('--model', type=str, help='Path to pretrained model')
    parser.add_argument('--models', nargs='+', help='Paths to multiple models for comparison')
    parser.add_argument('--datasets', nargs='+', help='Specific datasets to analyze')
    parser.add_argument('--locations', nargs='+', help='Specific body locations to analyze')
    parser.add_argument('--max-samples', type=int, default=1000,
                        help='Max samples to use for analysis')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--compare', action='store_true', help='Generate comparison across models')

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = get_project_root() / "analysis" / "figures" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    # モデルのリスト
    if args.models:
        model_paths = args.models
    elif args.model:
        model_paths = [args.model]
    else:
        parser.error("Either --model or --models must be specified")

    # データセット・部位のペアを検出
    dataset_location_pairs = find_dataset_location_pairs(
        dataset_filter=args.datasets,
        location_filter=args.locations
    )

    print(f"Found {len(dataset_location_pairs)} dataset-location pairs")

    # 各モデルを分析
    for model_path in model_paths:
        if not Path(model_path).exists():
            print(f"Warning: Model not found: {model_path}")
            continue

        analyze_model_features(
            model_path,
            dataset_location_pairs,
            output_dir,
            device=args.device,
            max_samples=args.max_samples
        )

    print_section_header("Feature Analysis Complete")
    print(f"Output directory: {output_dir}")


def main_with_args(args):
    """引数オブジェクトを受け取って実行（analyze.pyから呼ばれる用）"""
    # 出力ディレクトリ作成
    output_dir = get_project_root() / "analysis" / "figures" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    # モデルのリスト
    if args.models:
        model_paths = args.models
    elif args.model:
        model_paths = [args.model]
    else:
        print("Error: Either --model or --models must be specified")
        return

    # データセット・部位のペアを検出
    dataset_location_pairs = find_dataset_location_pairs(
        dataset_filter=args.datasets,
        location_filter=args.locations
    )

    print(f"Found {len(dataset_location_pairs)} dataset-location pairs")

    # 各モデルを分析
    for model_path in model_paths:
        if not Path(model_path).exists():
            print(f"Warning: Model not found: {model_path}")
            continue

        analyze_model_features(
            model_path,
            dataset_location_pairs,
            output_dir,
            device=args.device,
            max_samples=args.max_samples
        )

    print_section_header("Feature Analysis Complete")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
