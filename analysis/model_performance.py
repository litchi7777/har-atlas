"""
モデル性能の詳細分析

このスクリプトは、学習済みモデルの性能を多角的に分析します：
- 学習曲線（損失・精度の推移）
- 混同行列とクラス別メトリクス
- データセット別・身体部位別の性能比較
- エポック別の性能変化

使用方法:
    # 単一実験の分析
    python analysis/model_performance.py \\
      --experiment experiments/finetune/run_20251112_*/exp_0

    # 複数実験の比較
    python analysis/model_performance.py \\
      --experiments experiments/finetune/run_*/exp_0 \\
                   experiments/finetune/run_*/exp_1 \\
      --compare

    # 特定のデータセットのみ分析
    python analysis/model_performance.py \\
      --experiment experiments/finetune/run_*/exp_0 \\
      --dataset dsads

出力:
    analysis/figures/performance/ 以下に可視化結果を保存
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import glob
from typing import Dict, List, Optional, Tuple

# 共通ユーティリティをインポート
from utils import (
    get_project_root,
    get_experiment_root,
    load_experiment_config,
    load_experiment_results,
    print_section_header,
    print_subsection_header,
    format_metric
)

# スタイル設定
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_learning_curves(history: Dict, output_path: Path, title: str = "Learning Curves"):
    """
    学習曲線をプロット

    Args:
        history: 学習履歴 {'train_loss': [...], 'val_loss': [...], 'train_acc': [...], 'val_acc': [...]}
        output_path: 出力パス
        title: タイトル
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    epochs = range(1, len(history.get('train_loss', [])) + 1)

    # Loss
    ax = axes[0]
    if 'train_loss' in history:
        ax.plot(epochs, history['train_loss'], 'o-', label='Train Loss', linewidth=2, markersize=4)
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], 's-', label='Val Loss', linewidth=2, markersize=4)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Curves', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    if 'train_acc' in history:
        ax.plot(epochs, history['train_acc'], 'o-', label='Train Acc', linewidth=2, markersize=4)
    if 'val_acc' in history:
        ax.plot(epochs, history['val_acc'], 's-', label='Val Acc', linewidth=2, markersize=4)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Curves', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "Confusion Matrix",
    normalize: bool = True
):
    """
    混同行列をプロット

    Args:
        cm: 混同行列 (n_classes, n_classes)
        class_names: クラス名のリスト
        output_path: 出力パス
        title: タイトル
        normalize: 正規化するか
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(12, 10))

    # ヒートマップ
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_class_metrics(
    class_metrics: Dict[str, List[float]],
    class_names: List[str],
    output_path: Path,
    title: str = "Per-Class Metrics"
):
    """
    クラス別メトリクスをプロット

    Args:
        class_metrics: {'precision': [...], 'recall': [...], 'f1': [...]}
        class_names: クラス名のリスト
        output_path: 出力パス
        title: タイトル
    """
    fig, ax = plt.subplots(figsize=(14, max(6, len(class_names) * 0.3)))

    x = np.arange(len(class_names))
    width = 0.25

    metrics = ['precision', 'recall', 'f1']
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        if metric in class_metrics:
            values = class_metrics[metric]
            ax.barh(x + i * width, values, width, label=metric.capitalize(), color=color, alpha=0.8)

    ax.set_yticks(x + width)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_experiment_comparison(
    experiments: List[Dict],
    output_path: Path,
    metric: str = "accuracy"
):
    """
    複数実験の性能比較

    Args:
        experiments: [{'name': ..., 'metrics': {...}}, ...]
        output_path: 出力パス
        metric: 比較するメトリクス
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    names = [exp['name'] for exp in experiments]
    values = [exp['metrics'].get(metric, 0) for exp in experiments]

    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    bars = ax.barh(range(len(names)), values, color=colors, alpha=0.8)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel(metric.capitalize(), fontsize=12)
    ax.set_title(f'Experiment Comparison - {metric.capitalize()}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # 値をバーに表示
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(
            value + 0.01,
            i,
            format_metric(value, "accuracy" if "acc" in metric else "default"),
            va='center',
            fontsize=10
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def analyze_experiment(experiment_path: Path, output_dir: Path):
    """
    単一実験の詳細分析

    Args:
        experiment_path: 実験ディレクトリのパス
        output_dir: 出力ディレクトリ
    """
    print_section_header(f"Analyzing Experiment: {experiment_path.name}")

    # 実験結果の読み込み
    results = load_experiment_results(experiment_path)

    if not results:
        print("No results found for this experiment.")
        return

    # 学習曲線
    if 'history' in results:
        history = results['history']
        output_path = output_dir / f"{experiment_path.name}_learning_curves.png"
        plot_learning_curves(history, output_path, title=f"Learning Curves - {experiment_path.name}")

    # メトリクスサマリーを表示
    if 'summary' in results:
        summary = results['summary']
        print_subsection_header("Metrics Summary")

        for key, value in summary.items():
            if isinstance(value, (int, float)):
                print(f"  {key:30s}: {format_metric(value)}")

    # 混同行列（もし保存されていれば）
    cm_path = experiment_path / "confusion_matrix.npy"
    if cm_path.exists():
        cm = np.load(cm_path)
        class_names_path = experiment_path / "class_names.json"

        if class_names_path.exists():
            with open(class_names_path, 'r') as f:
                class_names = json.load(f)
        else:
            class_names = [f"Class {i}" for i in range(len(cm))]

        output_path = output_dir / f"{experiment_path.name}_confusion_matrix.png"
        plot_confusion_matrix(cm, class_names, output_path, title=f"Confusion Matrix - {experiment_path.name}")

    # クラス別メトリクス（もし保存されていれば）
    class_metrics_path = experiment_path / "class_metrics.json"
    if class_metrics_path.exists():
        with open(class_metrics_path, 'r') as f:
            class_metrics = json.load(f)

        class_names_path = experiment_path / "class_names.json"
        if class_names_path.exists():
            with open(class_names_path, 'r') as f:
                class_names = json.load(f)
        else:
            class_names = [f"Class {i}" for i in range(len(class_metrics.get('precision', [])))]

        output_path = output_dir / f"{experiment_path.name}_class_metrics.png"
        plot_class_metrics(class_metrics, class_names, output_path, title=f"Class Metrics - {experiment_path.name}")

    print()


def compare_experiments(experiment_paths: List[Path], output_dir: Path):
    """
    複数実験の比較

    Args:
        experiment_paths: 実験ディレクトリのパスのリスト
        output_dir: 出力ディレクトリ
    """
    print_section_header("Comparing Experiments")

    experiments = []

    for exp_path in experiment_paths:
        results = load_experiment_results(exp_path)

        if 'summary' in results:
            experiments.append({
                'name': exp_path.name,
                'path': exp_path,
                'metrics': results['summary']
            })

    if not experiments:
        print("No valid experiments found for comparison.")
        return

    # メトリクス比較表を出力
    print_subsection_header("Metrics Comparison Table")

    # 全実験に共通するメトリクスを抽出
    all_metrics = set()
    for exp in experiments:
        all_metrics.update(exp['metrics'].keys())

    # テーブルヘッダー
    print(f"{'Experiment':<30s}", end="")
    for metric in sorted(all_metrics):
        print(f"{metric:>15s}", end="")
    print()
    print("-" * (30 + 15 * len(all_metrics)))

    # データ行
    for exp in experiments:
        print(f"{exp['name']:<30s}", end="")
        for metric in sorted(all_metrics):
            value = exp['metrics'].get(metric, float('nan'))
            if isinstance(value, (int, float)) and not np.isnan(value):
                print(f"{format_metric(value):>15s}", end="")
            else:
                print(f"{'N/A':>15s}", end="")
        print()

    print()

    # 主要メトリクスの比較プロット
    main_metrics = ['test_accuracy', 'val_accuracy', 'test_loss', 'val_loss']

    for metric in main_metrics:
        if all(metric in exp['metrics'] for exp in experiments):
            output_path = output_dir / f"comparison_{metric}.png"
            plot_experiment_comparison(experiments, output_path, metric=metric)


def main():
    parser = argparse.ArgumentParser(description='Analyze model performance')
    parser.add_argument('--experiment', type=str, help='Path to experiment directory')
    parser.add_argument('--experiments', nargs='+', help='Paths to multiple experiment directories for comparison')
    parser.add_argument('--compare', action='store_true', help='Generate comparison plots')
    parser.add_argument('--mode', type=str, default='finetune', choices=['pretrain', 'finetune'],
                        help='Experiment mode')

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = get_project_root() / "analysis" / "figures" / "performance"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.experiments:
        # 複数実験の比較
        experiment_paths = [Path(exp) for exp in args.experiments]

        # 各実験の個別分析
        for exp_path in experiment_paths:
            if exp_path.exists():
                analyze_experiment(exp_path, output_dir)
            else:
                print(f"Warning: Experiment path not found: {exp_path}")

        # 比較分析
        if args.compare and len(experiment_paths) > 1:
            compare_experiments(experiment_paths, output_dir)

    elif args.experiment:
        # 単一実験の分析
        experiment_path = Path(args.experiment)

        if not experiment_path.exists():
            print(f"Error: Experiment path not found: {experiment_path}")
            return

        analyze_experiment(experiment_path, output_dir)

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python analysis/model_performance.py --experiment experiments/finetune/run_*/exp_0")
        print("  python analysis/model_performance.py --experiments exp_0 exp_1 exp_2 --compare")

    print_section_header("Analysis Complete")
    print(f"Output directory: {output_dir}")


def main_with_args(args):
    """引数オブジェクトを受け取って実行（analyze.pyから呼ばれる用）"""
    # 出力ディレクトリ作成
    output_dir = get_project_root() / "analysis" / "figures" / "performance"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.experiments:
        # 複数実験の比較
        experiment_paths = [Path(exp) for exp in args.experiments]

        # 各実験の個別分析
        for exp_path in experiment_paths:
            if exp_path.exists():
                analyze_experiment(exp_path, output_dir)
            else:
                print(f"Warning: Experiment path not found: {exp_path}")

        # 比較分析
        if args.compare and len(experiment_paths) > 1:
            compare_experiments(experiment_paths, output_dir)

    elif args.experiment:
        # 単一実験の分析
        experiment_path = Path(args.experiment)

        if not experiment_path.exists():
            print(f"Error: Experiment path not found: {experiment_path}")
            return

        analyze_experiment(experiment_path, output_dir)

    print_section_header("Analysis Complete")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
