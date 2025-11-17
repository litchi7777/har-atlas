#!/usr/bin/env python3
"""
2つのファインチューニング実験の比較可視化スクリプト

使用方法:
    python analysis/visualize_finetune_comparison.py
"""

import json
import yaml
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# プロットスタイル設定
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

def load_experiment_data(run_dir):
    """実験ディレクトリから全てのexp_*の結果を読み込む"""
    run_path = Path(run_dir)
    data = []

    # 全てのexp_*ディレクトリを走査
    for exp_dir in sorted(run_path.glob("exp_*")):
        exp_num = exp_dir.name.replace("exp_", "")

        # config.yamlを読み込む
        config_path = exp_dir / "config.yaml"
        results_path = exp_dir / "results.json"

        if not config_path.exists() or not results_path.exists():
            continue

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        with open(results_path, 'r') as f:
            results = json.load(f)

        # データセットとデバイス情報を抽出
        dataset_location_pairs = config['sensor_data']['dataset_location_pairs']
        if len(dataset_location_pairs) > 0:
            dataset = dataset_location_pairs[0][0]
            device = dataset_location_pairs[0][1]
        else:
            dataset = "unknown"
            device = "unknown"

        # 事前学習モデルのパスから識別
        pretrained_path = config['model']['pretrained_path']
        # Noneの場合は文字列に変換
        if pretrained_path is None:
            pretrained_path = 'None'

        data.append({
            'exp_num': int(exp_num),
            'run_dir': run_path.name,
            'pretrained_path': pretrained_path,
            'dataset': dataset,
            'device': device,
            'test_accuracy': results.get('test_accuracy', 0),
            'test_f1': results.get('test_f1', 0),
            'test_precision': results.get('test_precision', 0),
            'test_recall': results.get('test_recall', 0),
            'best_val_accuracy': results.get('best_val_accuracy', 0),
        })

    return pd.DataFrame(data)

def create_comparison_plots(df1, df2, output_dir):
    """比較グラフを生成"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # 全てのデータを結合
    df_combined = pd.concat([df1, df2], ignore_index=True)

    # 事前学習モデルのパスから詳細なモデル名を作成
    def create_model_label(pretrained_path):
        if pretrained_path is None or str(pretrained_path).lower() == 'none':
            return "supervised_from_scratch"
        parts = pretrained_path.split('/')
        run_name = parts[-4]  # run_YYYYMMDD_HHMMSS
        exp_name = parts[-3]  # exp_X
        return f"{run_name}_{exp_name}"

    df_combined['model'] = df_combined['pretrained_path'].apply(create_model_label)

    # データセット×デバイスの組み合わせを作成
    df_combined['dataset_device'] = df_combined['dataset'] + '_' + df_combined['device']

    # メトリクス
    metrics = ['test_accuracy', 'test_f1', 'test_precision', 'test_recall']
    metric_names = ['Test Accuracy', 'Test F1', 'Test Precision', 'Test Recall']

    # 1. データセット×デバイス別の性能比較（4メトリクス）
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle('Performance Comparison by Dataset & Device', fontsize=16, fontweight='bold')

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]

        # ピボットテーブル作成
        pivot = df_combined.pivot_table(
            values=metric,
            index='dataset_device',
            columns='model',
            aggfunc='first'
        )

        # バープロット
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title(metric_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Dataset_Device', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.legend(title='Model', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        # Y軸を0-1に固定
        ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_path / 'comparison_by_dataset_device.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'comparison_by_dataset_device.png'}")

    # 2. データセット別の平均性能（デバイスをまとめる）
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle('Average Performance by Dataset (across all devices)', fontsize=16, fontweight='bold')

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]

        # データセット別に平均を計算
        dataset_avg = df_combined.groupby(['dataset', 'model'])[metric].mean().reset_index()
        pivot = dataset_avg.pivot(index='dataset', columns='model', values=metric)

        # バープロット
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title(f'Average {metric_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel(f'Average {metric_name}', fontsize=12)
        ax.legend(title='Model', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_path / 'comparison_by_dataset_avg.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'comparison_by_dataset_avg.png'}")

    # 3. デバイス別の平均性能（データセットをまとめる）
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle('Average Performance by Device (across all datasets)', fontsize=16, fontweight='bold')

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]

        # デバイス別に平均を計算
        device_avg = df_combined.groupby(['device', 'model'])[metric].mean().reset_index()
        pivot = device_avg.pivot(index='device', columns='model', values=metric)

        # バープロット
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title(f'Average {metric_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Device', fontsize=12)
        ax.set_ylabel(f'Average {metric_name}', fontsize=12)
        ax.legend(title='Model', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_path / 'comparison_by_device_avg.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'comparison_by_device_avg.png'}")

    # 4. 散布図（Test Accuracy vs Test F1）
    fig, ax = plt.subplots(figsize=(12, 8))

    for model in df_combined['model'].unique():
        model_data = df_combined[df_combined['model'] == model]
        ax.scatter(
            model_data['test_accuracy'],
            model_data['test_f1'],
            label=model,
            s=100,
            alpha=0.6
        )

    ax.set_xlabel('Test Accuracy', fontsize=12)
    ax.set_ylabel('Test F1 Score', fontsize=12)
    ax.set_title('Test Accuracy vs F1 Score Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_path / 'scatter_accuracy_vs_f1.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'scatter_accuracy_vs_f1.png'}")

    # 5. ヒートマップ（Test Accuracy）
    models = sorted(df_combined['model'].unique())
    n_models = len(models)

    # 2行xN列のレイアウト（最大4モデル想定）
    n_cols = 2
    n_rows = (n_models + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8 * n_rows))
    fig.suptitle('Test Accuracy Heatmap by Dataset & Device', fontsize=16, fontweight='bold')

    # axesを1次元配列に変換（1つのモデルの場合に対応）
    if n_models == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(-1)
    else:
        axes = axes.flatten()

    for idx, model in enumerate(models):
        model_data = df_combined[df_combined['model'] == model]
        pivot = model_data.pivot_table(
            values='test_accuracy',
            index='dataset',
            columns='device',
            aggfunc='first'
        )

        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            ax=axes[idx],
            cbar_kws={'label': 'Test Accuracy'}
        )
        axes[idx].set_title(model, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Device', fontsize=12)
        axes[idx].set_ylabel('Dataset', fontsize=12)

    # 未使用のサブプロットを非表示
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path / 'heatmap_accuracy.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'heatmap_accuracy.png'}")

    # 6. サマリー統計テーブル
    summary_stats = df_combined.groupby('model')[metrics].agg(['mean', 'std', 'min', 'max'])
    print("\n=== Summary Statistics ===")
    print(summary_stats.to_string())

    # CSVに保存
    summary_stats.to_csv(output_path / 'summary_statistics.csv')
    print(f"\nSaved: {output_path / 'summary_statistics.csv'}")

    # 全データをCSVに保存
    df_combined.to_csv(output_path / 'all_results.csv', index=False)
    print(f"Saved: {output_path / 'all_results.csv'}")

def main():
    # 実験ディレクトリ（コマンドライン引数または設定で指定可能）
    import argparse

    parser = argparse.ArgumentParser(description="Compare finetune experiments")
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        default=[
            "experiments/finetune/run_20251112_123143_from_20251111_171703",
            "experiments/finetune/run_20251112_135205_from_20251111_072854",
        ],
        help="Experiment directories to compare (space-separated)",
    )
    parser.add_argument(
        "--output",
        default="experiments/analysis/finetune_comparison",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    print("Loading experiment data...")
    all_dfs = []
    for run_dir in args.run_dirs:
        df = load_experiment_data(run_dir)
        print(f"Loaded {len(df)} experiments from {run_dir}")
        all_dfs.append(df)

    # 全データを結合
    df_combined = pd.concat(all_dfs, ignore_index=True)

    print(f"\nTotal experiments: {len(df_combined)}")
    print(f"Unique models: {df_combined['pretrained_path'].nunique()}")

    print("\nCreating comparison plots...")
    # 全データフレームを結合して渡す
    df_all = pd.concat(all_dfs, ignore_index=True)
    # create_comparison_plotsは内部でdf1とdf2を結合するので、ダミーの空DataFrameを渡す
    create_comparison_plots(df_all, pd.DataFrame(), args.output)

    print(f"\n✓ All plots saved to {args.output}/")

if __name__ == "__main__":
    main()