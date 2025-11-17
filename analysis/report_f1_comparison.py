#!/usr/bin/env python3
"""
F1スコア比較レポート生成スクリプト

使用方法:
    python analysis/report_f1_comparison.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# プロットスタイル設定
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

# データ読み込み
df = pd.read_csv('experiments/analysis/finetune_all_models/all_results.csv')

# 出力ディレクトリ
output_dir = Path('experiments/analysis/f1_report')
output_dir.mkdir(exist_ok=True, parents=True)

# モデル名のマッピング（短縮版）
model_name_map = {
    'run_20251111_171703_exp_0': 'Pretrain_Arms',
    'run_20251111_171703_exp_1': 'Pretrain_Torso',
    'run_20251111_171703_exp_2': 'Pretrain_Legs',
    'run_20251111_072854_exp_2': 'Pretrain_NHANES',
    'supervised_from_scratch': 'Supervised (Baseline)',
}

df['model_name'] = df['model'].map(model_name_map)

# モデル別統計
model_stats = df.groupby('model_name')['test_f1'].agg(['mean', 'std', 'count']).reset_index()
model_stats = model_stats.sort_values('mean', ascending=False)

# ベースラインF1
baseline_f1 = model_stats[model_stats['model_name'] == 'Supervised (Baseline)']['mean'].values[0]

# 改善率を計算
model_stats['improvement'] = ((model_stats['mean'] - baseline_f1) / baseline_f1 * 100).round(2)

# 1. モデル別F1スコア比較（バープロット + エラーバー）
fig, ax = plt.subplots(figsize=(12, 6))

colors = ['#2ecc71' if 'Pretrain' in name else '#e74c3c' for name in model_stats['model_name']]

bars = ax.bar(
    range(len(model_stats)),
    model_stats['mean'],
    yerr=model_stats['std'],
    capsize=5,
    color=colors,
    alpha=0.8,
    edgecolor='black',
    linewidth=1.5
)

# ベースラインの水平線
ax.axhline(y=baseline_f1, color='red', linestyle='--', linewidth=2, label=f'Baseline: {baseline_f1:.3f}', alpha=0.7)

# 改善率をバーの上に表示
for i, (bar, improvement) in enumerate(zip(bars, model_stats['improvement'])):
    if improvement > 0:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + model_stats.iloc[i]['std'] + 0.01,
            f'+{improvement:.1f}%',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold',
            color='darkgreen'
        )

ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_ylabel('Test F1 Score', fontsize=14, fontweight='bold')
ax.set_title('F1 Score Comparison: Pretrained vs Supervised Baseline', fontsize=16, fontweight='bold')
ax.set_xticks(range(len(model_stats)))
ax.set_xticklabels(model_stats['model_name'], rotation=15, ha='right')
ax.set_ylim(0, 0.8)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'f1_model_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'f1_model_comparison.png'}")

# 2. データセット×デバイス別F1スコア（全モデル比較）
fig, ax = plt.subplots(figsize=(16, 8))

# dataset_deviceの組み合わせを作成
df['dataset_device'] = df['dataset'] + '_' + df['device']

dataset_device_stats = df.groupby(['dataset_device', 'model_name'])['test_f1'].mean().reset_index()
pivot = dataset_device_stats.pivot(index='dataset_device', columns='model_name', values='test_f1')

# 列の順序を指定（Baselineを最後に）
column_order = [col for col in pivot.columns if col != 'Supervised (Baseline)'] + ['Supervised (Baseline)']
pivot = pivot[column_order]

# 平均F1でソート
pivot['mean'] = pivot.mean(axis=1)
pivot = pivot.sort_values('mean', ascending=False).drop('mean', axis=1)

pivot.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', linewidth=0.8)
ax.set_xlabel('Dataset_Device', fontsize=14, fontweight='bold')
ax.set_ylabel('Test F1 Score', fontsize=14, fontweight='bold')
ax.set_title('F1 Score by Dataset×Device and Model', fontsize=16, fontweight='bold')
ax.legend(title='Model', fontsize=10, title_fontsize=11, loc='upper right')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'f1_by_dataset_device.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'f1_by_dataset_device.png'}")

# 3. 改善率のヒートマップ（データセット×デバイス × モデル）
fig, ax = plt.subplots(figsize=(14, 10))

# 各データセット×デバイス・モデルごとにベースラインとの改善率を計算
dataset_device_baseline = df[df['model_name'] == 'Supervised (Baseline)'].groupby('dataset_device')['test_f1'].mean()
improvement_data = []

for dataset_device in df['dataset_device'].unique():
    baseline_val = dataset_device_baseline.loc[dataset_device]
    for model in df['model_name'].unique():
        if model != 'Supervised (Baseline)':
            model_val = df[(df['dataset_device'] == dataset_device) & (df['model_name'] == model)]['test_f1'].mean()
            improvement_pct = ((model_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
            improvement_data.append({
                'dataset_device': dataset_device,
                'model': model,
                'improvement': improvement_pct
            })

improvement_df = pd.DataFrame(improvement_data)
pivot_improvement = improvement_df.pivot(index='dataset_device', columns='model', values='improvement')

# 平均改善率でソート
pivot_improvement['mean'] = pivot_improvement.mean(axis=1)
pivot_improvement = pivot_improvement.sort_values('mean', ascending=False).drop('mean', axis=1)

sns.heatmap(
    pivot_improvement,
    annot=True,
    fmt='.1f',
    cmap='RdYlGn',
    center=0,
    vmin=-20,
    vmax=20,
    cbar_kws={'label': 'Improvement (%)'},
    ax=ax,
    linewidths=0.5,
    linecolor='gray'
)

ax.set_title('F1 Improvement over Supervised Baseline (%) by Dataset×Device', fontsize=16, fontweight='bold')
ax.set_xlabel('Pretrained Model', fontsize=14, fontweight='bold')
ax.set_ylabel('Dataset_Device', fontsize=14, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(output_dir / 'f1_improvement_heatmap.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'f1_improvement_heatmap.png'}")

# 4. Box plot（分布の可視化）
fig, ax = plt.subplots(figsize=(12, 6))

# データを準備（モデル名の順序を制御）
df_sorted = df.copy()
df_sorted['model_name'] = pd.Categorical(
    df_sorted['model_name'],
    categories=model_stats['model_name'].tolist(),
    ordered=True
)
df_sorted = df_sorted.sort_values('model_name')

# Box plot
box_colors = ['lightgreen' if 'Pretrain' in str(name) else 'lightcoral'
              for name in df_sorted['model_name'].unique()]

bp = ax.boxplot(
    [df_sorted[df_sorted['model_name'] == model]['test_f1'].values
     for model in df_sorted['model_name'].unique()],
    labels=df_sorted['model_name'].unique(),
    patch_artist=True,
    showmeans=True,
    meanprops=dict(marker='D', markerfacecolor='red', markersize=8)
)

# 色を設定
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_ylabel('Test F1 Score', fontsize=14, fontweight='bold')
ax.set_title('F1 Score Distribution by Model', fontsize=16, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig(output_dir / 'f1_distribution_boxplot.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'f1_distribution_boxplot.png'}")

# 5. サマリー統計をテキストファイルに出力
with open(output_dir / 'summary.txt', 'w') as f:
    f.write('=' * 80 + '\n')
    f.write('F1 SCORE COMPARISON REPORT\n')
    f.write('=' * 80 + '\n\n')

    f.write('1. OVERALL MODEL RANKING (by mean F1)\n')
    f.write('-' * 80 + '\n')
    for i, row in model_stats.iterrows():
        f.write(f"{i+1}. {row['model_name']:30s}: {row['mean']:.4f} ± {row['std']:.4f}")
        if row['improvement'] > 0:
            f.write(f" (+{row['improvement']:.2f}% vs baseline)")
        f.write('\n')

    f.write('\n2. BASELINE PERFORMANCE\n')
    f.write('-' * 80 + '\n')
    f.write(f"Supervised (Baseline) F1: {baseline_f1:.4f}\n")

    f.write('\n3. PRETRAINED MODEL IMPROVEMENTS\n')
    f.write('-' * 80 + '\n')
    for i, row in model_stats.iterrows():
        if 'Pretrain' in row['model_name']:
            f.write(f"{row['model_name']:30s}: +{row['improvement']:.2f}%\n")

    f.write('\n4. BEST MODEL PER DATASET×DEVICE (F1)\n')
    f.write('-' * 80 + '\n')
    best_per_dataset_device = df.loc[df.groupby('dataset_device')['test_f1'].idxmax()]
    best_per_dataset_device = best_per_dataset_device.sort_values('test_f1', ascending=False)
    for _, row in best_per_dataset_device.iterrows():
        f.write(f"{row['dataset_device']:30s}: {row['model_name']:30s} (F1={row['test_f1']:.4f})\n")

    f.write('\n5. WORST PERFORMING DATASET×DEVICE (F1 < 0.5)\n')
    f.write('-' * 80 + '\n')
    # 各dataset_deviceの平均F1を計算
    avg_f1_per_dataset_device = df.groupby('dataset_device')['test_f1'].mean().sort_values()
    for dataset_device, f1 in avg_f1_per_dataset_device.items():
        if f1 < 0.5:
            f.write(f"{dataset_device:30s}: {f1:.4f} (Average across all models)\n")

    f.write('\n' + '=' * 80 + '\n')

print(f"Saved: {output_dir / 'summary.txt'}")
print(f"\n✓ F1 Score comparison report generated in {output_dir}/")
