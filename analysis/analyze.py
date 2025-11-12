#!/usr/bin/env python3
"""
HAR Foundation - 統一分析インターフェース

このスクリプトは、すべての分析機能への統一されたエントリーポイントを提供します。

使用方法:
    python analysis/analyze.py <analysis_type> [options]

分析タイプ:
    data          - データセット分布と品質分析
    embeddings    - 特徴空間の可視化（t-SNE/UMAP）
    performance   - モデル性能の詳細分析
    features      - 特徴量の詳細分析
    all           - 全分析を実行

例:
    # データセット分析
    python analysis/analyze.py data --dataset dsads --location Torso

    # 特徴空間の可視化
    python analysis/analyze.py embeddings \\
      --model experiments/pretrain/run_*/exp_0/models/best_model.pth

    # モデル性能分析
    python analysis/analyze.py performance \\
      --experiment experiments/finetune/run_*/exp_0

    # 全分析を実行
    python analysis/analyze.py all \\
      --model experiments/pretrain/run_*/exp_0/models/best_model.pth \\
      --experiment experiments/finetune/run_*/exp_0
"""

import sys
import argparse
from pathlib import Path

# 各分析モジュールをインポート
import dataset_distribution
import data_quality
import visualize_embeddings
import model_performance
import feature_analysis


def run_data_analysis(args):
    """データセット分布と品質分析を実行"""
    print("\n" + "="*80)
    print("📊 DATA ANALYSIS")
    print("="*80)

    # データセット分布分析
    if args.distribution or args.all_data:
        print("\n--- Dataset Distribution Analysis ---")
        dist_args = argparse.Namespace(
            dataset=args.dataset,
            location=args.location,
            all=args.all_data,
            compare=args.compare
        )
        dataset_distribution.main_with_args(dist_args)

    # データ品質分析
    if args.quality or args.all_data:
        print("\n--- Data Quality Analysis ---")
        quality_args = argparse.Namespace(
            dataset=args.dataset,
            location=args.location,
            datasets=args.datasets,
            all=args.all_data,
            compare=args.compare
        )
        data_quality.main_with_args(quality_args)


def run_embeddings_analysis(args):
    """特徴空間の可視化を実行"""
    print("\n" + "="*80)
    print("🗺️  EMBEDDINGS ANALYSIS")
    print("="*80)

    embed_args = argparse.Namespace(
        model=args.model,
        models=args.models,
        method=args.method,
        color_by=args.color_by,
        datasets=args.datasets,
        locations=args.locations,
        max_samples=args.max_samples,
        device=args.device
    )
    visualize_embeddings.main_with_args(embed_args)


def run_performance_analysis(args):
    """モデル性能分析を実行"""
    print("\n" + "="*80)
    print("📈 PERFORMANCE ANALYSIS")
    print("="*80)

    perf_args = argparse.Namespace(
        experiment=args.experiment,
        experiments=args.experiments,
        compare=args.compare,
        mode=args.mode
    )
    model_performance.main_with_args(perf_args)


def run_features_analysis(args):
    """特徴量の詳細分析を実行"""
    print("\n" + "="*80)
    print("🔬 FEATURE ANALYSIS")
    print("="*80)

    feat_args = argparse.Namespace(
        model=args.model,
        models=args.models,
        datasets=args.datasets,
        locations=args.locations,
        max_samples=args.max_samples,
        device=args.device,
        compare=args.compare
    )
    feature_analysis.main_with_args(feat_args)


def run_all_analyses(args):
    """全分析を実行"""
    print("\n" + "="*80)
    print("🚀 RUNNING ALL ANALYSES")
    print("="*80)

    # データ分析
    if args.dataset or args.all_data:
        run_data_analysis(args)

    # 特徴空間の可視化
    if args.model or args.models:
        run_embeddings_analysis(args)

    # モデル性能分析
    if args.experiment or args.experiments:
        run_performance_analysis(args)

    # 特徴量分析
    if args.model or args.models:
        run_features_analysis(args)

    print("\n" + "="*80)
    print("✅ ALL ANALYSES COMPLETED")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='HAR Foundation - 統一分析インターフェース',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # データセット分析
  python analysis/analyze.py data --dataset dsads --location Torso

  # 特徴空間の可視化
  python analysis/analyze.py embeddings \\
    --model experiments/pretrain/run_*/exp_0/models/best_model.pth

  # モデル性能分析
  python analysis/analyze.py performance \\
    --experiment experiments/finetune/run_*/exp_0

  # 特徴量分析
  python analysis/analyze.py features \\
    --model experiments/pretrain/run_*/exp_0/models/best_model.pth

  # 全分析を実行
  python analysis/analyze.py all \\
    --model experiments/pretrain/run_*/exp_0/models/best_model.pth \\
    --experiment experiments/finetune/run_*/exp_0 \\
    --dataset dsads --location Torso
        """
    )

    # サブコマンド
    subparsers = parser.add_subparsers(dest='analysis_type', help='分析タイプ')

    # ===== データ分析 =====
    data_parser = subparsers.add_parser('data', help='データセット分布と品質分析')
    data_parser.add_argument('--dataset', type=str, help='データセット名')
    data_parser.add_argument('--location', type=str, help='身体部位')
    data_parser.add_argument('--datasets', nargs='+', help='複数のデータセット')
    data_parser.add_argument('--all-data', action='store_true', help='全データセットを分析')
    data_parser.add_argument('--compare', action='store_true', help='データセット間の比較')
    data_parser.add_argument('--distribution', action='store_true', default=True,
                             help='分布分析を実行')
    data_parser.add_argument('--quality', action='store_true', default=True,
                             help='品質分析を実行')

    # ===== 特徴空間の可視化 =====
    embed_parser = subparsers.add_parser('embeddings', help='特徴空間の可視化')
    embed_parser.add_argument('--model', type=str, help='モデルパス')
    embed_parser.add_argument('--models', nargs='+', help='複数のモデルパス（比較用）')
    embed_parser.add_argument('--method', type=str, default='umap',
                              choices=['umap', 'tsne'], help='次元削減手法')
    embed_parser.add_argument('--color-by', type=str, default='body_part',
                              choices=['dataset', 'body_part', 'dataset_location'],
                              help='色分け基準')
    embed_parser.add_argument('--datasets', nargs='+', help='対象データセット')
    embed_parser.add_argument('--locations', nargs='+', help='対象身体部位')
    embed_parser.add_argument('--max-samples', type=int, default=500,
                              help='各クラスの最大サンプル数')
    embed_parser.add_argument('--device', type=str, default='cuda',
                              help='デバイス (cuda/cpu)')

    # ===== モデル性能分析 =====
    perf_parser = subparsers.add_parser('performance', help='モデル性能の詳細分析')
    perf_parser.add_argument('--experiment', type=str, help='実験ディレクトリパス')
    perf_parser.add_argument('--experiments', nargs='+', help='複数の実験（比較用）')
    perf_parser.add_argument('--compare', action='store_true', help='実験間の比較')
    perf_parser.add_argument('--mode', type=str, default='finetune',
                             choices=['pretrain', 'finetune'], help='実験モード')

    # ===== 特徴量分析 =====
    feat_parser = subparsers.add_parser('features', help='特徴量の詳細分析')
    feat_parser.add_argument('--model', type=str, help='モデルパス')
    feat_parser.add_argument('--models', nargs='+', help='複数のモデルパス（比較用）')
    feat_parser.add_argument('--datasets', nargs='+', help='対象データセット')
    feat_parser.add_argument('--locations', nargs='+', help='対象身体部位')
    feat_parser.add_argument('--max-samples', type=int, default=1000,
                              help='分析用の最大サンプル数')
    feat_parser.add_argument('--device', type=str, default='cuda',
                              help='デバイス (cuda/cpu)')
    feat_parser.add_argument('--compare', action='store_true', help='モデル間の比較')

    # ===== 全分析 =====
    all_parser = subparsers.add_parser('all', help='全分析を実行')
    all_parser.add_argument('--model', type=str, help='モデルパス')
    all_parser.add_argument('--models', nargs='+', help='複数のモデルパス')
    all_parser.add_argument('--experiment', type=str, help='実験ディレクトリパス')
    all_parser.add_argument('--experiments', nargs='+', help='複数の実験')
    all_parser.add_argument('--dataset', type=str, help='データセット名')
    all_parser.add_argument('--location', type=str, help='身体部位')
    all_parser.add_argument('--datasets', nargs='+', help='対象データセット')
    all_parser.add_argument('--locations', nargs='+', help='対象身体部位')
    all_parser.add_argument('--all-data', action='store_true', help='全データセットを分析')
    all_parser.add_argument('--compare', action='store_true', help='比較分析を有効化')
    all_parser.add_argument('--method', type=str, default='umap',
                            choices=['umap', 'tsne'], help='次元削減手法')
    all_parser.add_argument('--color-by', type=str, default='body_part',
                            choices=['dataset', 'body_part', 'dataset_location'],
                            help='色分け基準')
    all_parser.add_argument('--max-samples', type=int, default=500,
                            help='最大サンプル数')
    all_parser.add_argument('--device', type=str, default='cuda',
                            help='デバイス (cuda/cpu)')
    all_parser.add_argument('--mode', type=str, default='finetune',
                            choices=['pretrain', 'finetune'], help='実験モード')
    all_parser.add_argument('--distribution', action='store_true', default=True)
    all_parser.add_argument('--quality', action='store_true', default=True)

    args = parser.parse_args()

    # サブコマンドが指定されていない場合
    if not args.analysis_type:
        parser.print_help()
        sys.exit(1)

    # 分析タイプに応じて実行
    if args.analysis_type == 'data':
        run_data_analysis(args)
    elif args.analysis_type == 'embeddings':
        run_embeddings_analysis(args)
    elif args.analysis_type == 'performance':
        run_performance_analysis(args)
    elif args.analysis_type == 'features':
        run_features_analysis(args)
    elif args.analysis_type == 'all':
        run_all_analyses(args)


if __name__ == '__main__':
    main()
