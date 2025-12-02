#!/usr/bin/env python3
"""
HAR Foundation - 統一分析インターフェース

全ての分析タスクを1つのコマンドで実行できる統一エントリーポイント。

使用方法:
    python analysis/main.py <command> [options]

利用可能なコマンド:
    visualize     - 埋め込み可視化（UMAP/t-SNE）
    extract       - 特徴量抽出
    report        - F1スコア比較レポート
    compare       - ファインチューニング比較
    reconstruct   - センサーデータのリコンストラクション可視化

例:
    # 埋め込み可視化
    python analysis/main.py visualize \\
        --model experiments/pretrain/run_*/exp_0/models/checkpoint.pth \\
        --method umap \\
        --color-by body_part

    # 特徴量抽出
    python analysis/main.py extract \\
        --model experiments/pretrain/run_*/exp_0/models/checkpoint.pth \\
        --window-size 60

    # F1スコア比較レポート
    python analysis/main.py report

    # ファインチューニング比較
    python analysis/main.py compare \\
        --runs run_20251112_*

    # リコンストラクション可視化
    python analysis/main.py reconstruct \\
        --model experiments/pretrain/run_*/ssl_tasks=*/models/checkpoint_epoch_99.pth \\
        --num-samples 5

ヘルプ:
    python analysis/main.py <command> --help
"""

import sys
import argparse
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "analysis"))


def create_parser():
    """メインのArgumentParserを作成"""
    parser = argparse.ArgumentParser(
        description='HAR Foundation - 統一分析インターフェース',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
利用可能なコマンド:
  visualize     埋め込み可視化（UMAP/t-SNE）
  extract       特徴量抽出
  report        F1スコア比較レポート
  compare       ファインチューニング比較
  reconstruct   リコンストラクション可視化

各コマンドのヘルプ:
  python analysis/main.py <command> --help
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='実行するコマンド')

    # ========================================
    # visualize - 埋め込み可視化
    # ========================================
    visualize_parser = subparsers.add_parser(
        'visualize',
        help='埋め込み可視化（UMAP/t-SNE）'
    )
    visualize_parser.add_argument(
        '--model', type=str,
        default='experiments/pretrain/run_20251111_171703/exp_0/models/checkpoint_epoch_100.pth',
        help='事前学習済みモデルのパス'
    )
    visualize_parser.add_argument(
        '--models', nargs='+',
        help='複数モデルの比較用パス'
    )
    visualize_parser.add_argument(
        '--method', type=str, default='umap',
        choices=['umap', 'tsne', 'pca'],
        help='次元削減手法'
    )
    visualize_parser.add_argument(
        '--color-by', type=str, default='body_part',
        choices=['dataset', 'body_part', 'dataset_location', 'activity', 'activity_per_dataset'],
        help='色分けの基準'
    )
    visualize_parser.add_argument(
        '--datasets', nargs='+',
        help='特定のデータセットのみ（例: dsads mhealth）'
    )
    visualize_parser.add_argument(
        '--locations', nargs='+',
        help='特定の身体部位のみ（例: Torso Wrist）'
    )
    visualize_parser.add_argument(
        '--max-samples', type=int, default=100,
        help='クラスごとの最大サンプル数'
    )
    visualize_parser.add_argument(
        '--device', type=str, default='cuda',
        choices=['cuda', 'cpu'],
        help='デバイス'
    )
    visualize_parser.add_argument(
        '--output-dir', type=str, default='analysis/figures',
        help='出力ディレクトリ'
    )

    # ========================================
    # extract - 特徴量抽出
    # ========================================
    extract_parser = subparsers.add_parser(
        'extract',
        help='特徴量抽出'
    )
    extract_parser.add_argument(
        '--model', type=str, required=True,
        help='事前学習済みモデルのパス'
    )
    extract_parser.add_argument(
        '--window-size', type=int, default=None,
        help='ウィンドウサイズ（自動検出する場合は指定不要）'
    )
    extract_parser.add_argument(
        '--max-samples', type=int, default=100,
        help='クラスごとの最大サンプル数'
    )
    extract_parser.add_argument(
        '--max-users', type=int, default=20,
        help='大規模データセット用の最大ユーザー数'
    )
    extract_parser.add_argument(
        '--output-dir', type=str, default='analysis/embedding_explorer/data',
        help='出力ディレクトリ'
    )
    extract_parser.add_argument(
        '--device', type=str, default='cuda',
        choices=['cuda', 'cpu'],
        help='デバイス'
    )
    extract_parser.add_argument(
        '--datasets', nargs='+', default=None,
        help='特定のデータセットのみ'
    )
    extract_parser.add_argument(
        '--locations', nargs='+', default=None,
        help='特定の身体部位のみ'
    )
    extract_parser.add_argument(
        '--compute-umap', action='store_true',
        help='UMAP埋め込みも計算して保存'
    )

    # ========================================
    # report - F1スコア比較レポート
    # ========================================
    report_parser = subparsers.add_parser(
        'report',
        help='F1スコア比較レポート'
    )
    report_parser.add_argument(
        '--input-csv', type=str,
        default='experiments/analysis/finetune_all_models/all_results.csv',
        help='入力CSVファイル'
    )
    report_parser.add_argument(
        '--output-dir', type=str,
        default='experiments/analysis/f1_report',
        help='出力ディレクトリ'
    )

    # ========================================
    # compare - ファインチューニング比較
    # ========================================
    compare_parser = subparsers.add_parser(
        'compare',
        help='ファインチューニング比較'
    )
    compare_parser.add_argument(
        '--runs', nargs='+', required=True,
        help='比較する実験run（例: run_20251112_*）'
    )
    compare_parser.add_argument(
        '--output-dir', type=str, default='analysis/figures',
        help='出力ディレクトリ'
    )
    compare_parser.add_argument(
        '--metric', type=str, default='test_f1',
        choices=['test_f1', 'test_acc', 'val_f1', 'val_acc'],
        help='比較するメトリクス'
    )

    # ========================================
    # reconstruct - リコンストラクション可視化
    # ========================================
    reconstruct_parser = subparsers.add_parser(
        'reconstruct',
        help='センサーデータのリコンストラクション可視化'
    )
    reconstruct_parser.add_argument(
        '--model', type=str, required=True,
        help='事前学習済みモデルのパス（masking_*タスクを含むモデル）'
    )
    reconstruct_parser.add_argument(
        '--data-root', type=str,
        default='har-unified-dataset/data/processed',
        help='データルート'
    )
    reconstruct_parser.add_argument(
        '--datasets', nargs='+', default=None,
        help='使用するデータセット'
    )
    reconstruct_parser.add_argument(
        '--num-samples', type=int, default=5,
        help='可視化するサンプル数'
    )
    reconstruct_parser.add_argument(
        '--mask-ratio', type=float, default=0.15,
        help='マスク比率'
    )
    reconstruct_parser.add_argument(
        '--device', type=str, default='cuda',
        choices=['cuda', 'cpu'],
        help='デバイス'
    )
    reconstruct_parser.add_argument(
        '--output-dir', type=str, default='analysis/figures',
        help='出力ディレクトリ'
    )
    reconstruct_parser.add_argument(
        '--no-show', action='store_true',
        help='プロットを表示しない'
    )

    return parser


def cmd_visualize(args):
    """埋め込み可視化コマンド"""
    print("="*80)
    print("埋め込み可視化を実行します")
    print("="*80)

    # scripts/visualize_embeddings.pyのmain関数を呼び出す
    from scripts import visualize_embeddings

    # argsをsys.argvに変換してスクリプトを実行
    sys.argv = ['visualize_embeddings.py']

    if args.model:
        sys.argv.extend(['--model', args.model])
    if args.models:
        sys.argv.append('--models')
        sys.argv.extend(args.models)
    if args.method:
        sys.argv.extend(['--method', args.method])
    if args.color_by:
        sys.argv.extend(['--color-by', args.color_by])
    if args.datasets:
        sys.argv.append('--datasets')
        sys.argv.extend(args.datasets)
    if args.locations:
        sys.argv.append('--locations')
        sys.argv.extend(args.locations)
    if args.max_samples:
        sys.argv.extend(['--max-samples', str(args.max_samples)])
    if args.device:
        sys.argv.extend(['--device', args.device])
    if args.output_dir:
        sys.argv.extend(['--output-dir', args.output_dir])

    visualize_embeddings.main()


def cmd_extract(args):
    """特徴量抽出コマンド"""
    print("="*80)
    print("特徴量抽出を実行します")
    print("="*80)

    from embedding_explorer import extract_features

    sys.argv = ['extract_features.py']
    sys.argv.extend(['--model', args.model])

    if args.window_size:
        sys.argv.extend(['--window-size', str(args.window_size)])
    if args.max_samples:
        sys.argv.extend(['--max-samples', str(args.max_samples)])
    if args.max_users:
        sys.argv.extend(['--max-users', str(args.max_users)])
    if args.output_dir:
        sys.argv.extend(['--output-dir', args.output_dir])
    if args.device:
        sys.argv.extend(['--device', args.device])
    if args.datasets:
        sys.argv.append('--datasets')
        sys.argv.extend(args.datasets)
    if args.locations:
        sys.argv.append('--locations')
        sys.argv.extend(args.locations)
    if args.compute_umap:
        sys.argv.append('--compute-umap')

    extract_features.main()


def cmd_report(args):
    """F1スコア比較レポートコマンド"""
    print("="*80)
    print("F1スコア比較レポートを生成します")
    print("="*80)

    # scripts/report_f1_comparison.pyを実行
    import subprocess
    result = subprocess.run(
        ['python', 'analysis/scripts/report_f1_comparison.py'],
        cwd=project_root
    )

    if result.returncode == 0:
        print("\n✓ レポート生成完了")
    else:
        print("\n✗ レポート生成に失敗しました")
        sys.exit(1)


def cmd_compare(args):
    """ファインチューニング比較コマンド"""
    print("="*80)
    print("ファインチューニング比較を実行します")
    print("="*80)

    from scripts import visualize_finetune_comparison

    sys.argv = ['visualize_finetune_comparison.py']
    sys.argv.append('--runs')
    sys.argv.extend(args.runs)

    if args.output_dir:
        sys.argv.extend(['--output-dir', args.output_dir])
    if args.metric:
        sys.argv.extend(['--metric', args.metric])

    visualize_finetune_comparison.main()


def cmd_reconstruct(args):
    """リコンストラクション可視化コマンド"""
    print("="*80)
    print("リコンストラクション可視化を実行します")
    print("="*80)

    from scripts import visualize_reconstruction

    sys.argv = ['visualize_reconstruction.py']
    sys.argv.extend(['--model', args.model])

    if args.data_root:
        sys.argv.extend(['--data-root', args.data_root])
    if args.datasets:
        sys.argv.append('--datasets')
        sys.argv.extend(args.datasets)
    if args.num_samples:
        sys.argv.extend(['--num-samples', str(args.num_samples)])
    if args.mask_ratio:
        sys.argv.extend(['--mask-ratio', str(args.mask_ratio)])
    if args.device:
        sys.argv.extend(['--device', args.device])
    if args.output_dir:
        sys.argv.extend(['--output-dir', args.output_dir])
    if args.no_show:
        sys.argv.append('--no-show')

    visualize_reconstruction.main()


def main():
    """メイン関数"""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        print("\nエラー: コマンドを指定してください")
        print("\n利用可能なコマンド: visualize, extract, report, compare, reconstruct")
        sys.exit(1)

    # コマンドに応じて処理を実行
    if args.command == 'visualize':
        cmd_visualize(args)
    elif args.command == 'extract':
        cmd_extract(args)
    elif args.command == 'report':
        cmd_report(args)
    elif args.command == 'compare':
        cmd_compare(args)
    elif args.command == 'reconstruct':
        cmd_reconstruct(args)
    else:
        print(f"不明なコマンド: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
