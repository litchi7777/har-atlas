"""
全ウィンドウサイズの特徴抽出を一括実行

使用方法:
    # デフォルトのモデルパスで実行
    python analysis/embedding_explorer/extract_all_features.py

    # カスタムモデルパスで実行
    python analysis/embedding_explorer/extract_all_features.py \
        --model-5-0s experiments/pretrain/run_*/exp_2/models/checkpoint.pth \
        --model-2-0s experiments/pretrain/run_*/exp_0/models/checkpoint.pth \
        --model-1-0s experiments/pretrain/run_*/exp_1/models/checkpoint.pth \
        --model-0-5s experiments/pretrain/run_*/exp_2/models/checkpoint.pth

    # 最新のモデルを自動検出
    python analysis/embedding_explorer/extract_all_features.py --auto-detect
"""

import os
import sys
import argparse
import subprocess
import glob
from pathlib import Path

# プロジェクトルート
project_root = Path(__file__).parent.parent.parent


def find_latest_checkpoint(pattern, window_size_desc=""):
    """
    指定パターンで最新のチェックポイントを検索

    Args:
        pattern: グロブパターン（例: "experiments/pretrain/*/exp_0/models/checkpoint_epoch_*.pth"）
        window_size_desc: ウィンドウサイズの説明（ログ用）

    Returns:
        最新のチェックポイントパス（str）、見つからない場合はNone
    """
    checkpoints = glob.glob(str(project_root / pattern))
    if not checkpoints:
        print(f"  ⚠️  No checkpoints found for pattern: {pattern}")
        return None

    # epoch番号でソート（数値として）
    def extract_epoch(path):
        try:
            basename = Path(path).name
            if "epoch_" in basename:
                epoch_str = basename.split("epoch_")[1].split(".pth")[0]
                return int(epoch_str)
        except:
            return 0
        return 0

    latest = max(checkpoints, key=extract_epoch)
    epoch = extract_epoch(latest)
    print(f"  ✓ Found {window_size_desc}: {latest} (epoch {epoch})")
    return latest


def extract_features(model_path, max_samples=100, max_users=20, output_dir=None, device='cuda', output_file=None):
    """
    特徴抽出スクリプトを実行

    Args:
        model_path: モデルファイルのパス
        max_samples: 最大サンプル数
        max_users: 最大ユーザー数
        output_dir: 出力ディレクトリ
        device: デバイス ('cuda' or 'cpu')
        output_file: 出力ファイル名 (例: "features_2.0s.npz")

    Returns:
        実行成功時はTrue、失敗時はFalse
    """
    if not Path(model_path).exists():
        print(f"  ❌ Model file not found: {model_path}")
        return False

    if output_dir is None:
        output_dir = project_root / "analysis" / "embedding_explorer" / "data"

    cmd = [
        sys.executable,
        str(project_root / "analysis" / "embedding_explorer" / "extract_features.py"),
        "--model", str(model_path),
        "--max-samples", str(max_samples),
        "--max-users", str(max_users),
        "--output-dir", str(output_dir),
        "--device", device
    ]

    # 出力ファイル名を指定
    if output_file is not None:
        cmd.extend(["--output-file", str(output_file)])

    print(f"  Running: {' '.join(cmd[1:])}")
    print()

    result = subprocess.run(cmd, cwd=str(project_root))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description='Extract features for all window sizes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # モデルパス指定
    parser.add_argument('--model-5-0s', type=str, default=None, dest='model_5_0s',
                        help='Model path for 5.0s (150 samples)')
    parser.add_argument('--model-2-0s', type=str, default=None, dest='model_2_0s',
                        help='Model path for 2.0s (60 samples)')
    parser.add_argument('--model-1-0s', type=str, default=None, dest='model_1_0s',
                        help='Model path for 1.0s (30 samples)')
    parser.add_argument('--model-0-5s', type=str, default=None, dest='model_0_5s',
                        help='Model path for 0.5s (15 samples)')

    # 自動検出
    parser.add_argument('--auto-detect', action='store_true',
                        help='Auto-detect latest model checkpoints')

    # 共通パラメータ
    parser.add_argument('--max-samples', type=int, default=100,
                        help='Max samples per class per dataset-location')
    parser.add_argument('--max-users', type=int, default=20,
                        help='Max users for large datasets')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip if output file already exists')

    args = parser.parse_args()

    print("=" * 60)
    print("Feature Extraction for All Window Sizes")
    print("=" * 60)
    print()

    # 出力ディレクトリ作成
    if args.output_dir is None:
        output_dir = project_root / "analysis" / "embedding_explorer" / "data"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # モデルパスの設定
    models = {}

    if args.auto_detect or all([args.model_5_0s is None, args.model_2_0s is None,
                                  args.model_1_0s is None, args.model_0_5s is None]):
        print("🔍 Auto-detecting model checkpoints...")
        # 5.0s: window_size=150 (通常はexp_2)
        models['5.0s'] = find_latest_checkpoint(
            "experiments/pretrain/*/exp_2/models/checkpoint_epoch_*.pth",
            "5.0s (150 samples)"
        )
        # Fallback
        if models['5.0s'] is None:
            models['5.0s'] = "experiments/pretrain/run_20251111_171703/exp_2/models/checkpoint_epoch_45.pth"
            print(f"  Using fallback: {models['5.0s']}")

        # 2.0s: window_size=60 (通常はexp_0)
        models['2.0s'] = find_latest_checkpoint(
            "experiments/pretrain/run_20251112_*/exp_0/models/checkpoint_epoch_*.pth",
            "2.0s (60 samples)"
        )
        if models['2.0s'] is None:
            models['2.0s'] = "experiments/pretrain/run_20251112_192545/exp_0/models/checkpoint_epoch_40.pth"
            print(f"  Using fallback: {models['2.0s']}")

        # 1.0s: window_size=30 (通常はexp_1)
        models['1.0s'] = find_latest_checkpoint(
            "experiments/pretrain/run_20251112_*/exp_1/models/checkpoint_epoch_*.pth",
            "1.0s (30 samples)"
        )
        if models['1.0s'] is None:
            models['1.0s'] = "experiments/pretrain/run_20251112_192545/exp_1/models/checkpoint_epoch_40.pth"
            print(f"  Using fallback: {models['1.0s']}")

        # 0.5s: window_size=15 (通常はexp_2 in run_20251112*)
        models['0.5s'] = find_latest_checkpoint(
            "experiments/pretrain/run_20251112_*/exp_2/models/checkpoint_epoch_*.pth",
            "0.5s (15 samples)"
        )
        if models['0.5s'] is None:
            models['0.5s'] = "experiments/pretrain/run_20251112_192545/exp_2/models/checkpoint_epoch_39.pth"
            print(f"  Using fallback: {models['0.5s']}")
        print()
    else:
        # 手動指定されたモデルパスを使用
        models['5.0s'] = args.model_5_0s or "experiments/pretrain/run_20251111_171703/exp_2/models/checkpoint_epoch_45.pth"
        models['2.0s'] = args.model_2_0s or "experiments/pretrain/run_20251112_192545/exp_0/models/checkpoint_epoch_40.pth"
        models['1.0s'] = args.model_1_0s or "experiments/pretrain/run_20251112_192545/exp_1/models/checkpoint_epoch_40.pth"
        models['0.5s'] = args.model_0_5s or "experiments/pretrain/run_20251112_192545/exp_2/models/checkpoint_epoch_39.pth"

    print("=" * 60)
    print("Starting feature extraction...")
    print("=" * 60)
    print()

    success_count = 0
    total_count = len(models)

    for i, (window_size, model_path) in enumerate(models.items(), 1):
        # 出力ファイルの存在確認
        output_file = output_dir / f"features_{window_size}.npz"
        if args.skip_existing and output_file.exists():
            print(f"📊 [{i}/{total_count}] Skipping {window_size} (file exists)")
            print(f"  Output: {output_file}")
            print()
            success_count += 1
            continue

        print(f"📊 [{i}/{total_count}] Extracting features for {window_size}...")
        print(f"  Model: {model_path}")

        # 出力ファイル名を明示的に指定
        output_filename = f"features_{window_size}"
        success = extract_features(
            model_path,
            max_samples=args.max_samples,
            max_users=args.max_users,
            output_dir=output_dir,
            device=args.device,
            output_file=output_filename
        )

        if success:
            print(f"  ✓ {window_size} features extracted successfully")
            success_count += 1
        else:
            print(f"  ❌ Failed to extract {window_size} features")

        print()

    print("=" * 60)
    if success_count == total_count:
        print(f"✓ All {total_count} feature sets extracted successfully!")
    else:
        print(f"⚠️  {success_count}/{total_count} feature sets extracted successfully")
    print("=" * 60)
    print()
    print("Output files:")

    # 出力ファイル一覧
    for file in sorted(output_dir.glob("*")):
        size_mb = file.stat().st_size / 1024 / 1024
        print(f"  {file.name:30s} {size_mb:>8.2f} MB")

    print()
    print("Next step: Start the server with:")
    print("  python analysis/embedding_explorer/server.py --port 8050 --debug")


if __name__ == '__main__':
    main()
