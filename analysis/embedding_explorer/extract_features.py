"""
特徴ベクトル抽出スクリプト（リファクタリング版）

事前学習済みモデルから全データセット・全locationの特徴ベクトルを抽出し、
NPZファイルとJSONメタデータとして保存します。

使用方法:
    python analysis/embedding_explorer/extract_features.py \\
        --model experiments/pretrain/run_*/exp_0/models/checkpoint.pth \\
        --window-size 150 \\
        --max-samples 100 \\
        --output-dir analysis/embedding_explorer/data
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import umap

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "analysis"))

# 共通モジュールをインポート
from common import (
    load_pretrained_model,
    extract_features,
    load_sensor_data,
    find_dataset_location_pairs,
    get_label_dict,
    categorize_body_part,
    reduce_dimensions,
)


def main():
    parser = argparse.ArgumentParser(description='Extract features from pretrained model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to pretrained model')
    parser.add_argument('--window-size', type=int, default=None,
                        help='Window size (auto-detect if not specified)')
    parser.add_argument('--max-samples', type=int, default=100,
                        help='Max samples per class per dataset')
    parser.add_argument('--max-users', type=int, default=20,
                        help='Max users for large datasets')
    parser.add_argument('--output-dir', type=str,
                        default='analysis/embedding_explorer/data',
                        help='Output directory for features')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets to include')
    parser.add_argument('--locations', nargs='+', default=None,
                        help='Specific body locations to include')
    parser.add_argument('--compute-umap', action='store_true',
                        help='Compute UMAP embeddings and save')

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # モデル読み込み
    print("="*80)
    print("Loading pretrained model...")
    print("="*80)
    encoder, window_size = load_pretrained_model(
        args.model,
        window_size=args.window_size,
        device=args.device
    )

    if args.window_size is None:
        args.window_size = window_size

    # データセット・部位のペアを取得
    dataset_location_pairs = find_dataset_location_pairs(
        dataset_filter=args.datasets,
        location_filter=args.locations
    )

    if not dataset_location_pairs:
        print("Error: No dataset-location pairs found")
        return

    print(f"\nFound {len(dataset_location_pairs)} dataset-location pairs")

    # 特徴抽出
    all_features = []
    all_datasets = []
    all_locations = []
    all_labels = []
    all_activity_names = []
    all_dataset_location = []
    all_body_part_categories = []

    print("\n" + "="*80)
    print("Extracting features...")
    print("="*80)

    for dataset_name, location in tqdm(dataset_location_pairs, desc="Processing"):
        try:
            # データ読み込み
            X, y, _ = load_sensor_data(
                dataset_name,
                location,
                window_size=args.window_size,
                max_samples_per_class=args.max_samples,
                max_users=args.max_users
            )

            # 特徴抽出
            features = extract_features(
                encoder, X,
                device=args.device,
                show_progress=False
            )

            # アクティビティ名を取得
            label_mapping = get_label_dict(dataset_name)
            activity_names = [
                label_mapping.get(int(label), f'unknown_{label}')
                for label in y
            ]

            # 身体部位カテゴリ化
            body_part_category = categorize_body_part(location)

            # 保存
            all_features.append(features)
            all_datasets.extend([dataset_name] * len(features))
            all_locations.extend([location] * len(features))
            all_labels.extend(y.tolist())
            all_activity_names.extend(activity_names)
            all_dataset_location.extend([f"{dataset_name}/{location}"] * len(features))
            all_body_part_categories.extend([body_part_category] * len(features))

        except Exception as e:
            print(f"\nError processing {dataset_name}/{location}: {e}")
            continue

    # 特徴量を結合
    all_features = np.vstack(all_features)

    print(f"\n{'='*80}")
    print(f"Feature extraction complete")
    print(f"{'='*80}")
    print(f"Total samples: {len(all_features)}")
    print(f"Feature dimension: {all_features.shape[1]}")

    # メタデータ作成
    metadata = {
        'model_path': str(args.model),
        'window_size': args.window_size,
        'num_samples': len(all_features),
        'feature_dim': all_features.shape[1],
        'datasets': all_datasets,
        'locations': all_locations,
        'labels': all_labels,
        'activity_names': all_activity_names,
        'dataset_location': all_dataset_location,
        'body_part_categories': all_body_part_categories,
    }

    # ウィンドウサイズに応じたラベル
    time_label = {150: '5.0s', 60: '2.0s', 30: '1.0s', 15: '0.5s'}.get(
        args.window_size, f'{args.window_size}samples'
    )

    # 特徴量を保存（NPZ形式）
    features_path = output_dir / f"features_{time_label}.npz"
    np.savez_compressed(features_path, features=all_features)
    print(f"\nSaved features: {features_path}")

    # メタデータを保存（JSON形式）
    metadata_path = output_dir / f"metadata_{time_label}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {metadata_path}")

    # UMAP埋め込みを計算・保存（オプション）
    if args.compute_umap:
        print(f"\n{'='*80}")
        print("Computing UMAP embeddings...")
        print(f"{'='*80}")

        umap_embedded = reduce_dimensions(
            all_features,
            method='umap',
            n_components=2
        )

        umap_path = output_dir / f"umap_{time_label}.npz"
        np.savez_compressed(umap_path, embedding=umap_embedded)
        print(f"Saved UMAP embeddings: {umap_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
