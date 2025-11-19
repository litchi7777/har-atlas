"""
指定したモデルから特徴量を抽出してFlask server用に保存

使用方法:
    python analysis/embedding_explorer/extract_model_features.py \
        --model-path experiments/pretrain/run_20251117_074851/exp_0/models/best_model.pth \
        --model-name "rotation" \
        --max-samples 200
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from analysis.common import (
    load_pretrained_model,
    extract_features,
    load_sensor_data,
    find_dataset_location_pairs,
    get_label_dict,
    categorize_body_part,
)


def main():
    parser = argparse.ArgumentParser(description='特定モデルから特徴量抽出')
    parser.add_argument('--model-path', type=str, required=True,
                        help='モデルチェックポイントのパス')
    parser.add_argument('--model-name', type=str, required=True,
                        help='モデルの識別名（例: rotation, baseline, multi_task）')
    parser.add_argument('--max-samples', type=int, default=200,
                        help='各dataset-locationからの最大サンプル数')
    parser.add_argument('--max-users', type=int, default=20,
                        help='大規模データセット用の最大ユーザー数')
    parser.add_argument('--output-dir', type=str,
                        default='analysis/embedding_explorer/data',
                        help='出力ディレクトリ')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='使用デバイス')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='特定のデータセットのみ（例: dsads mhealth）')
    parser.add_argument('--locations', nargs='+', default=None,
                        help='特定の身体部位のみ（例: Wrist Hip）')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print(f"Model Feature Extraction: {args.model_name}")
    print("="*80)

    # モデル読み込み
    print(f"\n[1/4] Loading model from: {args.model_path}")
    encoder, window_size = load_pretrained_model(args.model_path, device=args.device)
    print(f"  Window size: {window_size} samples")
    print(f"  Model name: {args.model_name}")

    # データセット-location ペアを検索
    print(f"\n[2/4] Finding dataset-location pairs...")
    dataset_location_pairs = find_dataset_location_pairs()

    # フィルタリング
    if args.datasets:
        dataset_location_pairs = [
            (ds, loc) for ds, loc in dataset_location_pairs
            if ds in args.datasets
        ]
    if args.locations:
        dataset_location_pairs = [
            (ds, loc) for ds, loc in dataset_location_pairs
            if loc in args.locations
        ]

    print(f"Found {len(dataset_location_pairs)} dataset-location pairs")

    # アクティビティ名マッピングをインポート（ループの外で一度だけ）
    import sys
    dataset_info_dir = project_root / 'har-unified-dataset' / 'src'
    if str(dataset_info_dir) not in sys.path:
        sys.path.insert(0, str(dataset_info_dir))

    try:
        from dataset_info import DATASETS
        print(f"[INFO] Loaded DATASETS with {len(DATASETS)} datasets")
    except ImportError as e:
        print(f"[WARNING] Could not import DATASETS: {e}")
        DATASETS = {}

    # 特徴抽出
    print(f"\n[3/4] Extracting features...")
    all_features = []
    all_sensor_data = []  # 生データも保存
    all_metadata = {
        'datasets': [],
        'locations': [],
        'labels': [],
        'dataset_location': [],
        'activity_names': [],
        'body_parts': []
    }

    for dataset_name, location_name in tqdm(dataset_location_pairs, desc="Processing datasets"):
        print(f"\n[START] Processing {dataset_name}/{location_name}...")

        try:
            # データ読み込み
            X, y, metadata_dict = load_sensor_data(
                dataset_name=dataset_name,
                location=location_name,
                window_size=window_size,
                max_samples_per_class=args.max_samples,
                max_users=args.max_users
            )

            # アクティビティ名のマッピングを取得
            dataset_key = dataset_name.upper()
            if dataset_key in DATASETS and 'labels' in DATASETS[dataset_key]:
                activity_names = DATASETS[dataset_key]['labels']
            else:
                print(f"  [WARNING] No labels found for {dataset_name}, using default mapping")
                # デフォルトのマッピング（クラスIDをそのまま使用）
                unique_labels = sorted(set(y.tolist()))
                activity_names = {label: f"Activity_{label}" for label in unique_labels}

            if len(X) == 0:
                print(f"  [SKIP] No data for {dataset_name}/{location_name}")
                continue

            print(f"[LOADED] {dataset_name}/{location_name}: {len(X)} samples")

            # 特徴抽出
            features = extract_features(
                encoder, X,
                batch_size=256,
                device=args.device,
                show_progress=False
            )

            # メタデータ
            body_part = categorize_body_part(location_name)

            # 各サンプルのアクティビティ名を取得（yはクラスID）
            # activity_namesは辞書またはリストの可能性がある
            try:
                sample_activity_names = []

                # activity_namesが辞書の場合
                if isinstance(activity_names, dict):
                    for label in y:
                        label_int = int(label)
                        if label_int in activity_names:
                            sample_activity_names.append(activity_names[label_int])
                        else:
                            print(f"  [WARNING] Label {label_int} not found in activity_names dict (keys={list(activity_names.keys())})")
                            sample_activity_names.append(f"Unknown_{label_int}")
                # activity_namesがリストの場合
                elif isinstance(activity_names, (list, tuple)):
                    for label in y:
                        label_int = int(label)
                        if 0 <= label_int < len(activity_names):
                            sample_activity_names.append(activity_names[label_int])
                        else:
                            print(f"  [WARNING] Label {label_int} out of range for activity_names list (len={len(activity_names)})")
                            sample_activity_names.append(f"Unknown_{label_int}")
                else:
                    raise TypeError(f"Unexpected activity_names type: {type(activity_names)}")

            except Exception as e:
                print(f"  [ERROR] Failed to convert labels to activity names: {e}")
                print(f"  [DEBUG] y type: {type(y)}, y[:5]: {y[:5] if len(y) > 0 else 'empty'}")
                print(f"  [DEBUG] activity_names type: {type(activity_names)}, len: {len(activity_names)}")
                if isinstance(activity_names, dict):
                    print(f"  [DEBUG] activity_names keys: {list(activity_names.keys())[:10]}")
                raise

            all_features.append(features)
            all_sensor_data.append(X)  # 生データを保存
            all_metadata['datasets'].extend([dataset_name] * len(X))
            all_metadata['locations'].extend([location_name] * len(X))
            all_metadata['labels'].extend(y.tolist())
            all_metadata['dataset_location'].extend([f"{dataset_name}/{location_name}"] * len(X))
            all_metadata['activity_names'].extend(sample_activity_names)  # 修正: 各サンプルに対応
            all_metadata['body_parts'].extend([body_part] * len(X))

        except Exception as e:
            print(f"Error processing {dataset_name}/{location_name}: {e}")
            continue

    # 結合
    all_features = np.vstack(all_features)
    all_sensor_data = np.vstack(all_sensor_data)  # 生データも結合
    print(f"\nTotal samples: {len(all_features)}")
    print(f"Feature dimension: {all_features.shape[1]}")
    print(f"Sensor data shape: {all_sensor_data.shape}")

    # t-SNE計算（事前計算）
    print(f"\n[4/5] Computing t-SNE embeddings...")
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, verbose=1)
    tsne_embeddings = tsne.fit_transform(all_features)
    print(f"  t-SNE embeddings shape: {tsne_embeddings.shape}")

    # 保存
    print(f"\n[5/5] Saving to {output_dir}...")

    # 特徴量、生データ、t-SNE埋め込みを保存（.npz形式）
    features_path = output_dir / f"features_{args.model_name}.npz"
    np.savez_compressed(features_path,
                        features=all_features,
                        sensor_data=all_sensor_data,
                        tsne_embeddings=tsne_embeddings)
    print(f"  Saved features, sensor data, and t-SNE embeddings: {features_path}")

    # メタデータ保存（JSON形式）
    metadata_path = output_dir / f"metadata_{args.model_name}.json"
    with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")

    # 統計情報
    print("\n" + "="*80)
    print("Statistics:")
    print("="*80)
    print(f"Total samples: {len(all_features)}")
    print(f"Unique datasets: {len(set(all_metadata['datasets']))}")
    print(f"Unique locations: {len(set(all_metadata['locations']))}")
    print(f"Unique activities: {len(set(all_metadata['activity_names']))}")
    print(f"Unique body parts: {set(all_metadata['body_parts'])}")

    print("\n✅ Feature extraction completed!")


if __name__ == "__main__":
    main()
