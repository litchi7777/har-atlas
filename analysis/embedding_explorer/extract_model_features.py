"""
指定したモデルから特徴量を抽出してFlask server用に保存

使用方法:
    # 128次元（projection後）- プロトタイプ付き（デフォルト）
    python analysis/embedding_explorer/extract_model_features.py \
        --model-path experiments/pretrain/run_*/checkpoints/checkpoint_epoch_60.pth \
        --model-name "epoch60"

    # 512次元（backbone出力）- 従来方式
    python analysis/embedding_explorer/extract_model_features.py \
        --model-path experiments/pretrain/run_*/checkpoints/checkpoint_epoch_60.pth \
        --model-name "epoch60_backbone" \
        --feature-space backbone
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
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


def generate_model_name(model_path: str, epoch: int = None) -> str:
    """
    モデルパスからモデル名を自動生成

    Args:
        model_path: チェックポイントのパス
        epoch: エポック番号（Noneの場合はファイル名から取得）

    Returns:
        model_name: "mtl_hierarchical_win60_ep60" のような名前
    """
    import yaml
    model_path = Path(model_path)

    # エポック番号をファイル名から取得
    if epoch is None:
        filename = model_path.stem  # checkpoint_epoch_60 or best_model
        if 'epoch' in filename:
            try:
                epoch = int(filename.split('epoch_')[-1])
            except ValueError:
                epoch = 0
        else:
            epoch = 'best'

    # config.yamlを読み込み
    config_path = model_path.parent.parent / "config.yaml"
    if not config_path.exists():
        # checkpoints/の親を探す
        config_path = model_path.parent.parent / "config.yaml"

    name_parts = []

    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # MTL/Hierarchicalの状態
        mtl_enabled = config.get('multitask', {}).get('enabled', False)
        hier_enabled = config.get('hierarchical', {}).get('enabled', False)

        if mtl_enabled and hier_enabled:
            name_parts.append('combined')
        elif mtl_enabled:
            name_parts.append('mtl')
        elif hier_enabled:
            name_parts.append('hierarchical')
        else:
            name_parts.append('baseline')

        # ウィンドウサイズ
        window_size = config.get('data', {}).get('window_size', 60)
        name_parts.append(f'win{window_size}')

        # データセット数
        pairs = config.get('data', {}).get('dataset_location_pairs', [])
        if pairs:
            datasets = set(p[0] for p in pairs)
            name_parts.append(f'{len(datasets)}ds')

    # エポック
    name_parts.append(f'ep{epoch}')

    return '_'.join(name_parts)


def load_projections_and_prototypes(checkpoint_path: str, device: str = 'cuda'):
    """
    チェックポイントからprojection層とプロトタイプを読み込む

    Returns:
        projections: dict[body_part -> nn.Sequential]
        prototypes: dict[body_part -> np.ndarray]
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    loss_state = checkpoint.get('loss_fn_state_dict', {})

    # Body part一覧を取得
    body_parts = set()
    for key in loss_state.keys():
        if 'prototypes.prototypes.' in key:
            # hierarchical_loss.prototypes.prototypes.wrist -> wrist
            bp = key.split('.')[-1]
            body_parts.add(bp)

    projections = {}
    prototypes = {}

    for bp in body_parts:
        # Projection層を再構築 (Linear -> ReLU -> Linear)
        w0_key = f'hierarchical_loss.prototypes.projections.{bp}.0.weight'
        b0_key = f'hierarchical_loss.prototypes.projections.{bp}.0.bias'
        w2_key = f'hierarchical_loss.prototypes.projections.{bp}.2.weight'
        b2_key = f'hierarchical_loss.prototypes.projections.{bp}.2.bias'

        if all(k in loss_state for k in [w0_key, b0_key, w2_key, b2_key]):
            w0 = loss_state[w0_key]
            b0 = loss_state[b0_key]
            w2 = loss_state[w2_key]
            b2 = loss_state[b2_key]

            proj = nn.Sequential(
                nn.Linear(w0.shape[1], w0.shape[0]),
                nn.ReLU(),
                nn.Linear(w2.shape[1], w2.shape[0])
            )
            proj[0].weight.data = w0
            proj[0].bias.data = b0
            proj[2].weight.data = w2
            proj[2].bias.data = b2
            proj.to(device)
            proj.eval()
            projections[bp] = proj

        # プロトタイプを取得
        proto_key = f'hierarchical_loss.prototypes.prototypes.{bp}'
        if proto_key in loss_state:
            prototypes[bp] = loss_state[proto_key].cpu().numpy()

    print(f"  Loaded projections for: {list(projections.keys())}")
    print(f"  Loaded prototypes: {[(bp, p.shape) for bp, p in prototypes.items()]}")

    return projections, prototypes


def main():
    parser = argparse.ArgumentParser(description='特定モデルから特徴量抽出')
    parser.add_argument('--model-path', type=str, required=True,
                        help='モデルチェックポイントのパス')
    parser.add_argument('--model-name', type=str, default=None,
                        help='モデルの識別名（省略時は自動生成）')
    parser.add_argument('--feature-space', type=str, default='projected',
                        choices=['projected', 'backbone'],
                        help='特徴空間: projected=128次元（プロトタイプ付き）, backbone=512次元')
    parser.add_argument('--max-samples', type=int, default=10,
                        help='各dataset-locationからの最大サンプル数')
    parser.add_argument('--max-users', type=int, default=20,
                        help='大規模データセット用の最大ユーザー数')
    # デフォルトはスクリプトと同じディレクトリのdata/
    script_dir = Path(__file__).parent
    parser.add_argument('--output-dir', type=str,
                        default=str(script_dir / 'data'),
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

    # モデル名を自動生成（指定がない場合）
    if args.model_name is None:
        args.model_name = generate_model_name(args.model_path)
        print(f"[INFO] Auto-generated model name: {args.model_name}")

    print("="*80)
    print(f"Model Feature Extraction: {args.model_name}")
    print("="*80)

    # モデル読み込み
    print(f"\n[1/5] Loading model from: {args.model_path}")
    encoder, window_size = load_pretrained_model(args.model_path, device=args.device)
    print(f"  Window size: {window_size} samples")
    print(f"  Model name: {args.model_name}")
    print(f"  Feature space: {args.feature_space}")

    # Projection層とプロトタイプを読み込み（projected modeのみ）
    projections = {}
    prototypes = {}
    if args.feature_space == 'projected':
        print(f"\n[1.5/5] Loading projections and prototypes...")
        projections, prototypes = load_projections_and_prototypes(args.model_path, device=args.device)

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

    # Body partカテゴリ -> プロトタイプのbody part名へのマッピング
    body_part_to_prototype_key = {
        'Wrist': 'wrist',
        'Ankle': 'leg',      # Ankleはlegに含める
        'Head': 'head',
        'Phone': 'hip',      # Phone/Pocketはhipに含める
        'Back': 'chest',     # Backはchestに含める
        'Front': 'chest',    # Front(chest, hip, torso)はchestに含める
        'Arm': 'wrist',      # Armはwristに含める
        'Leg': 'leg',
        'PAX': 'hip',        # PAX(NHANES)はhipに含める
    }

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

            # 特徴抽出（512次元 backbone出力）
            features_backbone = extract_features(
                encoder, X,
                batch_size=256,
                device=args.device,
                show_progress=False
            )

            # メタデータ
            body_part = categorize_body_part(location_name)
            # プロトタイプのキーに変換
            proto_key = body_part_to_prototype_key.get(body_part, body_part.lower())

            # Projection適用（128次元に変換）
            if args.feature_space == 'projected' and proto_key in projections:
                proj = projections[proto_key]
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features_backbone).to(args.device)
                    features = proj(features_tensor).cpu().numpy()
            else:
                features = features_backbone
                if args.feature_space == 'projected' and proto_key not in projections:
                    print(f"  [WARNING] No projection for body_part '{body_part}' (key='{proto_key}'), using backbone features")

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

    # プロトタイプをt-SNE用に準備（projected modeのみ）
    prototype_features = None
    prototype_metadata = None
    if args.feature_space == 'projected' and prototypes:
        print(f"\n[3.5/5] Preparing prototypes for t-SNE...")

        # Atlasからatomic motion名を取得
        atlas_path = project_root / 'docs' / 'atlas' / 'activity_mapping.json'
        atomic_motion_names = {}  # {body_part: {index: name}}

        if atlas_path.exists():
            with open(atlas_path, 'r') as f:
                atlas = json.load(f)

            # 各body partのユニークなatomic motionを収集
            for bp in ['wrist', 'chest', 'leg', 'hip', 'head']:
                motions = set()
                for dataset, data in atlas.items():
                    if dataset in ['version', 'description', 'note']:
                        continue
                    activities = data.get('activities', {})
                    for activity, info in activities.items():
                        for bp_key, motion_list in info.get('atomic_motions', {}).items():
                            if bp_key == bp:
                                motions.update(motion_list)

                # ソートしてインデックス付け（プロトタイプのインデックスに対応）
                sorted_motions = sorted(motions)
                atomic_motion_names[bp] = {i: name for i, name in enumerate(sorted_motions)}

            print(f"  Loaded atomic motion names from Atlas")
            for bp in atomic_motion_names:
                print(f"    {bp}: {len(atomic_motion_names[bp])} motions")

        proto_list = []
        proto_meta = {'body_parts': [], 'prototype_ids': [], 'atomic_motion_names': []}
        for bp, proto_array in prototypes.items():
            proto_list.append(proto_array)
            for i in range(len(proto_array)):
                proto_meta['body_parts'].append(bp)
                # Atlasからatomic motion名を取得、なければインデックスを使用
                if bp in atomic_motion_names and i < len(atomic_motion_names[bp]):
                    motion_name = atomic_motion_names[bp].get(i, f"{bp}_{i}")
                else:
                    motion_name = f"{bp}_{i}"
                proto_meta['prototype_ids'].append(f"{bp}_{i}")
                proto_meta['atomic_motion_names'].append(motion_name)
        prototype_features = np.vstack(proto_list)
        prototype_metadata = proto_meta
        print(f"  Total prototypes: {len(prototype_features)}")

    # t-SNE計算（サンプル + プロトタイプを一緒に）
    print(f"\n[4/5] Computing t-SNE embeddings...")
    from sklearn.manifold import TSNE

    if prototype_features is not None:
        # サンプルとプロトタイプを結合してt-SNE
        combined_features = np.vstack([all_features, prototype_features])
        print(f"  Combined features shape: {combined_features.shape} (samples: {len(all_features)}, prototypes: {len(prototype_features)})")

        # perplexityを調整（サンプル数が少ない場合）
        perplexity = min(30, len(combined_features) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=1000, random_state=42, verbose=1)
        combined_embeddings = tsne.fit_transform(combined_features)

        tsne_embeddings = combined_embeddings[:len(all_features)]
        prototype_embeddings = combined_embeddings[len(all_features):]
        print(f"  t-SNE embeddings shape: {tsne_embeddings.shape}")
        print(f"  Prototype embeddings shape: {prototype_embeddings.shape}")
    else:
        perplexity = min(30, len(all_features) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=1000, random_state=42, verbose=1)
        tsne_embeddings = tsne.fit_transform(all_features)
        prototype_embeddings = None
        print(f"  t-SNE embeddings shape: {tsne_embeddings.shape}")

    # 保存
    print(f"\n[5/5] Saving to {output_dir}...")

    # 特徴量、生データ、t-SNE埋め込みを保存（.npz形式）
    features_path = output_dir / f"features_{args.model_name}.npz"
    save_dict = {
        'features': all_features,
        'sensor_data': all_sensor_data,
        'tsne_embeddings': tsne_embeddings,
        'feature_space': args.feature_space
    }
    if prototype_features is not None:
        save_dict['prototype_features'] = prototype_features
        save_dict['prototype_embeddings'] = prototype_embeddings
    np.savez_compressed(features_path, **save_dict)
    print(f"  Saved features, sensor data, and t-SNE embeddings: {features_path}")

    # メタデータ保存（JSON形式）
    metadata_path = output_dir / f"metadata_{args.model_name}.json"
    all_metadata['feature_space'] = args.feature_space
    if prototype_metadata is not None:
        all_metadata['prototypes'] = prototype_metadata
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
