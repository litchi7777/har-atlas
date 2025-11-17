"""
特徴ベクトル抽出スクリプト

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
import glob
import torch
import torch.nn as nn
from tqdm import tqdm
import yaml
import umap

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "har-unified-dataset" / "src"))

from dataset_info import DATASETS
from models.backbones import Resnet, IntegratedSSLModel


def map_location_to_category(location):
    """
    身体部位の生の名前をカテゴリに変換

    Args:
        location: 元の身体部位名（例: "RightUpperArm", "LeftAnkle"）

    Returns:
        category: カテゴリ名（Arm, Leg, Front, Ankle, Wrist, Phone, Back, Head）
    """
    location_lower = location.lower()

    # ATRデバイス（特定のデバイスID）
    if 'atr01' in location_lower or 'atr02' in location_lower:
        return 'Wrist'
    if 'atr03' in location_lower or 'atr04' in location_lower:
        return 'Arm'

    # Wrist（優先度高）
    if 'wrist' in location_lower:
        return 'Wrist'

    # Ankle（優先度高）
    if 'ankle' in location_lower:
        return 'Ankle'

    # Head
    if any(kw in location_lower for kw in ['head', 'forehead', 'ear']):
        return 'Head'

    # Phone
    if 'phone' in location_lower or 'pocket' in location_lower:
        return 'Phone'

    # Back
    if any(kw in location_lower for kw in ['back', 'lumbar', 'spine']):
        return 'Back'

    # Front (chest, torso, waist)
    if any(kw in location_lower for kw in ['chest', 'torso', 'waist', 'belt', 'hip']):
        return 'Front'

    # Arm (upper arm, forearm, shoulder, hand)
    if any(kw in location_lower for kw in ['arm', 'hand', 'shoulder', 'elbow', 'finger']):
        return 'Arm'

    # Leg (thigh, knee, shin, foot)
    if any(kw in location_lower for kw in ['leg', 'thigh', 'knee', 'shin', 'foot', 'calf']):
        return 'Leg'

    # デフォルト: そのまま返す
    return location


def load_pretrained_model(model_path, window_size=None, device='cuda'):
    """
    事前学習済みモデルを読み込む

    Args:
        model_path: モデルファイルのパス
        window_size: ウィンドウサイズ（Noneの場合は自動検出）
        device: 'cuda' or 'cpu'

    Returns:
        encoder: エンコーダーモデル（評価モード）
        actual_window_size: 実際のウィンドウサイズ
    """
    print(f"Loading model from: {model_path}")

    # チェックポイント読み込み
    checkpoint = torch.load(model_path, map_location=device)

    # window_sizeが指定されていない場合、configから推定
    if window_size is None:
        # チェックポイント内のconfigを確認
        if 'config' in checkpoint:
            config = checkpoint['config']
            if 'sensor_data' in config and 'window_size' in config['sensor_data']:
                window_size = config['sensor_data']['window_size']
                print(f"  Detected window_size from checkpoint config: {window_size}")

        # 実験ディレクトリのconfig.yamlから読み取る
        if window_size is None:
            config_path = Path(model_path).parent.parent / "config.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                if 'sensor_data' in config and 'window_size' in config['sensor_data']:
                    window_size = config['sensor_data']['window_size']
                    print(f"  Detected window_size from config.yaml: {window_size}")

        # それでもNoneなら150と仮定
        if window_size is None:
            window_size = 150
            print(f"  Assuming default window_size: {window_size}")

    print(f"  Window size: {window_size} samples")

    # アーキテクチャ自動選択
    nano_window = window_size < 20
    micro_window = 20 <= window_size < 100

    arch_type = "nano_window" if nano_window else ("micro_window" if micro_window else "default")
    print(f"  Architecture: {arch_type}")

    # バックボーン作成
    backbone = Resnet(
        n_channels=3,
        foundationUK=False,
        micro_window=micro_window,
        nano_window=nano_window
    )

    # IntegratedSSLModel作成
    ssl_tasks = ['binary_permute', 'binary_reverse', 'binary_timewarp']
    model = IntegratedSSLModel(
        backbone=backbone,
        ssl_tasks=ssl_tasks,
        hidden_dim=256,
        n_channels=3,
        sequence_length=window_size,
        device=torch.device(device)
    )

    # 重みを読み込み
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # バックボーン（エンコーダー）のみ取り出し
    encoder = model.backbone
    encoder.eval()

    print(f"  Model loaded successfully. Feature dim: {encoder.output_dim}")

    return encoder, window_size


def load_sensor_data_sample(dataset_name, location, window_size=150, max_samples=100, max_users=20):
    """
    指定されたデータセット・部位のセンサーデータをサンプリングして読み込む

    Args:
        dataset_name: データセット名
        location: 身体部位
        window_size: クリップ後のウィンドウサイズ
        max_samples: 各クラスから取得する最大サンプル数
        max_users: 最大ユーザー数（100以上の場合のみ適用）

    Returns:
        X: センサーデータ (N, 3, window_size)
        y: ラベル (N,)
        indices: 元のインデックス (N,)
    """
    data_root = project_root / "har-unified-dataset" / "data" / "processed"
    dataset_path = data_root / dataset_name

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # パターン: data_root/dataset/USER*/location/ACC/X.npy
    pattern = str(dataset_path / f"*/{location}/ACC/X.npy")
    x_paths = sorted(glob.glob(pattern))

    if not x_paths:
        raise FileNotFoundError(f"No X.npy files found for pattern: {pattern}")

    # nhanesなど巨大なデータセットは最初のN個のみ使用
    if len(x_paths) > 100:
        x_paths_to_use = x_paths[:max_users]
        print(f"  [INFO] Using {len(x_paths_to_use)} out of {len(x_paths)} users for {dataset_name}/{location}")
    else:
        x_paths_to_use = x_paths

    # 全ユーザーのデータを統合
    all_X = []
    all_y = []

    for x_path in x_paths_to_use:
        # ラベルファイルのパスを複数パターン試行
        y_path = x_path.replace("/X.npy", "/Y.npy")

        if not Path(y_path).exists():
            # パターン2: labels.npy
            y_path = Path(x_path).parent.parent / "labels.npy"

        if not Path(y_path).exists():
            continue

        X_data = np.load(x_path, mmap_mode='r')
        y_data = np.load(y_path, mmap_mode='r')

        if y_data.ndim != 1:
            continue

        # ウィンドウクリップ（中央切り出し）
        if X_data.shape[2] > window_size:
            start = (X_data.shape[2] - window_size) // 2
            X_data = X_data[:, :, start:start+window_size]

        all_X.append(X_data)
        all_y.append(y_data)

    if not all_X:
        raise ValueError(f"No valid data found for {dataset_name}/{location}")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # 負のラベルを除外
    valid_mask = y >= 0
    X = X[valid_mask]
    y = y[valid_mask]

    # クラスごとにサンプリング
    unique_labels = np.unique(y)
    sampled_X = []
    sampled_y = []
    sampled_indices = []

    for label in unique_labels:
        label_mask = y == label
        label_indices = np.where(label_mask)[0]

        if len(label_indices) > max_samples:
            selected = np.random.choice(label_indices, max_samples, replace=False)
        else:
            selected = label_indices

        sampled_X.append(X[selected])
        sampled_y.append(y[selected])
        sampled_indices.extend(selected.tolist())

    X = np.concatenate(sampled_X, axis=0)
    y = np.concatenate(sampled_y, axis=0)
    indices = np.array(sampled_indices)

    return X, y, indices


def extract_features(encoder, X, device='cuda', batch_size=256):
    """
    エンコーダーで特徴を抽出

    Args:
        encoder: エンコーダーモデル
        X: センサーデータ (N, 3, T)
        device: デバイス
        batch_size: バッチサイズ

    Returns:
        features: 特徴量 (N, feature_dim)
    """
    encoder.eval()
    features = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            batch = torch.FloatTensor(batch).to(device)
            batch_features = encoder(batch)

            # 3次元以上なら2次元に変換
            if batch_features.ndim > 2:
                batch_features = batch_features.mean(dim=-1)  # Global average pooling

            features.append(batch_features.cpu().numpy())

    features = np.concatenate(features, axis=0)
    return features


def main():
    parser = argparse.ArgumentParser(description='Extract features from pretrained model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--window-size', type=int, default=None,
                        help='Window size (auto-detect if not specified)')
    parser.add_argument('--max-samples', type=int, default=100,
                        help='Max samples per class per dataset-location')
    parser.add_argument('--max-users', type=int, default=20,
                        help='Max users for large datasets (>100 users)')
    parser.add_argument('--output-dir', type=str,
                        default=str(project_root / "analysis" / "embedding_explorer" / "data"),
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output filename without extension (e.g., "features_2.0s")')

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # モデル読み込み
    encoder, window_size = load_pretrained_model(args.model, window_size=args.window_size, device=args.device)

    # データセット・部位の検出
    data_root = project_root / "har-unified-dataset" / "data" / "processed"
    dataset_location_pairs = []

    print("\nScanning datasets...")
    for dataset_path in sorted(data_root.iterdir()):
        if not dataset_path.is_dir():
            continue

        dataset_name = dataset_path.name

        # nhanesは除外しないが、ユーザー数を制限
        pattern = str(dataset_path / "*/*/ACC/X.npy")
        x_paths = glob.glob(pattern)

        locations = set()
        for x_path in x_paths:
            parts = Path(x_path).parts
            if len(parts) >= 3:
                location = parts[-3]
                locations.add(location)

        for location in sorted(locations):
            dataset_location_pairs.append((dataset_name, location))

    print(f"Found {len(dataset_location_pairs)} dataset-location pairs")

    # 特徴抽出
    all_features = []
    all_datasets = []
    all_locations = []
    all_labels = []
    all_dataset_location = []
    all_activity_names = []

    for dataset_name, location in tqdm(dataset_location_pairs, desc="Extracting features"):
        try:
            print(f"\n[START] Processing {dataset_name}/{location}...")
            X, y, _ = load_sensor_data_sample(
                dataset_name, location,
                window_size=window_size,
                max_samples=args.max_samples,
                max_users=args.max_users
            )
            print(f"[LOADED] {dataset_name}/{location}: {len(X)} samples")

            # 特徴抽出
            features = extract_features(encoder, X, device=args.device)

            # アクティビティ名を取得
            dataset_info = DATASETS.get(dataset_name.upper(), {})
            label_mapping = dataset_info.get('labels', {})
            activity_names = [label_mapping.get(int(label), f'unknown_{label}') for label in y]

            # Locationをカテゴリに変換
            location_category = map_location_to_category(location)

            all_features.append(features)
            all_datasets.extend([dataset_name] * len(features))
            all_locations.extend([location_category] * len(features))  # カテゴリ化されたlocation
            all_labels.extend(y.tolist())
            all_dataset_location.extend([f"{dataset_name}/{location_category}"] * len(features))
            all_activity_names.extend(activity_names)

        except Exception as e:
            print(f"Error processing {dataset_name}/{location}: {e}")

    # 特徴量を結合
    all_features = np.concatenate(all_features, axis=0)

    print(f"\nTotal samples: {len(all_features)}")
    print(f"Feature dimension: {all_features.shape[1]}")

    # UMAP埋め込みを事前計算
    print("\nComputing UMAP embeddings...")
    umap_reducer = umap.UMAP(n_components=2, random_state=42, verbose=True)
    umap_embeddings = umap_reducer.fit_transform(all_features)
    print(f"UMAP embeddings computed: {umap_embeddings.shape}")

    # 保存
    # output_fileが指定されていればそれを使用、なければwindow_sizeから自動生成
    if args.output_file:
        time_label = args.output_file.replace('features_', '').replace('.npz', '')
        output_filename = args.output_file if not args.output_file.endswith('.npz') else args.output_file.replace('.npz', '')
    else:
        time_label = {150: '5.0s', 60: '2.0s', 30: '1.0s', 15: '0.5s'}.get(window_size, f'{window_size}samples')
        output_filename = f"features_{time_label}"

    # NPZファイルとして保存（UMAP埋め込みも含める）
    features_path = output_dir / f"{output_filename}.npz"
    np.savez_compressed(
        features_path,
        features=all_features,
        labels=np.array(all_labels),
        umap_embeddings=umap_embeddings
    )
    print(f"Saved features and UMAP embeddings: {features_path}")

    # メタデータをJSONで保存
    metadata = {
        'datasets': all_datasets,
        'locations': all_locations,
        'labels': all_labels,
        'dataset_location': all_dataset_location,
        'activity_names': all_activity_names,
        'window_size': window_size,
        'model_path': str(args.model),
        'total_samples': len(all_features),
        'feature_dim': all_features.shape[1]
    }

    metadata_path = output_dir / f"metadata_{time_label}.json" if not args.output_file else output_dir / f"metadata_{args.output_file.replace('features_', '')}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {metadata_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
