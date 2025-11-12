"""
Foundation Modelの特徴表現を可視化

このスクリプトは、事前学習済みエンコーダーにデータを入力し、
得られた特徴表現（埋め込み）を次元削減して可視化します。

使用方法:
    # 単一モデルで可視化
    python analysis/visualize_embeddings.py \\
      --model experiments/pretrain/run_*/exp_0/models/best_model.pth \\
      --method umap \\
      --color-by body_part

    # 複数モデルで比較
    python analysis/visualize_embeddings.py \\
      --models experiments/pretrain/run_*/exp_0/models/best_model.pth \\
               experiments/pretrain/run_*/exp_1/models/best_model.pth \\
      --method umap \\
      --color-by dataset

    # 特定のデータセット・部位のみ
    python analysis/visualize_embeddings.py \\
      --model experiments/pretrain/run_*/exp_0/models/best_model.pth \\
      --datasets dsads mhealth pamap2 \\
      --locations Torso Chest hand

出力:
    analysis/figures/ 以下に可視化結果を保存
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import json
import glob
import torch
import torch.nn as nn
from tqdm import tqdm

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "har-unified-dataset" / "src"))

from dataset_info import DATASETS
# モデルは直接読み込む
import sys
sys.path.insert(0, str(project_root / "src"))
from models.backbones import IntegratedSSLModel

# 日本語フォント設定（英語ラベルに変更するため無効化）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# スタイル設定
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.0)


def load_pretrained_model(model_path, device='cuda'):
    """
    事前学習済みモデルを読み込む

    Args:
        model_path: モデルファイルのパス
        device: 'cuda' or 'cpu'

    Returns:
        encoder: エンコーダーモデル（評価モード）
    """
    from models.backbones import Resnet

    print(f"Loading model from: {model_path}")

    # チェックポイント読み込み
    checkpoint = torch.load(model_path, map_location=device)

    # バックボーン作成
    backbone = Resnet(n_channels=3, foundationUK=False)

    # IntegratedSSLModel作成
    ssl_tasks = ['binary_permute', 'binary_reverse', 'binary_timewarp']  # デフォルト
    model = IntegratedSSLModel(
        backbone=backbone,
        ssl_tasks=ssl_tasks,
        hidden_dim=256,
        n_channels=3,
        sequence_length=150,
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

    print(f"Model loaded successfully. Feature dim: {encoder.output_dim}")

    return encoder


def load_sensor_data_sample(dataset_name, location, max_samples=500):
    """
    指定されたデータセット・部位のセンサーデータをサンプリングして読み込む

    Args:
        dataset_name: データセット名
        location: 身体部位
        max_samples: 各クラスから取得する最大サンプル数

    Returns:
        X: センサーデータ (N, 3, T)
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

    # 全ユーザーのデータを統合
    all_X = []
    all_y = []

    for x_path in x_paths:
        y_path = x_path.replace("/X.npy", "/Y.npy")

        if not Path(y_path).exists():
            continue

        X_data = np.load(x_path, mmap_mode='r')
        y_data = np.load(y_path, mmap_mode='r')

        if y_data.ndim != 1:
            continue

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
    sampled_indices = []

    for label in unique_labels:
        label_indices = np.where(y == label)[0]
        if len(label_indices) > max_samples:
            sampled = np.random.choice(label_indices, max_samples, replace=False)
        else:
            sampled = label_indices
        sampled_indices.extend(sampled)

    sampled_indices = np.array(sampled_indices)
    np.random.shuffle(sampled_indices)

    return X[sampled_indices], y[sampled_indices], sampled_indices


def extract_features(encoder, X, batch_size=256, device='cuda'):
    """
    エンコーダーで特徴量を抽出

    Args:
        encoder: エンコーダーモデル
        X: センサーデータ (N, 3, T)
        batch_size: バッチサイズ
        device: デバイス

    Returns:
        features: 特徴量 (N, feature_dim)
    """
    encoder.eval()
    features = []

    with torch.no_grad():
        for i in tqdm(range(0, len(X), batch_size), desc="Extracting features"):
            batch = X[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).to(device)

            # エンコーダーに入力
            batch_features = encoder(batch_tensor)

            # バッチサイズ以外の次元を全てflatten
            if batch_features.ndim > 2:
                batch_features = batch_features.reshape(batch_features.shape[0], -1)

            features.append(batch_features.cpu().numpy())

    features = np.concatenate(features, axis=0)
    return features


def reduce_dimensions(features, method='umap', n_components=2):
    """
    次元削減

    Args:
        features: 特徴量 (N, D)
        method: 'umap' or 'tsne'
        n_components: 削減後の次元数

    Returns:
        embedded: 次元削減後の特徴量 (N, n_components)
    """
    print(f"Reducing dimensions using {method.upper()}...")

    if method == 'umap':
        from umap import UMAP
        reducer = UMAP(
            n_components=n_components,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=n_components,
            perplexity=30,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    embedded = reducer.fit_transform(features)
    return embedded


def plot_embeddings(embedded, metadata, color_by='dataset', output_path=None, title=None):
    """
    埋め込みを可視化

    Args:
        embedded: 次元削減後の特徴量 (N, 2)
        metadata: メタデータ {
            'datasets': [...],
            'locations': [...],
            'labels': [...],
            'dataset_location': [...]
        }
        color_by: 'dataset', 'body_part', 'activity', 'dataset_location'
        output_path: 保存先パス
        title: タイトル
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    if color_by == 'dataset':
        unique_values = sorted(set(metadata['datasets']))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_values)))
        color_map = dict(zip(unique_values, colors))

        for val in unique_values:
            mask = np.array(metadata['datasets']) == val
            ax.scatter(
                embedded[mask, 0],
                embedded[mask, 1],
                c=[color_map[val]],
                label=val.upper(),
                alpha=0.6,
                s=10,
                edgecolors='none'
            )

    elif color_by == 'body_part':
        # 身体部位をカテゴリ化
        body_part_categories = {
            'hand': 'Hand/Arm', 'RightWrist': 'Hand/Arm', 'LeftWrist': 'Hand/Arm',
            'RightArm': 'Hand/Arm', 'LeftArm': 'Hand/Arm', 'Wrist': 'Hand/Arm',
            'RightUpperArm': 'Hand/Arm', 'LeftUpperArm': 'Hand/Arm',
            'RightLowerArm': 'Hand/Arm', 'LeftLowerArm': 'Hand/Arm',

            'Chest': 'Torso/Waist', 'Torso': 'Torso/Waist', 'LowerBack': 'Torso/Waist',
            'Back': 'Torso/Waist', 'Hip': 'Torso/Waist', 'chest': 'Torso/Waist',
            'Neck': 'Torso/Waist',

            'ankle': 'Leg/Foot', 'LeftAnkle': 'Leg/Foot', 'RightAnkle': 'Leg/Foot',
            'RightLeg': 'Leg/Foot', 'LeftLeg': 'Leg/Foot', 'Thigh': 'Leg/Foot',
            'RightThigh': 'Leg/Foot', 'LeftThigh': 'Leg/Foot',
            'RightCalf': 'Leg/Foot', 'LeftCalf': 'Leg/Foot',

            'PAX': 'Other'
        }

        categories = [body_part_categories.get(loc, 'Other') for loc in metadata['locations']]
        unique_categories = ['Hand/Arm', 'Torso/Waist', 'Leg/Foot', 'Other']
        colors = {'Hand/Arm': '#ff6b6b', 'Torso/Waist': '#4ecdc4', 'Leg/Foot': '#45b7d1', 'Other': '#gray'}

        for cat in unique_categories:
            mask = np.array(categories) == cat
            if not mask.any():
                continue
            ax.scatter(
                embedded[mask, 0],
                embedded[mask, 1],
                c=colors[cat],
                label=cat,
                alpha=0.6,
                s=10,
                edgecolors='none'
            )

    elif color_by == 'dataset_location':
        unique_values = sorted(set(metadata['dataset_location']))
        colors = plt.cm.tab20(np.linspace(0, 1, min(len(unique_values), 20)))
        color_map = dict(zip(unique_values, colors))

        for val in unique_values:
            mask = np.array(metadata['dataset_location']) == val
            ax.scatter(
                embedded[mask, 0],
                embedded[mask, 1],
                c=[color_map[val]],
                label=val,
                alpha=0.6,
                s=10,
                edgecolors='none'
            )

    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    # 凡例を図の外に配置
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=9,
        frameon=True,
        fancybox=True,
        shadow=True
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_multi_model_comparison(all_embeddings, all_metadata, model_names, color_by='body_part', output_path=None):
    """
    複数モデルの埋め込みを並べて比較

    Args:
        all_embeddings: [embedded1, embedded2, ...]
        all_metadata: [metadata1, metadata2, ...]
        model_names: ['Model 1', 'Model 2', ...]
        color_by: 色分け基準
        output_path: 保存先
    """
    n_models = len(all_embeddings)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))

    if n_models == 1:
        axes = [axes]

    # 身体部位カテゴリ化
    body_part_categories = {
        'hand': 'Hand/Arm', 'RightWrist': 'Hand/Arm', 'LeftWrist': 'Hand/Arm',
        'RightArm': 'Hand/Arm', 'LeftArm': 'Hand/Arm', 'Wrist': 'Hand/Arm',
        'RightUpperArm': 'Hand/Arm', 'LeftUpperArm': 'Hand/Arm',
        'RightLowerArm': 'Hand/Arm', 'LeftLowerArm': 'Hand/Arm',

        'Chest': 'Torso/Waist', 'Torso': 'Torso/Waist', 'LowerBack': 'Torso/Waist',
        'Back': 'Torso/Waist', 'Hip': 'Torso/Waist', 'chest': 'Torso/Waist',
        'Neck': 'Torso/Waist',

        'ankle': 'Leg/Foot', 'LeftAnkle': 'Leg/Foot', 'RightAnkle': 'Leg/Foot',
        'RightLeg': 'Leg/Foot', 'LeftLeg': 'Leg/Foot', 'Thigh': 'Leg/Foot',
        'RightThigh': 'Leg/Foot', 'LeftThigh': 'Leg/Foot',
        'RightCalf': 'Leg/Foot', 'LeftCalf': 'Leg/Foot',

        'PAX': 'Other'
    }

    colors = {'Hand/Arm': '#ff6b6b', 'Torso/Waist': '#4ecdc4', 'Leg/Foot': '#45b7d1', 'Other': '#gray'}

    for idx, (embedded, metadata, model_name, ax) in enumerate(zip(all_embeddings, all_metadata, model_names, axes)):
        if color_by == 'body_part':
            categories = [body_part_categories.get(loc, 'Other') for loc in metadata['locations']]
            unique_categories = ['Hand/Arm', 'Torso/Waist', 'Leg/Foot', 'Other']

            for cat in unique_categories:
                mask = np.array(categories) == cat
                if not mask.any():
                    continue
                ax.scatter(
                    embedded[mask, 0],
                    embedded[mask, 1],
                    c=colors[cat],
                    label=cat if idx == n_models - 1 else None,  # 最後のプロットだけラベル
                    alpha=0.6,
                    s=15,
                    edgecolors='none'
                )

        ax.set_xlabel('Dimension 1', fontsize=10)
        ax.set_ylabel('Dimension 2', fontsize=10)
        ax.set_title(model_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # 共通の凡例を右側に配置
    if n_models > 0:
        axes[-1].legend(
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize foundation model embeddings')
    parser.add_argument('--model', type=str, help='Path to pretrained model')
    parser.add_argument('--models', nargs='+', help='Paths to multiple models for comparison')
    parser.add_argument('--method', type=str, default='umap', choices=['umap', 'tsne'],
                        help='Dimensionality reduction method')
    parser.add_argument('--color-by', type=str, default='body_part',
                        choices=['dataset', 'body_part', 'dataset_location'],
                        help='Color points by')
    parser.add_argument('--datasets', nargs='+', help='Specific datasets to include')
    parser.add_argument('--locations', nargs='+', help='Specific body locations to include')
    parser.add_argument('--max-samples', type=int, default=500,
                        help='Max samples per class per dataset')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = project_root / "analysis" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # モデルのリスト
    if args.models:
        model_paths = args.models
    elif args.model:
        model_paths = [args.model]
    else:
        parser.error("Either --model or --models must be specified")

    # データセット・部位の検出
    data_root = project_root / "har-unified-dataset" / "data" / "processed"
    dataset_location_pairs = []

    for dataset_path in sorted(data_root.iterdir()):
        if not dataset_path.is_dir():
            continue

        dataset_name = dataset_path.name

        if args.datasets and dataset_name not in args.datasets:
            continue

        # USER*/location/ACC/X.npy のパターンで検索
        pattern = str(dataset_path / "*/*/ACC/X.npy")
        x_paths = glob.glob(pattern)

        locations = set()
        for x_path in x_paths:
            parts = Path(x_path).parts
            if len(parts) >= 3:
                location = parts[-3]
                locations.add(location)

        for location in sorted(locations):
            if args.locations and location not in args.locations:
                continue

            dataset_location_pairs.append((dataset_name, location))

    print(f"Found {len(dataset_location_pairs)} dataset-location pairs")

    # 複数モデルの比較
    if len(model_paths) > 1:
        all_embeddings = []
        all_metadata = []
        model_names = []

        for model_path in model_paths:
            print(f"\n{'='*80}")
            print(f"Processing model: {model_path}")
            print(f"{'='*80}")

            # モデル名を抽出
            if 'NHANES' in model_path or 'nhanes' in model_path:
                model_name = 'NHANES only'
            elif 'exp_0' in model_path:
                model_name = 'Hand/Arm Model (16 pairs)'
            elif 'exp_1' in model_path:
                model_name = 'Torso/Waist Model (10 pairs)'
            elif 'exp_2' in model_path:
                model_name = 'Leg/Foot Model (17 pairs)'
            else:
                model_name = Path(model_path).parent.parent.name

            model_names.append(model_name)

            # モデル読み込み
            encoder = load_pretrained_model(model_path, device=args.device)

            # データ収集と特徴抽出
            all_features = []
            all_datasets = []
            all_locations = []
            all_labels = []
            all_dataset_location = []

            for dataset_name, location in tqdm(dataset_location_pairs, desc=f"Processing {model_name}"):
                try:
                    X, y, _ = load_sensor_data_sample(dataset_name, location, max_samples=args.max_samples)

                    # 特徴抽出
                    features = extract_features(encoder, X, device=args.device)

                    all_features.append(features)
                    all_datasets.extend([dataset_name] * len(features))
                    all_locations.extend([location] * len(features))
                    all_labels.extend(y.tolist())
                    all_dataset_location.extend([f"{dataset_name}/{location}"] * len(features))

                except Exception as e:
                    print(f"Error processing {dataset_name}/{location}: {e}")

            # 特徴量を結合
            all_features = np.concatenate(all_features, axis=0)

            print(f"\nTotal samples: {len(all_features)}")
            print(f"Feature dimension: {all_features.shape[1]}")

            # 次元削減
            embedded = reduce_dimensions(all_features, method=args.method)

            all_embeddings.append(embedded)
            all_metadata.append({
                'datasets': all_datasets,
                'locations': all_locations,
                'labels': all_labels,
                'dataset_location': all_dataset_location
            })

        # 比較プロット
        output_path = output_dir / f"embeddings_comparison_{args.method}_{args.color_by}.png"
        plot_multi_model_comparison(all_embeddings, all_metadata, model_names,
                                   color_by=args.color_by, output_path=output_path)

    else:
        # 単一モデル
        model_path = model_paths[0]
        print(f"Processing model: {model_path}")

        encoder = load_pretrained_model(model_path, device=args.device)

        # データ収集と特徴抽出
        all_features = []
        all_datasets = []
        all_locations = []
        all_labels = []
        all_dataset_location = []

        for dataset_name, location in tqdm(dataset_location_pairs, desc="Processing datasets"):
            try:
                X, y, _ = load_sensor_data_sample(dataset_name, location, max_samples=args.max_samples)

                # 特徴抽出
                features = extract_features(encoder, X, device=args.device)

                all_features.append(features)
                all_datasets.extend([dataset_name] * len(features))
                all_locations.extend([location] * len(features))
                all_labels.extend(y.tolist())
                all_dataset_location.extend([f"{dataset_name}/{location}"] * len(features))

            except Exception as e:
                print(f"Error processing {dataset_name}/{location}: {e}")

        # 特徴量を結合
        all_features = np.concatenate(all_features, axis=0)

        print(f"\nTotal samples: {len(all_features)}")
        print(f"Feature dimension: {all_features.shape[1]}")

        # 次元削減
        embedded = reduce_dimensions(all_features, method=args.method)

        # メタデータ
        metadata = {
            'datasets': all_datasets,
            'locations': all_locations,
            'labels': all_labels,
            'dataset_location': all_dataset_location
        }

        # プロット
        model_name = Path(model_path).parent.parent.name
        title = f"Foundation Model Embeddings - {model_name}"
        output_path = output_dir / f"embeddings_{model_name}_{args.method}_{args.color_by}.png"

        plot_embeddings(embedded, metadata, color_by=args.color_by,
                       output_path=output_path, title=title)

    print("\nDone!")


def main_with_args(args):
    """引数オブジェクトを受け取って実行（analyze.pyから呼ばれる用）"""
    # main()の実装をそのまま使用
    main()


if __name__ == '__main__':
    main()
