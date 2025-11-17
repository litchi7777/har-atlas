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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


def load_pretrained_model(model_path, window_size=None, device='cuda'):
    """
    事前学習済みモデルを読み込む（ウィンドウサイズ自動検出対応）

    Args:
        model_path: モデルファイルのパス
        window_size: ウィンドウサイズ（Noneの場合は150と仮定）
        device: 'cuda' or 'cpu'

    Returns:
        encoder: エンコーダーモデル（評価モード）
        actual_window_size: 実際のウィンドウサイズ
    """
    from models.backbones import Resnet

    print(f"Loading model from: {model_path}")

    # チェックポイント読み込み
    checkpoint = torch.load(model_path, map_location=device)

    # window_sizeが指定されていない場合、configから推定を試みる
    if window_size is None:
        # チェックポイント内のconfigを確認
        if 'config' in checkpoint:
            config = checkpoint['config']
            if 'sensor_data' in config and 'window_size' in config['sensor_data']:
                window_size = config['sensor_data']['window_size']
                print(f"  Detected window_size from checkpoint config: {window_size}")

        # 実験ディレクトリのconfig.yamlから読み取る
        if window_size is None:
            import yaml
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
    nano_window = window_size < 20   # 15サンプル
    micro_window = 20 <= window_size < 100  # 30, 60サンプル
    # 150サンプルはdefault

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
    ssl_tasks = ['binary_permute', 'binary_reverse', 'binary_timewarp']  # デフォルト
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


def load_sensor_data_sample(dataset_name, location, window_size=150, max_samples=500):
    """
    指定されたデータセット・部位のセンサーデータをサンプリングして読み込む
    必要に応じてウィンドウクリップを実行

    Args:
        dataset_name: データセット名
        location: 身体部位
        window_size: クリップ後のウィンドウサイズ (デフォルト: 150)
        max_samples: 各クラスから取得する最大サンプル数

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

    # 全ユーザーのデータを統合
    # nhanesなど巨大なデータセットは最初の20ユーザーのみ使用
    max_users = 20 if len(x_paths) > 100 else len(x_paths)
    x_paths_to_use = x_paths[:max_users]

    if len(x_paths) > max_users:
        print(f"  [INFO] Using {max_users} out of {len(x_paths)} users for {dataset_name}/{location}")

    all_X = []
    all_y = []

    for x_path in x_paths_to_use:
        y_path = x_path.replace("/X.npy", "/Y.npy")

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
            'dataset_location': [...],
            'activity_names': [...] (optional)
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

    elif color_by == 'activity':
        # アクティビティ名で色分け
        if 'activity_names' in metadata and metadata['activity_names']:
            activity_names = metadata['activity_names']
            unique_activities = sorted(set(activity_names))
            # 色数を動的に調整（tab20c は最大20色、それ以上ならhsvを使用）
            if len(unique_activities) <= 20:
                colors = plt.cm.tab20(np.linspace(0, 1, len(unique_activities)))
            else:
                colors = plt.cm.hsv(np.linspace(0, 1, len(unique_activities)))
            color_map = dict(zip(unique_activities, colors))

            for activity in unique_activities:
                mask = np.array(activity_names) == activity
                if not mask.any():
                    continue
                ax.scatter(
                    embedded[mask, 0],
                    embedded[mask, 1],
                    c=[color_map[activity]],
                    label=activity,
                    alpha=0.6,
                    s=10,
                    edgecolors='none'
                )
        else:
            # activity_namesがない場合はラベルIDで色分け
            labels = metadata['labels']
            unique_labels = sorted(set(labels))
            colors = plt.cm.tab20(np.linspace(0, 1, min(len(unique_labels), 20)))
            color_map = dict(zip(unique_labels, colors))

            for label in unique_labels:
                mask = np.array(labels) == label
                if not mask.any():
                    continue
                ax.scatter(
                    embedded[mask, 0],
                    embedded[mask, 1],
                    c=[color_map[label]],
                    label=f'Activity {label}',
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


def plot_per_dataset_activities(embedded, metadata, output_path=None, title=None):
    """
    各データセットを個別にハイライトして3つのサブプロットで可視化

    Args:
        embedded: 次元削減後の特徴量 (N, 2)
        metadata: メタデータ
        output_path: 保存先パス
        title: 全体タイトル
    """
    datasets_list = sorted(set(metadata['datasets']))
    n_datasets = len(datasets_list)

    fig, axes = plt.subplots(1, n_datasets, figsize=(8*n_datasets, 6))
    if n_datasets == 1:
        axes = [axes]

    for idx, (ax, target_dataset) in enumerate(zip(axes, datasets_list)):
        # 背景：他のデータセット（濃いグレー）
        background_mask = np.array(metadata['datasets']) != target_dataset
        if background_mask.any():
            ax.scatter(
                embedded[background_mask, 0],
                embedded[background_mask, 1],
                c='#CCCCCC',  # より濃いグレー
                alpha=0.4,
                s=8,
                edgecolors='none',
                label='Other datasets'
            )

        # ハイライト：対象データセットのアクティビティ別
        target_mask = np.array(metadata['datasets']) == target_dataset
        target_activities = [metadata['activity_names'][i] for i in range(len(target_mask)) if target_mask[i]]
        unique_activities = sorted(set(target_activities))

        # 色マップ - より鮮やかで区別しやすい色を使用
        if len(unique_activities) <= 10:
            # 10色まで: Set1カラーマップ（鮮やか）
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_activities)))
        elif len(unique_activities) <= 12:
            # 12色まで: Set3カラーマップ
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_activities)))
        elif len(unique_activities) <= 20:
            # 20色まで: tab20
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_activities)))
        else:
            # それ以上: tab20 + tab20b + tab20c を組み合わせ（最大60色）
            n = len(unique_activities)
            colors_tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
            colors_tab20b = plt.cm.tab20b(np.linspace(0, 1, 20))
            colors_tab20c = plt.cm.tab20c(np.linspace(0, 1, 20))
            all_colors = np.vstack([colors_tab20, colors_tab20b, colors_tab20c])
            # 必要な数だけ均等にサンプリング
            indices = np.linspace(0, len(all_colors)-1, n).astype(int)
            colors = all_colors[indices]
        color_map = dict(zip(unique_activities, colors))

        # アクティビティごとにプロット
        activity_centers = {}  # アクティビティの中心座標を保存
        for activity in unique_activities:
            activity_mask = target_mask & (np.array(metadata['activity_names']) == activity)
            if not activity_mask.any():
                continue

            # プロット
            ax.scatter(
                embedded[activity_mask, 0],
                embedded[activity_mask, 1],
                c=[color_map[activity]],
                label=activity,
                alpha=0.8,
                s=20,
                edgecolors='white',
                linewidths=0.3
            )

            # クラスタの中心を計算
            center_x = embedded[activity_mask, 0].mean()
            center_y = embedded[activity_mask, 1].mean()
            activity_centers[activity] = (center_x, center_y)

        # アクティビティ名をプロット上に表示
        for activity, (cx, cy) in activity_centers.items():
            ax.text(
                cx, cy, activity,
                fontsize=5,
                fontweight='bold',
                color=color_map[activity],
                ha='center',
                va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color_map[activity], alpha=0.85, linewidth=1.2)
            )

        ax.set_xlabel('Dimension 1', fontsize=10)
        ax.set_ylabel('Dimension 2', fontsize=10)
        ax.set_title(f'{target_dataset.upper()}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_per_dataset_activities_interactive(embedded, metadata, output_path=None, title=None):
    """
    各データセットを個別にハイライトしてインタラクティブHTML可視化

    Args:
        embedded: 次元削減後の特徴量 (N, 2)
        metadata: メタデータ
        output_path: 保存先パス (.html)
        title: 全体タイトル
    """
    datasets_list = sorted(set(metadata['datasets']))
    n_datasets = len(datasets_list)

    # サブプロットを作成
    fig = make_subplots(
        rows=1, cols=n_datasets,
        subplot_titles=[dataset.upper() for dataset in datasets_list],
        horizontal_spacing=0.05
    )

    for idx, target_dataset in enumerate(datasets_list):
        col = idx + 1

        # 背景：他のデータセット（薄いグレー）
        background_mask = np.array(metadata['datasets']) != target_dataset
        if background_mask.any():
            fig.add_trace(
                go.Scattergl(
                    x=embedded[background_mask, 0],
                    y=embedded[background_mask, 1],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color='#CCCCCC',
                        opacity=0.3
                    ),
                    name='Other datasets',
                    showlegend=(col == 1),
                    hoverinfo='skip',
                    legendgroup='background'
                ),
                row=1, col=col
            )

        # ハイライト：対象データセットのアクティビティ別
        target_mask = np.array(metadata['datasets']) == target_dataset
        target_activities = [metadata['activity_names'][i] for i in range(len(target_mask)) if target_mask[i]]
        unique_activities = sorted(set(target_activities))

        # 色マップ
        import matplotlib.cm as cm
        if len(unique_activities) <= 10:
            colors_array = cm.Set1(np.linspace(0, 1, len(unique_activities)))
        elif len(unique_activities) <= 12:
            colors_array = cm.Set3(np.linspace(0, 1, len(unique_activities)))
        elif len(unique_activities) <= 20:
            colors_array = cm.tab20(np.linspace(0, 1, len(unique_activities)))
        else:
            n = len(unique_activities)
            colors_tab20 = cm.tab20(np.linspace(0, 1, 20))
            colors_tab20b = cm.tab20b(np.linspace(0, 1, 20))
            colors_tab20c = cm.tab20c(np.linspace(0, 1, 20))
            colors_array = np.vstack([colors_tab20, colors_tab20b, colors_tab20c])[:n]

        # matplotlibのRGBAをPlotlyのrgb文字列に変換
        def rgba_to_rgb_string(rgba):
            r, g, b, a = [int(x * 255) for x in rgba]
            return f'rgb({r},{g},{b})'

        color_map = {activity: rgba_to_rgb_string(colors_array[i])
                     for i, activity in enumerate(unique_activities)}

        # 各アクティビティをプロット
        for activity in unique_activities:
            activity_indices = [i for i in range(len(target_mask))
                               if target_mask[i] and metadata['activity_names'][i] == activity]

            if not activity_indices:
                continue

            activity_mask_full = np.zeros(len(embedded), dtype=bool)
            activity_mask_full[activity_indices] = True

            # ホバーテキスト作成
            hover_texts = []
            for i in activity_indices:
                hover_text = (
                    f"<b>{activity}</b><br>"
                    f"Dataset: {metadata['datasets'][i]}<br>"
                    f"Location: {metadata['locations'][i]}<br>"
                    f"X: {embedded[i, 0]:.2f}<br>"
                    f"Y: {embedded[i, 1]:.2f}"
                )
                hover_texts.append(hover_text)

            fig.add_trace(
                go.Scattergl(
                    x=embedded[activity_mask_full, 0],
                    y=embedded[activity_mask_full, 1],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=color_map[activity],
                        opacity=0.7,
                        line=dict(width=0.5, color='white')
                    ),
                    name=activity,
                    showlegend=(col == 1),
                    hovertext=hover_texts,
                    hoverinfo='text',
                    legendgroup=activity
                ),
                row=1, col=col
            )

    # レイアウト設定
    fig.update_xaxes(title_text="Dimension 1", showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(title_text="Dimension 2", showgrid=True, gridcolor='lightgray')

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center') if title else None,
        height=600,
        width=400 * n_datasets,
        hovermode='closest',
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            font=dict(size=10)
        )
    )

    if output_path:
        fig.write_html(output_path)
        print(f"Saved interactive HTML: {output_path}")
    else:
        fig.show()


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
    # デフォルトモデルパス
    DEFAULT_MODEL = "experiments/pretrain/run_20251111_171703/exp_0/models/checkpoint_epoch_100.pth"

    parser = argparse.ArgumentParser(description='Visualize foundation model embeddings')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help=f'Path to pretrained model (default: {DEFAULT_MODEL})')
    parser.add_argument('--models', nargs='+', help='Paths to multiple models for comparison')
    parser.add_argument('--method', type=str, default='umap', choices=['umap', 'tsne'],
                        help='Dimensionality reduction method')
    parser.add_argument('--color-by', type=str, default='body_part',
                        choices=['dataset', 'body_part', 'dataset_location', 'activity', 'activity_per_dataset'],
                        help='Color points by')
    parser.add_argument('--datasets', nargs='+', help='Specific datasets to include')
    parser.add_argument('--locations', nargs='+', help='Specific body locations to include')
    parser.add_argument('--max-samples', type=int, default=100,
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
        # デフォルトモデルを使用
        model_paths = [DEFAULT_MODEL]

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
            encoder, window_size = load_pretrained_model(model_path, device=args.device)

            # データ収集と特徴抽出
            all_features = []
            all_datasets = []
            all_locations = []
            all_labels = []
            all_dataset_location = []

            for dataset_name, location in tqdm(dataset_location_pairs, desc=f"Processing {model_name}"):
                try:
                    X, y, _ = load_sensor_data_sample(dataset_name, location, window_size=window_size, max_samples=args.max_samples)

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

        encoder, window_size = load_pretrained_model(model_path, device=args.device)

        # データ収集と特徴抽出
        all_features = []
        all_datasets = []
        all_locations = []
        all_labels = []
        all_dataset_location = []
        all_activity_names = []

        for dataset_name, location in tqdm(dataset_location_pairs, desc="Processing datasets"):
            try:
                print(f"\n[START] Processing {dataset_name}/{location}...")
                X, y, _ = load_sensor_data_sample(dataset_name, location, window_size=window_size, max_samples=args.max_samples)
                print(f"[LOADED] {dataset_name}/{location}: {len(X)} samples")

                # 特徴抽出
                features = extract_features(encoder, X, device=args.device)

                # アクティビティ名を取得
                dataset_info = DATASETS.get(dataset_name.upper(), {})
                label_mapping = dataset_info.get('labels', {})
                activity_names = [label_mapping.get(int(label), f'unknown_{label}') for label in y]

                all_features.append(features)
                all_datasets.extend([dataset_name] * len(features))
                all_locations.extend([location] * len(features))
                all_labels.extend(y.tolist())
                all_dataset_location.extend([f"{dataset_name}/{location}"] * len(features))
                all_activity_names.extend(activity_names)

            except Exception as e:
                print(f"Error processing {dataset_name}/{location}: {e}")

        # 特徴量を結合
        all_features = np.concatenate(all_features, axis=0)

        print(f"\nTotal samples: {len(all_features)}")
        print(f"Feature dimension: {all_features.shape[1]}")

        # メタデータ
        metadata = {
            'datasets': all_datasets,
            'locations': all_locations,
            'labels': all_labels,
            'dataset_location': all_dataset_location,
            'activity_names': all_activity_names
        }

        # body part groupの定義
        body_part_groups = {
            'hand_arm': ['hand', 'Hand', 'wrist', 'Wrist', 'RightWrist', 'LeftWrist',
                        'RightArm', 'LeftArm', 'arm', 'Arm', 'RightForearm', 'LeftForearm'],
            'torso_chest': ['Chest', 'Torso', 'LowerBack', 'Back', 'Hip', 'chest', 'Neck', 'torso'],
            'leg_foot': ['ankle', 'LeftAnkle', 'RightAnkle', 'RightLeg', 'LeftLeg',
                        'Thigh', 'RightThigh', 'LeftThigh', 'RightCalf', 'LeftCalf', 'leg', 'foot']
        }

        # body part groupごとに次元削減とプロット
        model_name = Path(model_path).parent.parent.name
        time_label = {150: '5.0s', 60: '2.0s', 30: '1.0s', 15: '0.5s'}.get(window_size, f'{window_size}samples')

        for group_name, group_locations in body_part_groups.items():
            # このグループに属するデータのみをフィルタ
            indices = [i for i, loc in enumerate(metadata['locations']) if loc in group_locations]

            if len(indices) == 0:
                print(f"\nNo data found for {group_name}, skipping...")
                continue

            # フィルタされた特徴量とメタデータ
            filtered_features = all_features[indices]
            filtered_metadata = {
                'datasets': [metadata['datasets'][i] for i in indices],
                'locations': [metadata['locations'][i] for i in indices],
                'labels': [metadata['labels'][i] for i in indices],
                'dataset_location': [metadata['dataset_location'][i] for i in indices],
                'activity_names': [metadata['activity_names'][i] for i in indices]
            }

            print(f"\n[{group_name}] Processing {len(indices)} samples from {len(set(filtered_metadata['datasets']))} datasets")

            # このグループだけで次元削減（高速化）
            print(f"[{group_name}] Reducing dimensions using {args.method.upper()}...")
            filtered_embedded = reduce_dimensions(filtered_features, method=args.method)

            title = f"Foundation Model Embeddings - {group_name.replace('_', '/')} - {time_label}"
            output_path_png = output_dir / f"embeddings_{group_name}_{time_label}_{args.method}_{args.color_by}.png"
            output_path_html = output_dir / f"embeddings_{group_name}_{time_label}_{args.method}_{args.color_by}.html"

            print(f"[{group_name}] Generating plots...")

            # PNG版
            if args.color_by == 'activity_per_dataset':
                plot_per_dataset_activities(filtered_embedded, filtered_metadata, output_path=output_path_png, title=title)
            else:
                plot_embeddings(filtered_embedded, filtered_metadata, color_by=args.color_by,
                               output_path=output_path_png, title=title)

            print(f"[{group_name}] Saved PNG: {output_path_png}")

            # インタラクティブHTML版
            if args.color_by == 'activity_per_dataset':
                plot_per_dataset_activities_interactive(filtered_embedded, filtered_metadata, output_path=output_path_html, title=title)
                print(f"[{group_name}] Saved HTML: {output_path_html}")

    print("\nDone!")


def main_with_args(args):
    """引数オブジェクトを受け取って実行（analyze.pyから呼ばれる用）"""
    # main()の実装をそのまま使用
    main()


if __name__ == '__main__':
    main()
