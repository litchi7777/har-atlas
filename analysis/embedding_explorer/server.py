"""
Embedding Explorer Web Server

インタラクティブな特徴空間可視化サーバー

使用方法:
    python analysis/embedding_explorer/server.py --port 5000

アクセス:
    http://localhost:5000
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
from flask import Flask, render_template, jsonify, request
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

app = Flask(__name__)


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

# グローバル変数でデータをキャッシュ
cached_data = {}


def load_features(window_size_label='5.0s'):
    """
    特徴ベクトルとメタデータを読み込む

    Args:
        window_size_label: ウィンドウサイズラベル ('5.0s', '2.0s', '1.0s', '0.5s')

    Returns:
        features: 特徴ベクトル (N, feature_dim)
        metadata: メタデータ辞書
        umap_embeddings: 事前計算されたUMAP埋め込み (N, 2) or None
    """
    data_dir = Path(__file__).parent / "data"

    # キャッシュチェック
    if window_size_label in cached_data:
        print(f"Using cached data for {window_size_label}")
        return cached_data[window_size_label]

    # NPZファイル読み込み
    features_path = data_dir / f"features_{window_size_label}.npz"
    metadata_path = data_dir / f"metadata_{window_size_label}.json"

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    print(f"Loading features from {features_path}")
    data = np.load(features_path)
    features = data['features']

    # UMAP埋め込みが存在すれば読み込む
    umap_embeddings = data.get('umap_embeddings', None)
    if umap_embeddings is not None:
        print(f"  Loaded precomputed UMAP embeddings: {umap_embeddings.shape}")

    print(f"Loading metadata from {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Locationをカテゴリ化（古いデータ互換性のため）
    metadata['locations'] = [map_location_to_category(loc) for loc in metadata['locations']]
    print(f"  Categorized locations: {len(set(metadata['locations']))} unique categories")

    # キャッシュに保存
    cached_data[window_size_label] = (features, metadata, umap_embeddings)

    return features, metadata, umap_embeddings


def apply_filters(features, metadata, selected_datasets=None, selected_activities=None, selected_locations=None):
    """
    フィルターを適用してデータをサブセット化

    Args:
        features: 特徴ベクトル (N, feature_dim)
        metadata: メタデータ辞書
        selected_datasets: 選択されたデータセットのリスト
        selected_activities: 選択されたアクティビティのリスト
        selected_locations: 選択されたlocationのリスト

    Returns:
        filtered_features: フィルタ後の特徴ベクトル
        filtered_metadata: フィルタ後のメタデータ
        filter_indices: フィルター適用後のインデックス
    """
    # 全インデックスから開始
    mask = np.ones(len(features), dtype=bool)

    # データセットフィルター
    if selected_datasets:
        dataset_mask = np.isin(metadata['datasets'], selected_datasets)
        mask &= dataset_mask

    # アクティビティフィルター
    if selected_activities:
        activity_mask = np.isin(metadata['activity_names'], selected_activities)
        mask &= activity_mask

    # Locationフィルター
    if selected_locations:
        location_mask = np.isin(metadata['locations'], selected_locations)
        mask &= location_mask

    # マスクを適用してインデックスを取得
    indices = np.where(mask)[0]

    # フィルタ後のデータ
    filtered_features = features[indices]
    filtered_metadata = {
        'datasets': [metadata['datasets'][i] for i in indices],
        'locations': [metadata['locations'][i] for i in indices],
        'labels': [metadata['labels'][i] for i in indices],
        'dataset_location': [metadata['dataset_location'][i] for i in indices],
        'activity_names': [metadata['activity_names'][i] for i in indices]
    }

    return filtered_features, filtered_metadata, indices


def reduce_dimensions(features, method='umap', n_components=2, precomputed_umap=None, filter_indices=None):
    """
    次元削減を実行

    Args:
        features: 特徴ベクトル (N, feature_dim)
        method: 次元削減手法 ('umap', 'tsne', 'pca')
        n_components: 削減後の次元数
        precomputed_umap: 事前計算されたUMAP埋め込み (N, 2) - methodが'umap'の場合に使用
        filter_indices: フィルター適用後のインデックス

    Returns:
        embedded: 次元削減後の特徴 (N, n_components)
    """
    print(f"Reducing dimensions using {method.upper()}...")

    if method == 'umap' and precomputed_umap is not None:
        # 事前計算されたUMAP埋め込みを使用
        print("  Using precomputed UMAP embeddings")
        if filter_indices is not None:
            embedded = precomputed_umap[filter_indices]
        else:
            embedded = precomputed_umap
    elif method == 'umap':
        # UMAPを新規計算（事前計算がない場合のフォールバック）
        print("  Computing UMAP on-the-fly...")
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        embedded = reducer.fit_transform(features)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
        embedded = reducer.fit_transform(features)
    elif method == 'pca':
        reducer = PCA(n_components=n_components)
        embedded = reducer.fit_transform(features)
    else:
        raise ValueError(f"Unknown method: {method}")

    return embedded


def create_plotly_figure(embedded, metadata, color_by='dataset',
                         selected_datasets=None, selected_activities=None, selected_locations=None):
    """
    Plotly図を作成（全データ表示、選択されたものをハイライト）

    Args:
        embedded: 次元削減後の特徴 (N, 2)
        metadata: メタデータ辞書
        color_by: 色分け基準 ('dataset', 'activity', 'location')
        selected_datasets: 選択されたデータセットのリスト
        selected_activities: 選択されたアクティビティのリスト
        selected_locations: 選択されたlocationのリスト

    Returns:
        fig: Plotly Figure オブジェクト
    """
    print(f"\n[DEBUG create_plotly_figure]")
    print(f"  embedded.shape: {embedded.shape}")
    print(f"  color_by: {color_by}")
    print(f"  selected_datasets: {selected_datasets}")
    print(f"  selected_activities: {selected_activities[:5] if selected_activities and len(selected_activities) > 5 else selected_activities}")
    print(f"  selected_locations: {selected_locations}")

    # 色分け基準に応じてカテゴリを取得
    if color_by == 'dataset':
        categories = metadata['datasets']
        unique_categories = sorted(set(categories))
        legend_title = 'Dataset'
        selected_items = selected_datasets or []
    elif color_by == 'activity':
        # アクティビティはデータセット・ロケーションとペアで扱う（dataset/activity/location形式）
        categories = [f"{dataset}/{activity}/{location}"
                     for dataset, activity, location in zip(metadata['datasets'],
                                                            metadata['activity_names'],
                                                            metadata['locations'])]
        unique_categories = sorted(set(categories))
        legend_title = 'Dataset/Activity/Location'
        selected_items = selected_activities or []
    elif color_by == 'location':
        categories = metadata['locations']
        unique_categories = sorted(set(categories))
        legend_title = 'Location'
        selected_items = selected_locations or []
    else:
        raise ValueError(f"Unknown color_by: {color_by}")

    # 各サンプルがハイライト対象かどうかを判定
    highlight_mask = np.ones(len(embedded), dtype=bool)

    # データセットフィルター
    if selected_datasets:
        dataset_mask = np.isin(metadata['datasets'], selected_datasets)
        highlight_mask &= dataset_mask

    # アクティビティフィルター
    if selected_activities:
        activity_mask = np.isin(metadata['activity_names'], selected_activities)
        highlight_mask &= activity_mask

    # Locationフィルター
    if selected_locations:
        location_mask = np.isin(metadata['locations'], selected_locations)
        highlight_mask &= location_mask

    # カラーマップ作成
    import matplotlib.cm as cm
    n_colors = len(unique_categories)
    if n_colors <= 10:
        colors_array = cm.Set1(np.linspace(0, 1, n_colors))
    elif n_colors <= 20:
        colors_array = cm.tab20(np.linspace(0, 1, n_colors))
    else:
        colors_array = cm.tab20(np.linspace(0, 1, 20))
        colors_array = np.tile(colors_array, (n_colors // 20 + 1, 1))[:n_colors]

    def rgba_to_rgb_string(rgba):
        r, g, b, a = [int(x * 255) for x in rgba]
        return f'rgb({r},{g},{b})'

    color_map = {cat: rgba_to_rgb_string(colors_array[i])
                 for i, cat in enumerate(unique_categories)}

    # Figure作成
    fig = go.Figure()

    sys.stderr.write(f"\n[DEBUG] Creating traces for {len(unique_categories)} categories\n")
    sys.stderr.write(f"  len(categories): {len(categories)}\n")
    sys.stderr.write(f"  len(embedded): {len(embedded)}\n")
    sys.stderr.write(f"  len(highlight_mask): {len(highlight_mask)}\n")
    sys.stderr.write(f"  Unique categories: {unique_categories[:10]}{'...' if len(unique_categories) > 10 else ''}\n")
    sys.stderr.write(f"  Total samples with highlight: {np.sum(highlight_mask)}\n")
    sys.stderr.flush()

    # カテゴリごとにトレースを追加
    total_highlighted = 0
    for idx, category in enumerate(unique_categories):
        category_mask = np.array(categories) == category

        # ハイライト対象のインデックス
        highlighted_indices = np.where(category_mask & highlight_mask)[0]

        if idx < 3:  # First 3 categories for debugging
            sys.stderr.write(f"  Category '{category}':\n")
            sys.stderr.write(f"    category_mask sum: {np.sum(category_mask)}\n")
            sys.stderr.write(f"    highlighted_indices: {len(highlighted_indices)}\n")
            sys.stderr.flush()
        elif len(highlighted_indices) > 0:
            sys.stderr.write(f"  Category '{category}': {len(highlighted_indices)} highlighted\n")
            sys.stderr.flush()
        total_highlighted += len(highlighted_indices)

        # 選択されたデータのみ追加
        if len(highlighted_indices) > 0:
            hover_texts = []
            for i in highlighted_indices:
                hover_text = (
                    f"<b>{category}</b><br>"
                    f"Dataset: {metadata['datasets'][i]}<br>"
                    f"Activity: {metadata['activity_names'][i]}<br>"
                    f"Location: {metadata['locations'][i]}<br>"
                    f"X: {embedded[i, 0]:.2f}<br>"
                    f"Y: {embedded[i, 1]:.2f}"
                )
                hover_texts.append(hover_text)

            fig.add_trace(
                go.Scattergl(
                    x=embedded[highlighted_indices, 0].tolist(),
                    y=embedded[highlighted_indices, 1].tolist(),
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=color_map[category],
                        opacity=0.7,
                        line=dict(width=0.5, color='white')
                    ),
                    name=category,
                    hovertext=hover_texts,
                    hoverinfo='text'
                )
            )

    print(f"\n[DEBUG] Trace creation complete:")
    print(f"  Total highlighted: {total_highlighted}")
    print(f"  Total traces added: {len(fig.data)}")

    # レイアウト設定
    fig.update_layout(
        title='Embedding Space Visualization',
        xaxis=dict(title='Dimension 1', showgrid=True, gridcolor='lightgray'),
        yaxis=dict(title='Dimension 2', showgrid=True, gridcolor='lightgray'),
        height=700,
        hovermode='closest',
        template='plotly_white',
        legend=dict(
            title=legend_title,
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            font=dict(size=10)
        )
    )

    return fig


@app.route('/')
def index():
    """メインページ"""
    return render_template('index.html')


@app.route('/api/metadata/<window_size>')
def get_metadata(window_size):
    """メタデータを取得"""
    try:
        _, metadata, _ = load_features(window_size)

        # ユニークな値を取得
        unique_datasets = sorted(set(metadata['datasets']))
        unique_locations = sorted(set(metadata['locations']))

        # アクティビティをデータセット別に整理
        activities_by_dataset = {}
        for dataset, activity in zip(metadata['datasets'], metadata['activity_names']):
            if dataset not in activities_by_dataset:
                activities_by_dataset[dataset] = set()
            activities_by_dataset[dataset].add(activity)

        # セットをソート済みリストに変換
        activities_by_dataset = {
            dataset: sorted(list(activities))
            for dataset, activities in activities_by_dataset.items()
        }

        return jsonify({
            'datasets': unique_datasets,
            'activities_by_dataset': activities_by_dataset,
            'locations': unique_locations,
            'total_samples': metadata['total_samples']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/visualize', methods=['POST'])
def visualize():
    """可視化を生成（全データ表示、選択されたものをハイライト）"""
    try:
        params = request.json

        window_size = params.get('window_size', '5.0s')
        method = params.get('method', 'umap')
        color_by = params.get('color_by', 'dataset')
        selected_datasets = params.get('selected_datasets', None)
        selected_activities = params.get('selected_activities', None)
        selected_locations = params.get('selected_locations', None)

        # データ読み込み（全データ）
        features, metadata, umap_embeddings = load_features(window_size)

        print(f"Total samples: {len(features)}")

        # 次元削減（全データに対して実行）
        if method == 'umap' and umap_embeddings is not None:
            # 事前計算されたUMAP埋め込みを使用
            print("Using precomputed UMAP embeddings")
            embedded = umap_embeddings
        elif method == 'umap':
            print("Computing UMAP on-the-fly...")
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedded = reducer.fit_transform(features)
        elif method == 'tsne':
            print("Computing t-SNE...")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            embedded = reducer.fit_transform(features)
        elif method == 'pca':
            print("Computing PCA...")
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            embedded = reducer.fit_transform(features)
        else:
            return jsonify({'error': f'Unknown method: {method}'}), 400

        # プロット作成（全データ、選択されたものをハイライト）
        fig = create_plotly_figure(
            embedded, metadata,
            color_by=color_by,
            selected_datasets=selected_datasets,
            selected_activities=selected_activities,
            selected_locations=selected_locations
        )

        # ハイライトされたサンプル数を計算
        highlight_mask = np.ones(len(features), dtype=bool)
        print(f"Selected filters:")
        print(f"  Datasets: {selected_datasets}")
        print(f"  Activities: {selected_activities[:5] if selected_activities else None}...")
        print(f"  Locations: {selected_locations}")

        if selected_datasets:
            highlight_mask &= np.isin(metadata['datasets'], selected_datasets)
            print(f"  After dataset filter: {np.sum(highlight_mask)} samples")
        if selected_activities:
            highlight_mask &= np.isin(metadata['activity_names'], selected_activities)
            print(f"  After activity filter: {np.sum(highlight_mask)} samples")
        if selected_locations:
            highlight_mask &= np.isin(metadata['locations'], selected_locations)
            print(f"  After location filter: {np.sum(highlight_mask)} samples")
        n_highlighted = np.sum(highlight_mask)
        print(f"Final highlighted samples: {n_highlighted}")

        # 辞書に変換（to_dict()を使用してデータの損失を防ぐ）
        graph_dict = fig.to_dict()

        return jsonify({
            'graph': graph_dict,
            'n_samples': len(features),
            'n_highlighted': int(n_highlighted)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Embedding Explorer Server')
    parser.add_argument('--port', type=int, default=8050,
                        help='Port to run server on')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run server on')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Embedding Explorer Server")
    print(f"{'='*60}")
    print(f"Access at: http://localhost:{args.port}")
    print(f"Hot reload: {'Enabled' if args.debug else 'Disabled'}")
    print(f"{'='*60}\n")

    # ホットリロードを有効化（debugモードで自動的に有効）
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=args.debug)
