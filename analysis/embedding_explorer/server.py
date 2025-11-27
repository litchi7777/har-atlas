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

    # HHARデバイス名（スマートウォッチとスマートフォン）
    # gear, lgwatch = Samsung Gear / LG Watch（スマートウォッチ）→ Wrist
    if any(kw in location_lower for kw in ['gear_', 'lgwatch_']):
        return 'Wrist'
    # nexus4, s3, s3mini = スマートフォン → Phone
    if any(kw in location_lower for kw in ['nexus4_', 's3_', 's3mini_']):
        return 'Phone'

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


def clear_cache(model_name=None):
    """キャッシュをクリア"""
    global cached_data
    if model_name:
        if model_name in cached_data:
            del cached_data[model_name]
            print(f"Cleared cache for {model_name}")
    else:
        cached_data = {}
        print("Cleared all cache")


def get_available_models():
    """利用可能なモデル一覧を取得"""
    data_dir = Path(__file__).parent / "data"
    models = []

    # metadata_*.json ファイルを探す
    for metadata_file in data_dir.glob("metadata_*.json"):
        model_name = metadata_file.stem.replace("metadata_", "")
        # 旧形式（window_size_label）もサポート
        models.append(model_name)

    return sorted(models)


def load_features(model_name='5.0s'):
    """
    特徴ベクトルとメタデータを読み込む

    Args:
        model_name: モデル名 or window_size_label ('rotation', '5.0s', '2.0s', etc.)

    Returns:
        features: 特徴ベクトル (N, feature_dim)
        metadata: メタデータ辞書
        sensor_data: 生センサーデータ (N, 3, window_size) or None
        tsne_embeddings: 事前計算されたt-SNE埋め込み (N, 2) or None
        prototype_data: プロトタイプ情報 (embeddings, metadata) or None
    """
    data_dir = Path(__file__).parent / "data"

    # キャッシュチェック
    if model_name in cached_data:
        print(f"Using cached data for {model_name}")
        return cached_data[model_name]

    # NPZファイル読み込み
    features_path = data_dir / f"features_{model_name}.npz"
    metadata_path = data_dir / f"metadata_{model_name}.json"

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    print(f"Loading features from {features_path}")
    data = np.load(features_path)
    features = data['features']

    # 生データが存在すれば読み込む
    sensor_data = data.get('sensor_data', None)
    if sensor_data is not None:
        print(f"  Loaded sensor data: {sensor_data.shape}")

    # t-SNE埋め込みが存在すれば読み込む
    tsne_embeddings = data.get('tsne_embeddings', None)
    if tsne_embeddings is not None:
        print(f"  Loaded precomputed t-SNE embeddings: {tsne_embeddings.shape}")

    # プロトタイプが存在すれば読み込む
    prototype_embeddings = data.get('prototype_embeddings', None)
    prototype_features = data.get('prototype_features', None)

    print(f"Loading metadata from {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # プロトタイプメタデータ
    prototype_data = None
    if prototype_embeddings is not None and 'prototypes' in metadata:
        prototype_data = {
            'embeddings': prototype_embeddings,
            'features': prototype_features,
            'body_parts': metadata['prototypes']['body_parts'],
            'prototype_ids': metadata['prototypes']['prototype_ids'],
            'atomic_motion_names': metadata['prototypes'].get('atomic_motion_names', metadata['prototypes']['prototype_ids'])
        }
        print(f"  Loaded prototypes: {len(prototype_embeddings)} prototypes")

    # Locationをカテゴリ化（古いデータ互換性のため）
    if 'locations' in metadata:
        metadata['locations'] = [map_location_to_category(loc) for loc in metadata['locations']]
        print(f"  Categorized locations: {len(set(metadata['locations']))} unique categories")

    # body_partsがない場合は追加
    if 'body_parts' not in metadata and 'locations' in metadata:
        metadata['body_parts'] = metadata['locations']

    # キャッシュに保存（プロトタイプも含む）
    cached_data[model_name] = (features, metadata, sensor_data, tsne_embeddings, prototype_data)

    return features, metadata, sensor_data, tsne_embeddings, prototype_data


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


def create_plotly_figure(embedded, metadata, color_by='dataset',
                         selected_datasets=None, selected_activities=None, selected_locations=None,
                         prototype_data=None, show_prototypes=True):
    """
    Plotly図を作成（全データ表示、選択されたものをハイライト）

    Args:
        embedded: 次元削減後の特徴 (N, 2)
        metadata: メタデータ辞書
        color_by: 色分け基準 ('dataset', 'activity', 'location')
        selected_datasets: 選択されたデータセットのリスト
        selected_activities: 選択されたアクティビティのリスト
        selected_locations: 選択されたlocationのリスト
        prototype_data: プロトタイプデータ (embeddings, body_parts, prototype_ids) or None
        show_prototypes: プロトタイプを表示するかどうか

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
    elif color_by == 'dataset_activity':
        # Dataset × Activity × Location の組み合わせ
        categories = [f"{dataset}_{activity}_{location}"
                     for dataset, activity, location in zip(metadata['datasets'],
                                                             metadata['activity_names'],
                                                             metadata['locations'])]
        unique_categories = sorted(set(categories))
        legend_title = 'Dataset: Activity (Location)'
        selected_items = []  # フィルターは別途適用
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

    # アクティビティフィルター（dataset_activity ペアで処理）
    if selected_activities:
        # selected_activities は ["dsads_Walking", "mhealth_Standing", ...] の形式
        # 各サンプルの dataset_activity を構築
        sample_dataset_activities = [
            f"{dataset}_{activity}"
            for dataset, activity in zip(metadata['datasets'], metadata['activity_names'])
        ]
        activity_mask = np.isin(sample_dataset_activities, selected_activities)
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
    legendgroup_seen = set()  # 各legendgroupで最初のトレースか判定

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

            # dataset_activityの場合は凡例をグループ化
            # Dataset_Location を接頭辞として、Activity を個別トレースに（legendgroupなし）
            if color_by == 'dataset_activity' and '_' in category:
                parts = category.split('_')
                if len(parts) >= 3:
                    dataset = parts[0]
                    activity = '_'.join(parts[1:-1])  # 中間部分全てをactivityとして扱う
                    location = parts[-1]
                    # 接頭辞として表示（legendgroupは使わない = 個別トグル可能）
                    trace_name = f"[{dataset}_{location}] {activity}"
                else:
                    # fallback
                    trace_name = category
            else:
                trace_name = category

            trace_dict = dict(
                x=embedded[highlighted_indices, 0].tolist(),
                y=embedded[highlighted_indices, 1].tolist(),
                mode='markers',
                marker=dict(
                    size=5,
                    color=color_map[category],
                    opacity=0.7,
                    line=dict(width=0.5, color='white')
                ),
                name=trace_name,
                hovertext=hover_texts,
                hoverinfo='text',
                customdata=highlighted_indices.tolist()  # グローバルインデックスを保存
            )

            # legendgroupは使用しない（個別トグルを可能にするため）

            fig.add_trace(go.Scattergl(**trace_dict))

    print(f"\n[DEBUG] Trace creation complete:")
    print(f"  Total highlighted: {total_highlighted}")
    print(f"  Total traces added: {len(fig.data)}")

    # プロトタイプを追加（存在する場合）
    if show_prototypes and prototype_data is not None:
        proto_embeddings = prototype_data['embeddings']
        proto_body_parts = prototype_data['body_parts']
        proto_ids = prototype_data['prototype_ids']
        proto_names = prototype_data.get('atomic_motion_names', proto_ids)

        print(f"\n[DEBUG] Adding {len(proto_embeddings)} prototypes")

        # Body part ごとの色（星マーカー用）
        proto_colors = {
            'wrist': '#e74c3c',   # 赤
            'leg': '#2ecc71',     # 緑
            'hip': '#3498db',     # 青
            'chest': '#f39c12',   # オレンジ
            'head': '#9b59b6',    # 紫
        }

        # Body partごとにプロトタイプを追加
        unique_body_parts = sorted(set(proto_body_parts))
        for bp in unique_body_parts:
            bp_mask = np.array(proto_body_parts) == bp
            bp_indices = np.where(bp_mask)[0]

            if len(bp_indices) == 0:
                continue

            bp_embeddings = proto_embeddings[bp_indices]
            bp_names = [proto_names[i] for i in bp_indices]

            hover_texts = [
                f"<b>🌟 {name}</b><br>"
                f"Body Part: {bp}<br>"
                f"X: {bp_embeddings[i, 0]:.2f}<br>"
                f"Y: {bp_embeddings[i, 1]:.2f}"
                for i, name in enumerate(bp_names)
            ]

            fig.add_trace(go.Scattergl(
                x=bp_embeddings[:, 0].tolist(),
                y=bp_embeddings[:, 1].tolist(),
                mode='markers',
                marker=dict(
                    size=15,
                    color=proto_colors.get(bp, '#95a5a6'),
                    symbol='star',
                    line=dict(width=2, color='white')
                ),
                name=f'Proto: {bp}',
                hovertext=hover_texts,
                hoverinfo='text',
                legendgroup='prototypes',
                legendgrouptitle_text='Prototypes'
            ))

        print(f"  Added {len(unique_body_parts)} prototype groups")

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
            font=dict(size=9),
            tracegroupgap=10,  # グループ間のスペース
            itemsizing='constant',  # アイコンサイズを一定に
            itemwidth=30  # アイテム幅
        )
    )

    return fig


@app.route('/')
def index():
    """メインページ"""
    return render_template('index.html')


@app.route('/api/models')
def get_models_endpoint():
    """利用可能なモデル一覧を返す"""
    try:
        models = get_available_models()
        return jsonify({
            'success': True,
            'models': models
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/metadata/<window_size>')
def get_metadata(window_size):
    """メタデータを取得"""
    try:
        print(f"[get_metadata] Loading metadata for window_size: {window_size}")
        _, metadata, _, _, prototype_data = load_features(window_size)

        print(f"[get_metadata] Metadata keys: {list(metadata.keys())}")
        print(f"[get_metadata] Total datasets entries: {len(metadata.get('datasets', []))}")

        # 必須フィールドの確認
        if 'datasets' not in metadata:
            raise ValueError("Missing 'datasets' field in metadata")
        if 'activity_names' not in metadata:
            raise ValueError("Missing 'activity_names' field in metadata")
        if 'locations' not in metadata:
            raise ValueError("Missing 'locations' field in metadata")

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

        response_data = {
            'datasets': unique_datasets,
            'activities_by_dataset': activities_by_dataset,
            'locations': unique_locations,
            'total_samples': len(metadata['datasets']),  # メタデータから動的に計算
            'has_prototypes': prototype_data is not None
        }

        print(f"[get_metadata] Response: {len(unique_datasets)} datasets, {len(unique_locations)} locations, {len(metadata['datasets'])} total samples")

        return jsonify(response_data)
    except Exception as e:
        import traceback
        print(f"[get_metadata] ERROR: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/visualize', methods=['POST'])
def visualize():
    """可視化を生成（全データ表示、選択されたものをハイライト）"""
    try:
        params = request.json

        model_name = params.get('model_name', params.get('window_size', '5.0s'))  # 互換性のため
        method = params.get('method', 'tsne')  # デフォルトをt-SNEに変更
        color_by = params.get('color_by', 'dataset_activity')  # デフォルトをdataset_activityに変更
        selected_datasets = params.get('selected_datasets', None)
        selected_activities = params.get('selected_activities', None)
        selected_locations = params.get('selected_locations', None)
        show_prototypes = params.get('show_prototypes', True)  # プロトタイプ表示オプション

        # データ読み込み（全データ）
        features, metadata, sensor_data, tsne_embeddings, prototype_data = load_features(model_name)

        print(f"Total samples: {len(features)}")

        # 次元削減（t-SNEは事前計算を使用、PCAはオンザフライ）
        if method == 'tsne':
            if tsne_embeddings is not None:
                print("Using precomputed t-SNE embeddings")
                embedded = tsne_embeddings
            else:
                print("Computing t-SNE on-the-fly (precomputed not available)...")
                reducer = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
                embedded = reducer.fit_transform(features)
        elif method == 'pca':
            print("Computing PCA...")
            reducer = PCA(n_components=2)
            embedded = reducer.fit_transform(features)
        else:
            return jsonify({'error': f'Unknown method: {method}. Only tsne and pca are supported.'}), 400

        # プロット作成（全データ、選択されたものをハイライト）
        fig = create_plotly_figure(
            embedded, metadata,
            color_by=color_by,
            selected_datasets=selected_datasets,
            selected_activities=selected_activities,
            selected_locations=selected_locations,
            prototype_data=prototype_data,
            show_prototypes=show_prototypes
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


@app.route('/api/sensor_data', methods=['POST'])
def get_sensor_data():
    """
    クリックされたポイントの生センサーデータを取得
    """
    try:
        params = request.json
        dataset = params.get('dataset')
        location = params.get('location')
        activity = params.get('activity')
        point_index = params.get('point_index', 0)  # 全データ内でのインデックス
        model_name = params.get('model_name')

        print(f"\n[API /sensor_data] Request:")
        print(f"  Dataset: {dataset}, Location: {location}, Activity: {activity}")
        print(f"  Point index: {point_index}")

        # キャッシュからデータ取得
        features, metadata, sensor_data, _, _ = load_features(model_name)

        if sensor_data is None:
            return jsonify({'error': 'Sensor data not available. Please re-run extract_model_features.py with updated version.'}), 404

        # point_indexを直接使用
        if point_index >= len(sensor_data):
            return jsonify({'error': f'Invalid point index: {point_index}'}), 400

        raw_sensor = sensor_data[point_index]  # shape: (3, window_size)

        print(f"  Retrieved sensor data at index {point_index}, shape: {raw_sensor.shape}")

        return jsonify({
            'sensor_data': raw_sensor.tolist(),  # (3, window_size)
            'dataset': dataset,
            'location': location,
            'activity': activity,
            'shape': list(raw_sensor.shape)
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
