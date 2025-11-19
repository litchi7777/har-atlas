"""
Simple Model Embedding Explorer

モデル別の特徴空間をt-SNEで可視化

使用方法:
    python analysis/embedding_explorer/simple_server.py --port 5000

アクセス:
    http://localhost:5000
"""

import os
import sys
from pathlib import Path
import json
import argparse
import numpy as np
from flask import Flask, render_template, jsonify, request
import plotly.graph_objects as go
import plotly.utils
from sklearn.manifold import TSNE

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

app = Flask(__name__)

# データディレクトリ
DATA_DIR = Path(__file__).parent / "data"

# グローバルキャッシュ
cached_data = {}


def get_available_models():
    """利用可能なモデル一覧を取得"""
    models = []
    for metadata_file in DATA_DIR.glob("metadata_*.json"):
        model_name = metadata_file.stem.replace("metadata_", "")
        models.append(model_name)
    return sorted(models)


def load_model_data(model_name):
    """モデルの特徴量とメタデータを読み込み"""
    if model_name in cached_data:
        return cached_data[model_name]

    features_path = DATA_DIR / f"features_{model_name}.npz"
    metadata_path = DATA_DIR / f"metadata_{model_name}.json"

    if not features_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Data for model '{model_name}' not found")

    # 特徴量読み込み
    features = np.load(features_path)['features']

    # メタデータ読み込み
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    data = {
        'features': features,
        'metadata': metadata
    }

    cached_data[model_name] = data
    return data


def get_unique_values(metadata):
    """メタデータからユニークな値を取得"""
    return {
        'datasets': sorted(list(set(metadata['datasets']))),
        'locations': sorted(list(set(metadata['locations']))),
        'activities': sorted(list(set(metadata['activity_names']))),
        'body_parts': sorted(list(set(metadata['body_parts'])))
    }


def compute_tsne(features, perplexity=30, max_iter=1000):
    """t-SNEで2次元に削減"""
    print(f"Computing t-SNE on {features.shape[0]} samples...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, features.shape[0] - 1),
        max_iter=max_iter,
        random_state=42
    )
    embedded = tsne.fit_transform(features)
    return embedded


def create_activity_plot(embedded, metadata, dataset_activity_colors):
    """Dataset × Activity 別の可視化を作成"""
    fig = go.Figure()

    # Dataset × Activity の組み合わせを作成
    dataset_activity_pairs = []
    for i in range(len(metadata['datasets'])):
        pair = f"{metadata['datasets'][i]}_{metadata['activity_names'][i]}"
        dataset_activity_pairs.append(pair)

    unique_pairs = sorted(list(set(dataset_activity_pairs)))

    for pair in unique_pairs:
        # このペアのインデックス
        indices = [i for i, p in enumerate(dataset_activity_pairs) if p == pair]

        if len(indices) == 0:
            continue

        # 色
        color = dataset_activity_colors.get(pair, '#808080')

        # dataset と activity を分離
        dataset, activity = pair.split('_', 1)

        # ホバーテキスト
        hover_texts = [
            f"Dataset: {metadata['datasets'][i]}<br>"
            f"Location: {metadata['locations'][i]}<br>"
            f"Activity: {metadata['activity_names'][i]}<br>"
            f"Body Part: {metadata['body_parts'][i]}"
            for i in indices
        ]

        fig.add_trace(go.Scatter(
            x=embedded[indices, 0],
            y=embedded[indices, 1],
            mode='markers',
            name=f"{dataset}: {activity}",
            marker=dict(
                size=6,
                color=color,
                line=dict(width=0.5, color='white'),
                opacity=0.7
            ),
            text=hover_texts,
            hoverinfo='text'
        ))

    fig.update_layout(
        title="Feature Space Visualization (t-SNE, colored by Dataset × Activity)",
        xaxis_title="t-SNE Component 1",
        yaxis_title="t-SNE Component 2",
        hovermode='closest',
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        )
    )

    return fig


@app.route('/')
def index():
    """メインページ"""
    return render_template('simple_index.html')


@app.route('/api/models')
def get_models():
    """利用可能なモデル一覧を返す"""
    try:
        models = get_available_models()
        return jsonify({
            'success': True,
            'models': models
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/model_info/<model_name>')
def get_model_info(model_name):
    """モデルの情報を返す"""
    try:
        data = load_model_data(model_name)
        unique_values = get_unique_values(data['metadata'])

        return jsonify({
            'success': True,
            'model_name': model_name,
            'num_samples': len(data['features']),
            'feature_dim': data['features'].shape[1],
            'unique_values': unique_values
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/visualize', methods=['POST'])
def visualize():
    """可視化を生成"""
    try:
        params = request.json
        model_name = params.get('model_name')
        selected_datasets = params.get('datasets', [])
        selected_activities = params.get('activities', [])
        selected_locations = params.get('locations', [])
        perplexity = params.get('perplexity', 30)

        # データ読み込み
        data = load_model_data(model_name)
        features = data['features']
        metadata = data['metadata']

        # フィルタリング
        indices = list(range(len(features)))

        if selected_datasets:
            indices = [i for i in indices if metadata['datasets'][i] in selected_datasets]

        if selected_activities:
            indices = [i for i in indices if metadata['activity_names'][i] in selected_activities]

        if selected_locations:
            indices = [i for i in indices if metadata['locations'][i] in selected_locations]

        if len(indices) == 0:
            return jsonify({
                'success': False,
                'error': 'No data matches the selected filters'
            }), 400

        # フィルタされた特徴量とメタデータ
        filtered_features = features[indices]
        filtered_metadata = {
            'datasets': [metadata['datasets'][i] for i in indices],
            'locations': [metadata['locations'][i] for i in indices],
            'activity_names': [metadata['activity_names'][i] for i in indices],
            'body_parts': [metadata['body_parts'][i] for i in indices]
        }

        # t-SNE
        embedded = compute_tsne(filtered_features, perplexity=perplexity)

        # Dataset × Activity 別の色を生成
        dataset_activity_pairs = [
            f"{filtered_metadata['datasets'][i]}_{filtered_metadata['activity_names'][i]}"
            for i in range(len(filtered_metadata['datasets']))
        ]
        unique_pairs = sorted(list(set(dataset_activity_pairs)))

        import plotly.express as px
        colors = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.Set1
        dataset_activity_colors = {pair: colors[i % len(colors)] for i, pair in enumerate(unique_pairs)}

        # プロット作成
        fig = create_activity_plot(embedded, filtered_metadata, dataset_activity_colors)

        # JSONに変換
        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return jsonify({
            'success': True,
            'plot': plot_json,
            'num_samples': len(filtered_features)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def main():
    parser = argparse.ArgumentParser(description='Simple Model Embedding Explorer')
    parser.add_argument('--port', type=int, default=5000,
                        help='ポート番号')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='ホスト')
    parser.add_argument('--debug', action='store_true',
                        help='デバッグモード')

    args = parser.parse_args()

    print("="*80)
    print("Simple Model Embedding Explorer")
    print("="*80)
    print(f"\nServer starting on http://{args.host}:{args.port}")
    print(f"Data directory: {DATA_DIR}")

    # 利用可能なモデルを表示
    try:
        models = get_available_models()
        print(f"\nAvailable models: {models}")
    except Exception as e:
        print(f"\nWarning: Could not load models: {e}")

    print("\nPress Ctrl+C to stop the server")
    print("="*80)

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
