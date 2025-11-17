"""
Foundation Modelの特徴表現を可視化（リファクタリング版）

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

出力:
    analysis/figures/ 以下に可視化結果を保存
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
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
    setup_plotting_style,
    get_color_palette,
    get_body_part_colors,
    get_dataset_colors,
)

# プロット用スタイル設定
setup_plotting_style()

# 出力ディレクトリ
output_dir = project_root / "analysis" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)


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
        colors = get_color_palette(len(unique_values))
        dataset_colors = get_dataset_colors()

        for i, val in enumerate(unique_values):
            mask = np.array(metadata['datasets']) == val
            color = dataset_colors.get(val.lower(), colors[i])
            ax.scatter(
                embedded[mask, 0],
                embedded[mask, 1],
                c=[color],
                label=val.upper(),
                alpha=0.6,
                s=10,
                edgecolors='none'
            )

    elif color_by == 'body_part':
        # 身体部位をカテゴリ化
        categories = [categorize_body_part(loc) for loc in metadata['locations']]
        unique_categories = sorted(set(categories))
        body_part_colors = get_body_part_colors()

        for cat in unique_categories:
            mask = np.array(categories) == cat
            if not mask.any():
                continue
            color = body_part_colors.get(cat, '#gray')
            ax.scatter(
                embedded[mask, 0],
                embedded[mask, 1],
                c=color,
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
            colors = get_color_palette(len(unique_activities))
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
            colors = get_color_palette(len(unique_labels))
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
        colors = get_color_palette(len(unique_values))
        color_map = dict(zip(unique_values, colors))

        for val in unique_values:
            mask = np.array(metadata['dataset_location']) == val
            if not mask.any():
                continue
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
    ax.grid(True, alpha=0.3)

    # 凡例を外に配置（多数のアイテムがある場合）
    if len(ax.get_legend_handles_labels()[0]) > 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

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
                c='#CCCCCC',
                alpha=0.4,
                s=8,
                edgecolors='none',
                label='Other datasets'
            )

        # ハイライト：対象データセットのアクティビティ別
        target_mask = np.array(metadata['datasets']) == target_dataset
        target_activities = [metadata['activity_names'][i] for i in range(len(target_mask)) if target_mask[i]]
        unique_activities = sorted(set(target_activities))

        # 色マップ
        colors = get_color_palette(len(unique_activities))
        color_map = dict(zip(unique_activities, colors))

        # アクティビティごとにプロット
        activity_centers = {}
        for activity in unique_activities:
            activity_mask = target_mask & (np.array(metadata['activity_names']) == activity)
            if not activity_mask.any():
                continue

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
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                         edgecolor=color_map[activity], alpha=0.85, linewidth=1.2)
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

        # 背景：他のデータセット
        background_mask = np.array(metadata['datasets']) != target_dataset
        if background_mask.any():
            fig.add_trace(
                go.Scattergl(
                    x=embedded[background_mask, 0],
                    y=embedded[background_mask, 1],
                    mode='markers',
                    marker=dict(size=3, color='#CCCCCC', opacity=0.3),
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

        colors = get_color_palette(len(unique_activities))

        # matplotlibのRGBA → Plotlyのrgb文字列
        def rgba_to_rgb_string(rgba):
            if isinstance(rgba, (list, tuple)) and len(rgba) >= 3:
                r, g, b = [int(x * 255) if x <= 1 else int(x) for x in rgba[:3]]
                return f'rgb({r},{g},{b})'
            return rgba

        color_map = {activity: rgba_to_rgb_string(colors[i])
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


def plot_multi_model_comparison(all_embeddings, all_metadata, model_names,
                                color_by='body_part', output_path=None):
    """
    複数モデルの埋め込みを並べて比較

    Args:
        all_embeddings: モデルごとの埋め込みのリスト
        all_metadata: モデルごとのメタデータのリスト
        model_names: モデル名のリスト
        color_by: 色分けの基準
        output_path: 保存先パス
    """
    n_models = len(all_embeddings)
    fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 6))

    if n_models == 1:
        axes = [axes]

    for idx, (embedded, metadata, model_name) in enumerate(zip(all_embeddings, all_metadata, model_names)):
        ax = axes[idx]

        if color_by == 'body_part':
            categories = [categorize_body_part(loc) for loc in metadata['locations']]
            unique_categories = sorted(set(categories))
            body_part_colors = get_body_part_colors()

            for cat in unique_categories:
                mask = np.array(categories) == cat
                if not mask.any():
                    continue
                color = body_part_colors.get(cat, '#gray')
                ax.scatter(
                    embedded[mask, 0],
                    embedded[mask, 1],
                    c=color,
                    label=cat if idx == 0 else "",
                    alpha=0.6,
                    s=10,
                    edgecolors='none'
                )

        ax.set_xlabel('Dimension 1', fontsize=10)
        ax.set_ylabel('Dimension 2', fontsize=10)
        ax.set_title(model_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

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
    parser.add_argument('--method', type=str, default='umap', choices=['umap', 'tsne', 'pca'],
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
    parser.add_argument('--output-dir', type=str, default=str(output_dir),
                        help='Output directory for figures')

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # モデルパスのリスト
    if args.models:
        model_paths = args.models
    elif args.model:
        model_paths = [args.model]
    else:
        model_paths = [DEFAULT_MODEL]

    # データセット・部位の検出
    dataset_location_pairs = find_dataset_location_pairs(
        dataset_filter=args.datasets,
        location_filter=args.locations
    )

    if not dataset_location_pairs:
        print("Error: No dataset-location pairs found")
        return

    print(f"Found {len(dataset_location_pairs)} dataset-location pairs")

    # 複数モデルの比較
    if len(model_paths) > 1:
        all_embeddings = []
        all_metadata = []
        model_names = []

        for model_path in model_paths:
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
                    X, y, _ = load_sensor_data(
                        dataset_name, location,
                        window_size=window_size,
                        max_samples_per_class=args.max_samples
                    )

                    # 特徴抽出
                    features = extract_features(encoder, X, device=args.device, show_progress=False)

                    all_features.append(features)
                    all_datasets.extend([dataset_name] * len(features))
                    all_locations.extend([location] * len(features))
                    all_labels.extend(y.tolist())
                    all_dataset_location.extend([f"{dataset_name}/{location}"] * len(features))

                except Exception as e:
                    print(f"Error processing {dataset_name}/{location}: {e}")
                    continue

            # 特徴量を結合
            all_features = np.vstack(all_features)
            print(f"\nModel: {model_name}")
            print(f"Total samples: {len(all_features)}")

            # 次元削減
            embedded = reduce_dimensions(all_features, method=args.method)

            metadata = {
                'datasets': all_datasets,
                'locations': all_locations,
                'labels': all_labels,
                'dataset_location': all_dataset_location
            }

            all_embeddings.append(embedded)
            all_metadata.append(metadata)

        # 複数モデル比較プロット
        output_path = output_dir_path / f"embeddings_comparison_{args.method}_{args.color_by}.png"
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
                X, y, _ = load_sensor_data(
                    dataset_name, location,
                    window_size=window_size,
                    max_samples_per_class=args.max_samples,
                    max_users=20  # 大規模データセット対策
                )
                print(f"[LOADED] {dataset_name}/{location}: {len(X)} samples")

                # 特徴抽出
                features = extract_features(encoder, X, device=args.device, show_progress=False)

                # アクティビティ名を取得
                label_mapping = get_label_dict(dataset_name)
                activity_names = [label_mapping.get(int(label), f'unknown_{label}') for label in y]

                all_features.append(features)
                all_datasets.extend([dataset_name] * len(features))
                all_locations.extend([location] * len(features))
                all_labels.extend(y.tolist())
                all_dataset_location.extend([f"{dataset_name}/{location}"] * len(features))
                all_activity_names.extend(activity_names)

            except Exception as e:
                print(f"Error processing {dataset_name}/{location}: {e}")
                continue

        # 特徴量を結合
        all_features = np.vstack(all_features)
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

            # このグループだけで次元削減
            print(f"[{group_name}] Reducing dimensions using {args.method.upper()}...")
            filtered_embedded = reduce_dimensions(filtered_features, method=args.method)

            title = f"Foundation Model Embeddings - {group_name.replace('_', '/')} - {time_label}"
            output_path_png = output_dir_path / f"embeddings_{group_name}_{time_label}_{args.method}_{args.color_by}.png"
            output_path_html = output_dir_path / f"embeddings_{group_name}_{time_label}_{args.method}_{args.color_by}.html"

            print(f"[{group_name}] Generating plots...")

            # PNG版
            if args.color_by == 'activity_per_dataset':
                plot_per_dataset_activities(filtered_embedded, filtered_metadata,
                                           output_path=output_path_png, title=title)
            else:
                plot_embeddings(filtered_embedded, filtered_metadata, color_by=args.color_by,
                               output_path=output_path_png, title=title)

            print(f"[{group_name}] Saved PNG: {output_path_png}")

            # インタラクティブHTML版
            if args.color_by == 'activity_per_dataset':
                plot_per_dataset_activities_interactive(filtered_embedded, filtered_metadata,
                                                       output_path=output_path_html, title=title)
                print(f"[{group_name}] Saved HTML: {output_path_html}")

    print("\nDone!")


if __name__ == '__main__':
    main()
