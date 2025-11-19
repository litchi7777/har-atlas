"""
可視化ユーティリティ

次元削減、プロット設定、カラーパレットなど、
可視化関連の共通処理を提供します。
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal, Dict


def setup_plotting_style(style: str = 'whitegrid', context: str = 'paper', font_scale: float = 1.0):
    """
    プロット用のスタイル設定

    Args:
        style: seabornスタイル ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
        context: コンテキスト ('paper', 'notebook', 'talk', 'poster')
        font_scale: フォントスケール
    """
    sns.set_style(style)
    sns.set_context(context, font_scale=font_scale)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def reduce_dimensions(
    features: np.ndarray,
    method: Literal['umap', 'tsne', 'pca'] = 'umap',
    n_components: int = 2,
    **kwargs
) -> np.ndarray:
    """
    特徴量を次元削減

    Args:
        features: 特徴量 (N, D)
        method: 次元削減手法 ('umap', 'tsne', 'pca')
        n_components: 削減後の次元数
        **kwargs: 各手法固有のパラメータ

    Returns:
        embedded: 次元削減後の特徴量 (N, n_components)
    """
    print(f"Reducing dimensions using {method.upper()}...")

    if method == 'umap':
        import umap
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=kwargs.get('n_neighbors', 15),
            min_dist=kwargs.get('min_dist', 0.1),
            metric=kwargs.get('metric', 'cosine'),
            random_state=kwargs.get('random_state', 42)
        )
        embedded = reducer.fit_transform(features)

    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=n_components,
            perplexity=kwargs.get('perplexity', 30),
            max_iter=kwargs.get('max_iter', 1000),
            random_state=kwargs.get('random_state', 42)
        )
        embedded = reducer.fit_transform(features)

    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(
            n_components=n_components,
            random_state=kwargs.get('random_state', 42)
        )
        embedded = reducer.fit_transform(features)

    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"  Reduced from {features.shape[1]}D to {n_components}D")
    return embedded


def get_color_palette(
    n_colors: int,
    palette_name: str = 'auto'
) -> list:
    """
    カラーパレットを取得

    Args:
        n_colors: 色の数
        palette_name: パレット名 ('auto', 'Set1', 'Set3', 'tab20', 'hsv', など)

    Returns:
        colors: 色のリスト
    """
    if palette_name == 'auto':
        # 色数に応じて自動選択
        if n_colors <= 10:
            palette_name = 'Set1'
        elif n_colors <= 12:
            palette_name = 'Set3'
        elif n_colors <= 20:
            palette_name = 'tab20'
        else:
            palette_name = 'hsv'

    if palette_name == 'hsv':
        # 連続カラーマップから均等サンプリング
        cmap = plt.cm.hsv
        colors = [cmap(i / n_colors) for i in range(n_colors)]
    elif palette_name in ['tab20', 'tab20b', 'tab20c']:
        # tab20系（最大20色）
        cmap = getattr(plt.cm, palette_name)
        colors = [cmap(i / min(n_colors, 20)) for i in range(n_colors)]
    else:
        # seabornパレット
        colors = sns.color_palette(palette_name, n_colors)

    return colors


def get_body_part_colors() -> Dict[str, str]:
    """
    身体部位カテゴリごとの色マッピングを取得

    Returns:
        colors: カテゴリ名 -> カラーコードの辞書
    """
    return {
        'Wrist': '#ff6b6b',      # 赤系
        'Ankle': '#4ecdc4',      # 青緑系
        'Arm': '#45b7d1',        # 青系
        'Leg': '#96ceb4',        # 緑系
        'Front': '#ffeaa7',      # 黄色系
        'Back': '#dfe6e9',       # グレー系
        'Head': '#a29bfe',       # 紫系
        'Phone': '#fdcb6e',      # オレンジ系
        'PAX': '#b2bec3',        # ダークグレー
        'Other': '#636e72',      # グレー
    }


def get_dataset_colors() -> Dict[str, str]:
    """
    データセットごとの色マッピングを取得

    Returns:
        colors: データセット名 -> カラーコードの辞書
    """
    # 主要データセット用の固定カラー
    return {
        'dsads': '#e74c3c',
        'mhealth': '#3498db',
        'pamap2': '#2ecc71',
        'harth': '#f39c12',
        'realdisp': '#9b59b6',
        'uschad': '#1abc9c',
        'forthtrace': '#e67e22',
        'har70plus': '#34495e',
        'lara': '#16a085',
        'mex': '#c0392b',
        'paal': '#8e44ad',
        'selfback': '#27ae60',
        'nhanes': '#95a5a6',
    }


def save_figure(
    fig,
    output_path: str,
    dpi: int = 150,
    bbox_inches: str = 'tight',
    **kwargs
):
    """
    図を保存

    Args:
        fig: matplotlib figure
        output_path: 保存先パス
        dpi: 解像度
        bbox_inches: bounding box設定
        **kwargs: その他のsavefigパラメータ
    """
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    print(f"Saved: {output_path}")
