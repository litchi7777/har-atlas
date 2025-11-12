"""
分析スクリプト共通ユーティリティ

このモジュールは、analysis/配下の全スクリプトで共通して使用する
関数やクラスを提供します。
"""

import os
import sys
import json
import glob
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "har-unified-dataset" / "src"))

from dataset_info import DATASETS


def get_project_root() -> Path:
    """プロジェクトルートのパスを取得"""
    return Path(__file__).parent.parent


def get_processed_data_root() -> Path:
    """処理済みデータのルートディレクトリを取得"""
    return get_project_root() / "har-unified-dataset" / "data" / "processed"


def get_experiment_root(mode: str = "finetune") -> Path:
    """
    実験結果のルートディレクトリを取得

    Args:
        mode: 'pretrain' or 'finetune'

    Returns:
        実験ディレクトリのパス
    """
    return get_project_root() / "experiments" / mode


def find_dataset_location_pairs(
    dataset_filter: Optional[List[str]] = None,
    location_filter: Optional[List[str]] = None
) -> List[Tuple[str, str]]:
    """
    利用可能なデータセット・身体部位のペアを検出

    Args:
        dataset_filter: 含めるデータセット名のリスト（Noneの場合は全て）
        location_filter: 含める身体部位のリスト（Noneの場合は全て）

    Returns:
        (dataset_name, location)のタプルのリスト
    """
    data_root = get_processed_data_root()
    pairs = []

    for dataset_path in sorted(data_root.iterdir()):
        if not dataset_path.is_dir():
            continue

        dataset_name = dataset_path.name

        if dataset_filter and dataset_name not in dataset_filter:
            continue

        # USER*/location/ACC/X.npy のパターンで検索
        pattern = str(dataset_path / "*/*/ACC/X.npy")
        x_paths = glob.glob(pattern)

        locations = set()
        for x_path in x_paths:
            parts = Path(x_path).parts
            if len(parts) >= 3:
                location = parts[-3]  # ACC の2つ上
                locations.add(location)

        for location in sorted(locations):
            if location_filter and location not in location_filter:
                continue

            pairs.append((dataset_name, location))

    return pairs


def load_sensor_data(
    dataset_name: str,
    location: str,
    max_samples_per_class: Optional[int] = None,
    exclude_negative_labels: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    指定されたデータセット・部位のセンサーデータを読み込む

    Args:
        dataset_name: データセット名（例: 'dsads'）
        location: 身体部位（例: 'Torso'）
        max_samples_per_class: 各クラスから取得する最大サンプル数（Noneで全て）
        exclude_negative_labels: 負のラベルを除外するか

    Returns:
        X: センサーデータ (N, 3, T)
        y: ラベル (N,)
        metadata: メタデータ辞書
    """
    data_root = get_processed_data_root()
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
        # labels.npy を優先的に読み込み、なければ Y.npy
        labels_path = x_path.replace("/X.npy", "/labels.npy")
        y_path = x_path.replace("/X.npy", "/Y.npy")

        if Path(labels_path).exists():
            y_data = np.load(labels_path, mmap_mode='r')
        elif Path(y_path).exists():
            y_data = np.load(y_path, mmap_mode='r')
        else:
            print(f"Warning: No label file found for {x_path}, skipping")
            continue

        # Y.npyはラベル（1次元）を想定
        if y_data.ndim != 1:
            print(f"Warning: Label file is not 1D at {x_path}, skipping")
            continue

        # データ読み込み
        X_data = np.load(x_path, mmap_mode='r')

        all_X.append(X_data)
        all_y.append(y_data)

    if not all_X:
        raise ValueError(f"No valid data found for {dataset_name}/{location}")

    # 結合
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # 負のラベルを除外
    if exclude_negative_labels:
        valid_mask = y >= 0
        X = X[valid_mask]
        y = y[valid_mask]

    # クラスごとにサンプリング
    if max_samples_per_class is not None:
        unique_labels = np.unique(y)
        sampled_indices = []

        for label in unique_labels:
            label_indices = np.where(y == label)[0]
            if len(label_indices) > max_samples_per_class:
                sampled = np.random.choice(label_indices, max_samples_per_class, replace=False)
            else:
                sampled = label_indices
            sampled_indices.extend(sampled)

        sampled_indices = np.array(sampled_indices)
        np.random.shuffle(sampled_indices)

        X = X[sampled_indices]
        y = y[sampled_indices]

    # メタデータ読み込み
    metadata_path = dataset_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    return X, y, metadata


def load_pretrained_model(
    model_path: str,
    device: str = 'cuda'
) -> torch.nn.Module:
    """
    事前学習済みモデルを読み込む

    Args:
        model_path: モデルファイルのパス
        device: 'cuda' or 'cpu'

    Returns:
        encoder: エンコーダーモデル（評価モード）
    """
    from src.models.model import SSLModel

    print(f"Loading model from: {model_path}")

    # チェックポイント読み込み
    checkpoint = torch.load(model_path, map_location=device)

    # モデルの設定を取得
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # デフォルト設定（ResNet18）
        config = {
            'model': {
                'name': 'resnet',
                'backbone': 'resnet18',
                'feature_dim': 256,
                'projection_dim': 128
            }
        }

    # モデル初期化
    model = SSLModel(
        backbone_name=config['model'].get('backbone', 'resnet18'),
        feature_dim=config['model'].get('feature_dim', 256),
        projection_dim=config['model'].get('projection_dim', 128)
    )

    # 重みを読み込み
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # エンコーダーのみ取り出し
    encoder = model.encoder
    encoder.to(device)
    encoder.eval()

    print(f"Model loaded successfully. Feature dim: {config['model'].get('feature_dim', 256)}")

    return encoder


def extract_features(
    encoder: torch.nn.Module,
    X: np.ndarray,
    batch_size: int = 256,
    device: str = 'cuda',
    show_progress: bool = True
) -> np.ndarray:
    """
    エンコーダーで特徴量を抽出

    Args:
        encoder: エンコーダーモデル
        X: センサーデータ (N, 3, T)
        batch_size: バッチサイズ
        device: デバイス
        show_progress: プログレスバーを表示するか

    Returns:
        features: 特徴量 (N, feature_dim)
    """
    from tqdm import tqdm

    encoder.eval()
    features = []

    iterator = range(0, len(X), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Extracting features")

    with torch.no_grad():
        for i in iterator:
            batch = X[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).to(device)

            # エンコーダーに入力
            batch_features = encoder(batch_tensor)

            features.append(batch_features.cpu().numpy())

    features = np.concatenate(features, axis=0)
    return features


def get_label_dict(dataset_name: str) -> Dict[int, str]:
    """
    データセットのラベル辞書を取得

    Args:
        dataset_name: データセット名

    Returns:
        ラベル辞書 {0: 'walking', 1: 'sitting', ...}
    """
    dataset_upper = dataset_name.upper()
    return DATASETS.get(dataset_upper, {}).get('labels', {})


def categorize_body_part(location: str) -> str:
    """
    身体部位を大まかなカテゴリに分類

    Args:
        location: 身体部位名

    Returns:
        カテゴリ名（'手・腕', '胴体・腰', '足', 'その他'）
    """
    body_part_categories = {
        'hand': '手・腕',
        'RightWrist': '手・腕',
        'LeftWrist': '手・腕',
        'RightArm': '手・腕',
        'LeftArm': '手・腕',
        'Wrist': '手・腕',
        'RightUpperArm': '手・腕',
        'LeftUpperArm': '手・腕',
        'RightLowerArm': '手・腕',
        'LeftLowerArm': '手・腕',

        'Chest': '胴体・腰',
        'Torso': '胴体・腰',
        'LowerBack': '胴体・腰',
        'Back': '胴体・腰',
        'Hip': '胴体・腰',
        'chest': '胴体・腰',
        'Neck': '胴体・腰',

        'ankle': '足',
        'LeftAnkle': '足',
        'RightAnkle': '足',
        'RightLeg': '足',
        'LeftLeg': '足',
        'Thigh': '足',
        'RightThigh': '足',
        'LeftThigh': '足',
        'RightCalf': '足',
        'LeftCalf': '足',

        'PAX': 'その他'
    }

    return body_part_categories.get(location, 'その他')


def get_body_part_color(category: str) -> str:
    """
    身体部位カテゴリの色を取得

    Args:
        category: カテゴリ名

    Returns:
        カラーコード
    """
    colors = {
        '手・腕': '#ff6b6b',
        '胴体・腰': '#4ecdc4',
        '足': '#45b7d1',
        'その他': '#gray'
    }
    return colors.get(category, '#gray')


def load_experiment_config(experiment_path: Path) -> Dict[str, Any]:
    """
    実験の設定ファイルを読み込む

    Args:
        experiment_path: 実験ディレクトリのパス

    Returns:
        設定辞書
    """
    config_path = experiment_path / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def load_experiment_results(experiment_path: Path) -> Dict[str, Any]:
    """
    実験結果を読み込む

    Args:
        experiment_path: 実験ディレクトリのパス

    Returns:
        結果辞書（メトリクス、損失履歴等）
    """
    results = {}

    # summary.json
    summary_path = experiment_path / "summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            results['summary'] = json.load(f)

    # metrics.json
    metrics_path = experiment_path / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            results['metrics'] = json.load(f)

    # history.json（学習履歴）
    history_path = experiment_path / "history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            results['history'] = json.load(f)

    return results


def find_checkpoint_files(
    experiment_path: Path,
    checkpoint_type: str = "best"
) -> List[Path]:
    """
    実験ディレクトリからチェックポイントファイルを検索

    Args:
        experiment_path: 実験ディレクトリのパス
        checkpoint_type: 'best', 'latest', 'all'

    Returns:
        チェックポイントファイルのパスのリスト
    """
    models_dir = experiment_path / "models"

    if not models_dir.exists():
        return []

    if checkpoint_type == "best":
        pattern = "best_model.pth"
    elif checkpoint_type == "latest":
        pattern = "checkpoint_epoch_*.pth"
    elif checkpoint_type == "all":
        pattern = "*.pth"
    else:
        raise ValueError(f"Unknown checkpoint_type: {checkpoint_type}")

    checkpoint_files = sorted(models_dir.glob(pattern))

    return checkpoint_files


def print_section_header(title: str, width: int = 80):
    """
    セクションヘッダーを出力

    Args:
        title: タイトル
        width: 幅
    """
    print(f"\n{'='*width}")
    print(f"{title}")
    print(f"{'='*width}")


def print_subsection_header(title: str, width: int = 80):
    """
    サブセクションヘッダーを出力

    Args:
        title: タイトル
        width: 幅
    """
    print(f"\n{'-'*width}")
    print(f"{title}")
    print(f"{'-'*width}")


def format_metric(value: float, metric_type: str = "default") -> str:
    """
    メトリクスを見やすくフォーマット

    Args:
        value: 値
        metric_type: 'accuracy', 'loss', 'percent', 'default'

    Returns:
        フォーマット済み文字列
    """
    if metric_type == "accuracy":
        return f"{value*100:.2f}%"
    elif metric_type == "loss":
        return f"{value:.4f}"
    elif metric_type == "percent":
        return f"{value:.2f}%"
    else:
        return f"{value:.4f}"
