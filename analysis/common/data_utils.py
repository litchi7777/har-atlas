"""
データ読み込み・処理ユーティリティ

センサーデータの読み込み、ウィンドウクリッピング、身体部位カテゴリ化など、
データ処理関連の共通処理を提供します。
"""

import sys
import json
import glob
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "har-unified-dataset" / "src"))

from dataset_info import DATASETS


def get_processed_data_root() -> Path:
    """処理済みデータのルートディレクトリを取得"""
    return project_root / "har-unified-dataset" / "data" / "processed"


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
    window_size: Optional[int] = None,
    max_samples_per_class: Optional[int] = None,
    exclude_negative_labels: bool = True,
    max_users: Optional[int] = None,
    user_ids: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    指定されたデータセット・部位のセンサーデータを読み込む

    Args:
        dataset_name: データセット名（例: 'dsads'）
        location: 身体部位（例: 'Torso'）
        window_size: クリップ後のウィンドウサイズ（Noneの場合はクリップしない）
        max_samples_per_class: 各クラスから取得する最大サンプル数（Noneで全て）
        exclude_negative_labels: 負のラベルを除外するか
        max_users: 最大ユーザー数（大規模データセット用、Noneで全て）
        user_ids: 使用するユーザーIDのリスト（例: [1, 2]でテストユーザーのみ）

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

    # 特定のユーザーIDのみ使用
    if user_ids is not None:
        import re
        filtered_paths = []
        for p in x_paths:
            # パスからユーザーIDを抽出（例: USER1, USER02, user_1 など）
            match = re.search(r'[Uu][Ss][Ee][Rr]_?(\d+)', p)
            if match:
                uid = int(match.group(1))
                if uid in user_ids:
                    filtered_paths.append(p)
        print(f"  [INFO] Using users {user_ids}: {len(filtered_paths)} out of {len(x_paths)} for {dataset_name}/{location}")
        x_paths = filtered_paths
    # 大規模データセットは最初のN人のみ使用
    elif max_users is not None and len(x_paths) > max_users:
        print(f"  [INFO] Using {max_users} out of {len(x_paths)} users for {dataset_name}/{location}")
        x_paths = x_paths[:max_users]

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

        # ラベルは1次元を想定
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

    # ウィンドウクリップ（中央切り出し）
    if window_size is not None and X.shape[2] > window_size:
        X = clip_windows(X, window_size, strategy='center')

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


def clip_windows(
    X: np.ndarray,
    window_size: int,
    strategy: str = 'center'
) -> np.ndarray:
    """
    ウィンドウをクリップする

    Args:
        X: センサーデータ (N, C, T)
        window_size: クリップ後のウィンドウサイズ
        strategy: 'center', 'start', 'end', 'random'

    Returns:
        clipped_X: (N, C, window_size)
    """
    N, C, T = X.shape

    if T < window_size:
        raise ValueError(f"Window size {window_size} is larger than data length {T}")

    if T == window_size:
        return X

    max_start = T - window_size

    if strategy == 'center':
        # 中央から切り出し
        start = max_start // 2
        return X[:, :, start:start+window_size]

    elif strategy == 'start':
        # 先頭から切り出し
        return X[:, :, :window_size]

    elif strategy == 'end':
        # 末尾から切り出し
        return X[:, :, -window_size:]

    elif strategy == 'random':
        # ランダムに切り出し（サンプルごとに異なる位置）
        clipped_X = np.zeros((N, C, window_size), dtype=X.dtype)
        for i in range(N):
            start = np.random.randint(0, max_start + 1)
            clipped_X[i] = X[i, :, start:start+window_size]
        return clipped_X

    else:
        raise ValueError(f"Unknown clip strategy: {strategy}")


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
    身体部位をカテゴリに分類（統一版）

    Args:
        location: 身体部位名

    Returns:
        カテゴリ名
    """
    location_lower = location.lower()

    # HHAR デバイス名（優先度高）- すべてPhone（ポケット/手持ち）として扱う
    hhar_devices = ['gear_1', 'gear_2', 'lgwatch_1', 'lgwatch_2',
                    'nexus4_1', 'nexus4_2', 's3_1', 's3_2', 's3mini_1', 's3mini_2']
    if location_lower in hhar_devices or location in hhar_devices:
        return 'Phone'

    # Wrist（優先度高）
    if 'wrist' in location_lower or 'atr01' in location_lower or 'atr02' in location_lower:
        return 'Wrist'

    # Ankle（優先度高）
    if 'ankle' in location_lower:
        return 'Ankle'

    # Head
    if any(kw in location_lower for kw in ['head', 'forehead', 'ear', 'neck']):
        return 'Head'

    # Phone/Pocket
    if 'phone' in location_lower or 'pocket' in location_lower:
        return 'Phone'

    # Back
    if any(kw in location_lower for kw in ['back', 'lumbar', 'spine']):
        return 'Back'

    # Front (chest, torso, waist, hip)
    if any(kw in location_lower for kw in ['chest', 'torso', 'waist', 'belt', 'hip']):
        return 'Front'

    # Arm (upper arm, forearm, shoulder, hand, elbow)
    if any(kw in location_lower for kw in ['arm', 'hand', 'shoulder', 'elbow', 'finger', 'atr03', 'atr04']):
        return 'Arm'

    # Leg (thigh, knee, shin, foot, calf)
    if any(kw in location_lower for kw in ['leg', 'thigh', 'knee', 'shin', 'foot', 'calf']):
        return 'Leg'

    # PAX (NHANES accelerometer)
    if 'pax' in location_lower:
        return 'PAX'

    # デフォルト: そのまま返す
    return location
