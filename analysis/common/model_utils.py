"""
モデル読み込み・特徴抽出ユーティリティ

事前学習済みモデルの読み込み、特徴量抽出、ウィンドウサイズ検出など、
モデル関連の共通処理を提供します。
"""

import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from models.backbones import Resnet, IntegratedSSLModel


def get_model_window_size(model_path: str, default: int = 150) -> int:
    """
    モデルのウィンドウサイズを自動検出

    Args:
        model_path: モデルファイルのパス
        default: デフォルトのウィンドウサイズ

    Returns:
        window_size: ウィンドウサイズ（時間ステップ数）
    """
    model_path = Path(model_path)

    # 1. チェックポイント内のconfigから取得を試みる
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'config' in checkpoint:
            config = checkpoint['config']
            if 'sensor_data' in config and 'window_size' in config['sensor_data']:
                window_size = config['sensor_data']['window_size']
                print(f"  Detected window_size from checkpoint: {window_size}")
                return window_size
    except Exception as e:
        print(f"  Warning: Could not read checkpoint config: {e}")

    # 2. 実験ディレクトリのconfig.yamlから取得
    config_path = model_path.parent.parent / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            if 'sensor_data' in config and 'window_size' in config['sensor_data']:
                window_size = config['sensor_data']['window_size']
                print(f"  Detected window_size from config.yaml: {window_size}")
                return window_size
        except Exception as e:
            print(f"  Warning: Could not read config.yaml: {e}")

    # 3. デフォルト値を使用
    print(f"  Using default window_size: {default}")
    return default


def load_pretrained_model(
    model_path: str,
    window_size: Optional[int] = None,
    device: str = 'cuda',
    n_channels: int = 3,
    hidden_dim: int = 256,
    ssl_tasks: Optional[list] = None,
) -> Tuple[torch.nn.Module, int]:
    """
    事前学習済みモデルを読み込む

    Args:
        model_path: モデルファイルのパス
        window_size: ウィンドウサイズ（Noneの場合は自動検出）
        device: 'cuda' or 'cpu'
        n_channels: 入力チャンネル数
        hidden_dim: 隠れ層の次元数
        ssl_tasks: SSLタスクのリスト

    Returns:
        encoder: エンコーダーモデル（評価モード）
        window_size: 実際のウィンドウサイズ
    """
    print(f"Loading model from: {model_path}")

    # ウィンドウサイズの自動検出
    if window_size is None:
        window_size = get_model_window_size(model_path)

    print(f"  Window size: {window_size} samples")

    # アーキテクチャ自動選択
    nano_window = window_size < 20   # 15サンプル用
    micro_window = 20 <= window_size < 100  # 30, 60サンプル用

    arch_type = "nano" if nano_window else ("micro" if micro_window else "default")
    print(f"  Architecture: {arch_type}_window ({window_size} samples)")

    # チェックポイント読み込み
    checkpoint = torch.load(model_path, map_location=device)

    # バックボーン作成
    backbone = Resnet(
        n_channels=n_channels,
        foundationUK=False,
        micro_window=micro_window,
        nano_window=nano_window
    )

    # IntegratedSSLModel作成
    if ssl_tasks is None:
        ssl_tasks = ['binary_permute', 'binary_reverse', 'binary_timewarp']

    model = IntegratedSSLModel(
        backbone=backbone,
        ssl_tasks=ssl_tasks,
        hidden_dim=hidden_dim,
        n_channels=n_channels,
        sequence_length=window_size,
        device=torch.device(device)
    )

    # 重みを読み込み
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()

    # エンコーダーのみ取り出し
    encoder = model.backbone
    encoder.eval()

    print(f"  Model loaded successfully. Feature dim: {encoder.output_dim}")

    return encoder, window_size


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
