"""
共通ユーティリティ関数

プロジェクト全体で使用される共通機能を提供します。
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """
    再現性のため、すべての乱数生成器のシードを設定

    Args:
        seed: シード値
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # マルチGPU対応

    # CuDNNの決定的動作を有効化（パフォーマンスがやや低下する可能性あり）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_name: str = "cuda") -> torch.device:
    """
    デバイスを取得（CUDA、MPS、CPUの自動判定）

    Args:
        device_name: 希望するデバイス名 ('cuda', 'mps', 'cpu')

    Returns:
        利用可能なデバイス
    """
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    モデルのパラメータ数をカウント

    Args:
        model: PyTorchモデル

    Returns:
        パラメータ数の辞書（total, trainable, non_trainable）
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": non_trainable_params,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    save_path: str,
    filename: Optional[str] = None,
    is_best: bool = False,
) -> str:
    """
    モデルチェックポイントを保存

    Args:
        model: 保存するモデル
        optimizer: オプティマイザー
        epoch: 現在のエポック
        metrics: 評価メトリクス
        save_path: 保存先ディレクトリ
        filename: ファイル名（Noneの場合はエポック番号から生成）
        is_best: ベストモデルとして保存するか

    Returns:
        保存したファイルパス
    """
    os.makedirs(save_path, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    # ファイル名を決定
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pth"

    save_file = os.path.join(save_path, filename)
    torch.save(checkpoint, save_file)

    # ベストモデルの場合は別途保存
    if is_best:
        best_file = os.path.join(save_path, "best_model.pth")
        torch.save(checkpoint, best_file)

    return save_file


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    チェックポイントからモデルを読み込み

    Args:
        checkpoint_path: チェックポイントファイルパス
        model: ロード先のモデル
        optimizer: オプティマイザー（Noneの場合はロードしない）
        device: デバイス
        strict: state_dictの厳密なマッチングを要求するか

    Returns:
        チェックポイント情報（epoch, metrics等）
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # モデルの重みをロード
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    # オプティマイザーの状態をロード（指定されている場合）
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
    }


def ensure_dir(directory: str) -> Path:
    """
    ディレクトリが存在することを確認（存在しない場合は作成）

    Args:
        directory: ディレクトリパス

    Returns:
        Pathオブジェクト
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_time(seconds: float) -> str:
    """
    秒を人間が読みやすい形式にフォーマット

    Args:
        seconds: 秒数

    Returns:
        フォーマットされた時間文字列
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    オプティマイザーから現在の学習率を取得

    Args:
        optimizer: オプティマイザー

    Returns:
        現在の学習率
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return 0.0
