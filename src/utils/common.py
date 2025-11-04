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
        device_name: 希望するデバイス名 ('cuda', 'cuda:0', 'cuda:1', 'mps', 'cpu')

    Returns:
        利用可能なデバイス
    """
    if device_name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_name)
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
    keep_top_k: Optional[int] = None,
    metric_key: str = "val.loss",
    mode: str = "min",
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
        keep_top_k: 保持するチェックポイントの最大数（Noneの場合は全て保持）
        metric_key: 評価に使用するメトリクスのキー（例: "val.loss", "train.loss"）
        mode: 評価モード（"min"または"max"）

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

    # Top-k チェックポイント管理
    if keep_top_k is not None and keep_top_k > 0:
        _manage_checkpoints(save_path, keep_top_k, metric_key, mode)

    return save_file


def _manage_checkpoints(
    save_path: str,
    keep_top_k: int,
    metric_key: str = "val.loss",
    mode: str = "min",
) -> None:
    """
    チェックポイントディレクトリ内のファイルを管理し、top-kのみを保持

    Args:
        save_path: チェックポイントディレクトリ
        keep_top_k: 保持するチェックポイント数
        metric_key: 評価に使用するメトリクスのキー
        mode: "min"（小さいほど良い）または "max"（大きいほど良い）
    """
    from pathlib import Path
    import glob

    # checkpoint_epoch_*.pth ファイルを検索
    checkpoint_pattern = os.path.join(save_path, "checkpoint_epoch_*.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)

    if len(checkpoint_files) <= keep_top_k:
        return  # 削除の必要なし

    # 各チェックポイントのメトリクスを読み込む
    checkpoint_metrics = []
    for ckpt_file in checkpoint_files:
        try:
            ckpt = torch.load(ckpt_file, map_location="cpu")
            metrics = ckpt.get("metrics", {})

            # ネストされたメトリクスキーをサポート（例: "val.loss"）
            metric_value = metrics
            for key in metric_key.split("."):
                if isinstance(metric_value, dict):
                    metric_value = metric_value.get(key)
                else:
                    break

            # メトリクスが取得できない場合はスキップ
            if metric_value is None:
                continue

            checkpoint_metrics.append({
                "file": ckpt_file,
                "epoch": ckpt.get("epoch", 0),
                "metric": float(metric_value),
            })
        except Exception as e:
            # ファイル読み込みに失敗した場合は警告して続行
            print(f"Warning: Failed to load checkpoint {ckpt_file}: {e}")
            continue

    if len(checkpoint_metrics) <= keep_top_k:
        return

    # メトリクスでソート
    reverse = (mode == "max")
    checkpoint_metrics.sort(key=lambda x: x["metric"], reverse=reverse)

    # 削除するチェックポイント（下位のもの）
    to_delete = checkpoint_metrics[keep_top_k:]

    for ckpt_info in to_delete:
        try:
            os.remove(ckpt_info["file"])
            print(f"Removed checkpoint: {Path(ckpt_info['file']).name} "
                  f"(epoch={ckpt_info['epoch']}, {metric_key}={ckpt_info['metric']:.4f})")
        except Exception as e:
            print(f"Warning: Failed to remove checkpoint {ckpt_info['file']}: {e}")


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
