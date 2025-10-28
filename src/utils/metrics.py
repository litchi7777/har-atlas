"""
評価メトリクス

分類タスクの評価指標を計算します。
"""

from typing import Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def calculate_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    average: str = "macro",
) -> Dict[str, float]:
    """
    分類メトリクスを計算

    Args:
        y_true: 正解ラベル
        y_pred: 予測ラベル
        average: 多クラス時の平均化方法 ('macro', 'micro', 'weighted')

    Returns:
        メトリクス辞書
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }

    return metrics


def get_confusion_matrix(
    y_true: Union[List[int], np.ndarray], y_pred: Union[List[int], np.ndarray]
) -> np.ndarray:
    """
    混同行列を計算

    Args:
        y_true: 正解ラベル
        y_pred: 予測ラベル

    Returns:
        混同行列
    """
    return confusion_matrix(y_true, y_pred)


def get_classification_report(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    target_names: Optional[List[str]] = None,
) -> str:
    """
    詳細な分類レポートを取得

    Args:
        y_true: 正解ラベル
        y_pred: 予測ラベル
        target_names: クラス名のリスト

    Returns:
        分類レポート文字列
    """
    return classification_report(y_true, y_pred, target_names=target_names)
