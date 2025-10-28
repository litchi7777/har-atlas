"""
損失関数モジュール
"""

from .ssl_losses import NTXentLoss, IntegratedSSLLoss, MultiTaskLoss, get_ssl_loss

__all__ = ["NTXentLoss", "IntegratedSSLLoss", "MultiTaskLoss", "get_ssl_loss"]
