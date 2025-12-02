"""
共通定数定義

プロジェクト全体で使用される定数を一元管理します。
"""

from typing import List

# サポートされるバックボーン（時系列センサーデータ用）
SUPPORTED_SENSOR_BACKBONES: List[str] = ["simple_cnn", "resnet1d", "deepconvlstm", "resnet", "limu_bert"]

# サポートされるオプティマイザー
SUPPORTED_OPTIMIZERS: List[str] = ["adam", "sgd", "adamw"]

# サポートされるスケジューラー
SUPPORTED_SCHEDULERS: List[str] = ["step", "cosine", "plateau", "exponential"]

# サポートされるSSLメソッド
SUPPORTED_SSL_METHODS: List[str] = ["simclr", "moco", "byol", "barlow_twins", "simsiam"]

# デフォルト値
DEFAULT_SEED: int = 42
DEFAULT_NUM_WORKERS: int = 4
DEFAULT_PIN_MEMORY: bool = True

# ログフォーマット
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
