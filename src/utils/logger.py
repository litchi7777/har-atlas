"""
ロギングユーティリティ

ファイルとコンソールへのロギングを設定します。
"""

import os
import logging
from datetime import datetime
from typing import Optional

from .constants import LOG_FORMAT, DATE_FORMAT


def setup_logger(
    name: str, log_dir: Optional[str] = None, log_file: Optional[str] = None, level: int = logging.INFO, console_only: bool = False
) -> logging.Logger:
    """
    ロガーを設定（ファイルとコンソールハンドラー付き）

    Args:
        name: ロガー名
        log_dir: ログディレクトリ（console_only=Trueの場合は不要）
        log_file: ログファイル名（Noneの場合は自動生成）
        level: ログレベル
        console_only: Trueの場合、コンソール出力のみ（ファイルに書き込まない）

    Returns:
        設定済みロガー
    """
    # ロガーを作成（既存のロガーを取得）
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # ハンドラーがすでに存在する場合はスキップ（重複防止）
    if logger.handlers:
        return logger

    # フォーマッターを作成
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # ファイルハンドラー（console_only=Falseの場合のみ）
    if not console_only:
        if log_dir is None:
            raise ValueError("log_dir must be specified when console_only=False")

        os.makedirs(log_dir, exist_ok=True)

        # ログファイル名を決定
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{name}_{timestamp}.log"

        log_path = os.path.join(log_dir, log_file)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # コンソールハンドラー
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # ハンドラーを追加
    logger.addHandler(console_handler)

    # 親ロガーへの伝播を防止（重複ログ防止）
    logger.propagate = False

    return logger
