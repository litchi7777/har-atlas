"""
設定管理モジュール

YAML設定ファイルの読み込み、検証、保存を行います。
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .constants import (
    SUPPORTED_SENSOR_BACKBONES,
    SUPPORTED_OPTIMIZERS,
    SUPPORTED_SCHEDULERS,
    SUPPORTED_SSL_METHODS,
)


class ConfigValidationError(Exception):
    """設定検証エラー"""

    pass


def load_config(config_path: str) -> Dict[str, Any]:
    """
    YAML設定ファイルを読み込み

    Args:
        config_path: 設定ファイルのパス

    Returns:
        設定辞書

    Raises:
        FileNotFoundError: ファイルが存在しない場合
        yaml.YAMLError: YAML解析エラーの場合
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    設定をYAMLファイルとして保存

    Args:
        config: 設定辞書
        save_path: 保存先パス
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def validate_config(config: Dict[str, Any], mode: str = "pretrain") -> None:
    """
    設定の妥当性を検証

    Args:
        config: 検証する設定辞書
        mode: モード ('pretrain' または 'finetune')

    Raises:
        ConfigValidationError: 設定が無効な場合
    """
    # データセットタイプを確認
    dataset_type = config.get("dataset_type", "image")

    # 必須セクションの確認
    if dataset_type == "sensor":
        required_sections = ["model", "sensor_data", "training", "device", "seed"]
    else:
        required_sections = ["model", "data", "training", "device", "seed"]

    for section in required_sections:
        if section not in config:
            raise ConfigValidationError(f"Missing required section: {section}")

    # モデル設定の検証
    validate_model_config(config["model"], mode, dataset_type)

    # データ設定の検証
    if dataset_type == "sensor":
        validate_sensor_data_config(config["sensor_data"], mode)
    else:
        validate_data_config(config["data"], mode)

    # トレーニング設定の検証
    validate_training_config(config["training"])


def validate_model_config(
    model_config: Dict[str, Any], mode: str, dataset_type: str = "image"
) -> None:
    """
    モデル設定の検証

    Args:
        model_config: モデル設定辞書
        mode: モード ('pretrain' または 'finetune')
        dataset_type: データセットタイプ ('image' または 'sensor')

    Raises:
        ConfigValidationError: 設定が無効な場合
    """
    # バックボーンの検証（センサーデータのみサポート）
    if "backbone" not in model_config:
        raise ConfigValidationError("Missing 'backbone' in model config")

    supported = SUPPORTED_SENSOR_BACKBONES
    if model_config["backbone"] not in supported:
        raise ConfigValidationError(
            f"Unsupported backbone: {model_config['backbone']}. " f"Supported: {supported}"
        )

    # モード固有の検証
    if mode == "finetune":
        if "num_classes" not in model_config:
            raise ConfigValidationError("Missing 'num_classes' in finetune model config")

        if model_config["num_classes"] < 2:
            raise ConfigValidationError(
                f"num_classes must be >= 2, got {model_config['num_classes']}"
            )


def validate_sensor_data_config(sensor_config: Dict[str, Any], mode: str) -> None:
    """
    センサーデータ設定の検証

    Args:
        sensor_config: センサーデータ設定辞書
        mode: モード ('pretrain' または 'finetune')

    Raises:
        ConfigValidationError: 設定が無効な場合
    """
    # バッチローダー使用時は最小限のチェック
    if "batch_loader" in sensor_config:
        batch_loader_config = sensor_config["batch_loader"]
        if "dataset_patterns" not in batch_loader_config:
            raise ConfigValidationError("Missing 'dataset_patterns' in batch_loader config")
        if "sample_threshold" not in batch_loader_config:
            raise ConfigValidationError("Missing 'sample_threshold' in batch_loader config")
        return  # バッチローダー使用時は他のチェックをスキップ

    # 従来の設定の検証
    if "dataset_name" not in sensor_config:
        raise ConfigValidationError("Missing 'dataset_name' in sensor_data config")

    if "data_root" not in sensor_config:
        raise ConfigValidationError("Missing 'data_root' in sensor_data config")

    if "mode" not in sensor_config:
        raise ConfigValidationError("Missing 'mode' in sensor_data config")

    if sensor_config["mode"] not in ["single_device", "multi_device"]:
        raise ConfigValidationError(
            f"mode must be 'single_device' or 'multi_device', got {sensor_config['mode']}"
        )

    # fine-tuningの場合はユーザー分割が必要
    if mode == "finetune":
        required_keys = ["train_users", "val_users", "test_users"]
        for key in required_keys:
            if key not in sensor_config:
                raise ConfigValidationError(f"Missing '{key}' in sensor_data config")


def validate_data_config(data_config: Dict[str, Any], mode: str) -> None:
    """
    データ設定の検証

    Args:
        data_config: データ設定辞書
        mode: モード ('pretrain' または 'finetune')

    Raises:
        ConfigValidationError: 設定が無効な場合
    """
    # バッチサイズの検証
    if "batch_size" not in data_config:
        raise ConfigValidationError("Missing 'batch_size' in data config")

    if data_config["batch_size"] < 1:
        raise ConfigValidationError(f"batch_size must be >= 1, got {data_config['batch_size']}")

    # トレーニングデータパスの検証
    if "train_path" not in data_config:
        raise ConfigValidationError("Missing 'train_path' in data config")

    # fine-tuningの場合は検証データパスも必要
    if mode == "finetune" and "val_path" not in data_config:
        raise ConfigValidationError("Missing 'val_path' in finetune data config")


def validate_training_config(training_config: Dict[str, Any]) -> None:
    """
    トレーニング設定の検証

    Args:
        training_config: トレーニング設定辞書

    Raises:
        ConfigValidationError: 設定が無効な場合
    """
    # エポック数の検証
    if "epochs" not in training_config:
        raise ConfigValidationError("Missing 'epochs' in training config")

    if training_config["epochs"] < 1:
        raise ConfigValidationError(f"epochs must be >= 1, got {training_config['epochs']}")

    # 学習率の検証
    if "learning_rate" not in training_config:
        raise ConfigValidationError("Missing 'learning_rate' in training config")

    if training_config["learning_rate"] <= 0:
        raise ConfigValidationError(
            f"learning_rate must be > 0, got {training_config['learning_rate']}"
        )

    # オプティマイザーの検証（指定されている場合）
    if "optimizer" in training_config:
        optimizer = training_config["optimizer"].lower()
        if optimizer not in SUPPORTED_OPTIMIZERS:
            raise ConfigValidationError(
                f"Unsupported optimizer: {optimizer}. " f"Supported: {SUPPORTED_OPTIMIZERS}"
            )

    # スケジューラーの検証（指定されている場合）
    if "scheduler" in training_config:
        scheduler = training_config["scheduler"].lower()
        if scheduler not in SUPPORTED_SCHEDULERS:
            raise ConfigValidationError(
                f"Unsupported scheduler: {scheduler}. " f"Supported: {SUPPORTED_SCHEDULERS}"
            )


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    2つの設定をマージ（override_configがbase_configを上書き）

    Args:
        base_config: ベース設定
        override_config: 上書き設定

    Returns:
        マージされた設定
    """
    import copy

    result = copy.deepcopy(base_config)

    for key, value in override_config.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    ドット記法でネストされた設定値を取得

    Args:
        config: 設定辞書
        key_path: キーパス（例: 'model.backbone'）
        default: デフォルト値

    Returns:
        設定値、または見つからない場合はdefault
    """
    keys = key_path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value
