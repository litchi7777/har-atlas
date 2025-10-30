#!/usr/bin/env python
"""
Data Augmentation Test Script

PyTorchベースの拡張機能が正しく動作するかテストします。
"""

import torch
import numpy as np
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.data.augmentations import (
    Jittering,
    Scaling,
    Rotation,
    Amplitude,
    ChannelMixing,
    ChannelArithmetic,
    PhaseShift,
    get_augmentation_pipeline,
)


def test_basic_augmentations():
    """基本拡張のテスト"""
    print("=" * 80)
    print("Testing Basic Augmentations")
    print("=" * 80)

    # テストデータ作成 (channels=6, time_steps=100)
    x_np = np.random.randn(6, 100).astype(np.float32)
    x_torch = torch.from_numpy(x_np)
    x_gpu = x_torch.cuda() if torch.cuda.is_available() else x_torch

    # Jittering
    print("\n1. Testing Jittering...")
    jitter = Jittering(sigma=0.05)
    y_np = jitter(x_np)
    y_torch = jitter(x_torch)
    y_gpu = jitter(x_gpu)
    print(f"   NumPy shape: {y_np.shape}, type: {type(y_np)}")
    print(f"   Torch shape: {y_torch.shape}, type: {type(y_torch)}")
    print(f"   GPU shape: {y_gpu.shape}, device: {y_gpu.device}")
    print("   ✓ Jittering passed")

    # Scaling
    print("\n2. Testing Scaling...")
    scale = Scaling(sigma=0.1)
    y_torch = scale(x_torch)
    y_gpu = scale(x_gpu)
    print(f"   Torch shape: {y_torch.shape}")
    print(f"   GPU shape: {y_gpu.shape}, device: {y_gpu.device}")
    print("   ✓ Scaling passed")

    # Rotation
    print("\n3. Testing Rotation...")
    rotation = Rotation(max_angle=15.0)
    y_torch = rotation(x_torch)
    y_gpu = rotation(x_gpu)
    print(f"   Torch shape: {y_torch.shape}")
    print(f"   GPU shape: {y_gpu.shape}, device: {y_gpu.device}")
    print("   ✓ Rotation passed")


def test_advanced_augmentations():
    """高度な拡張のテスト"""
    print("\n" + "=" * 80)
    print("Testing Advanced Augmentations")
    print("=" * 80)

    x_torch = torch.randn(6, 100)
    x_gpu = x_torch.cuda() if torch.cuda.is_available() else x_torch

    # Amplitude
    print("\n1. Testing Amplitude...")
    amp = Amplitude(sigma=0.2)
    y_torch = amp(x_torch)
    y_gpu = amp(x_gpu)
    print(f"   Torch shape: {y_torch.shape}")
    print(f"   GPU shape: {y_gpu.shape}, device: {y_gpu.device}")
    print("   ✓ Amplitude passed")

    # ChannelMixing
    print("\n2. Testing ChannelMixing...")
    mix = ChannelMixing(alpha=0.5)
    y_torch = mix(x_torch)
    y_gpu = mix(x_gpu)
    print(f"   Torch shape: {y_torch.shape}")
    print(f"   GPU shape: {y_gpu.shape}, device: {y_gpu.device}")
    print("   ✓ ChannelMixing passed")

    # ChannelArithmetic
    print("\n3. Testing ChannelArithmetic...")
    arith = ChannelArithmetic(alpha=0.3)
    y_torch = arith(x_torch)
    y_gpu = arith(x_gpu)
    print(f"   Torch shape: {y_torch.shape}")
    print(f"   GPU shape: {y_gpu.shape}, device: {y_gpu.device}")
    print("   ✓ ChannelArithmetic passed")

    # PhaseShift
    print("\n4. Testing PhaseShift...")
    shift = PhaseShift(max_shift_ratio=0.2)
    y_torch = shift(x_torch)
    y_gpu = shift(x_gpu)
    print(f"   Torch shape: {y_torch.shape}")
    print(f"   GPU shape: {y_gpu.shape}, device: {y_gpu.device}")
    print("   ✓ PhaseShift passed")


def test_augmentation_pipelines():
    """拡張パイプラインのテスト"""
    print("\n" + "=" * 80)
    print("Testing Augmentation Pipelines")
    print("=" * 80)

    x_torch = torch.randn(6, 100)
    x_gpu = x_torch.cuda() if torch.cuda.is_available() else x_torch

    modes = ["light", "medium", "heavy", "mtl"]
    for mode in modes:
        print(f"\n{mode.upper()} mode:")
        pipeline = get_augmentation_pipeline(mode=mode, max_epochs=100)

        # CPU
        y_torch = pipeline(x_torch)
        print(f"   CPU output shape: {y_torch.shape}, type: {type(y_torch)}")

        # GPU
        y_gpu = pipeline(x_gpu)
        print(f"   GPU output shape: {y_gpu.shape}, device: {y_gpu.device}")
        print(f"   ✓ {mode} pipeline passed")


def test_batch_processing():
    """バッチ処理のテスト（pretrainで使用される形式）"""
    print("\n" + "=" * 80)
    print("Testing Batch Processing")
    print("=" * 80)

    batch_size = 4
    channels = 6
    time_steps = 100

    # バッチデータ作成
    x_batch = torch.randn(batch_size, channels, time_steps)
    x_gpu = x_batch.cuda() if torch.cuda.is_available() else x_batch

    # MTLパイプラインでバッチ処理
    pipeline = get_augmentation_pipeline(mode="mtl", max_epochs=100)

    print(f"\nProcessing batch of shape: {x_gpu.shape}")

    # 各サンプルに適用
    augmented_batch = []
    for sample in x_gpu:
        aug_sample = pipeline(sample)
        augmented_batch.append(aug_sample)

    y_batch = torch.stack(augmented_batch)
    print(f"Output batch shape: {y_batch.shape}, device: {y_batch.device}")
    print("✓ Batch processing passed")


def main():
    print("Data Augmentation Test")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))

    try:
        test_basic_augmentations()
        test_advanced_augmentations()
        test_augmentation_pipelines()
        test_batch_processing()

        print("\n" + "=" * 80)
        print("✅ All tests passed!")
        print("=" * 80)
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ Test failed!")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
