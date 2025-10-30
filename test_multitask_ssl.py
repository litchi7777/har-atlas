"""
マルチタスクSSL実装の統合テスト

変換予測タスク（binary_*）とマスク再構成タスク（masking_*）の両方をテスト
"""

import torch
import numpy as np
from src.models.sensor_models import IntegratedSSLModel, Resnet
from src.losses.ssl_losses import IntegratedSSLLoss
from src.data.augmentations import (
    Permutation,
    Reverse,
    TimeWarping,
    ChannelMasking,
    TimeMasking,
    TimeChannelMasking,
)


def test_binary_tasks():
    """Binary拡張タスクのテスト"""
    print("\n" + "=" * 80)
    print("Testing Binary Tasks (binary_permute, binary_reverse, binary_timewarp)")
    print("=" * 80)

    # パラメータ
    batch_size = 8
    n_channels = 3
    sequence_length = 150
    ssl_tasks = ["binary_permute", "binary_reverse", "binary_timewarp"]

    # モデル作成
    backbone = Resnet(n_channels=n_channels, foundationUK=False)
    model = IntegratedSSLModel(
        backbone=backbone,
        ssl_tasks=ssl_tasks,
        hidden_dim=256,
        n_channels=n_channels,
        sequence_length=sequence_length,
    )

    # 損失関数作成
    task_weights = {task: 1.0 for task in ssl_tasks}
    criterion = IntegratedSSLLoss(ssl_tasks=ssl_tasks, task_weights=task_weights)

    # ダミーデータ作成
    x = torch.randn(batch_size, n_channels, sequence_length)

    # 各タスクについて順伝播
    predictions = {}
    labels = {}
    for task in ssl_tasks:
        pred = model(x, task)
        predictions[task] = pred
        # Binary分類なのでランダムなラベル (0 or 1)
        labels[task] = torch.randint(0, 2, (batch_size,))

        print(f"\nTask: {task}")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {pred.shape}")
        print(f"  Expected: ({batch_size}, 2)")
        assert pred.shape == (batch_size, 2), f"Unexpected output shape: {pred.shape}"

    # 損失計算
    total_loss, task_losses = criterion(predictions, labels)

    print(f"\nTotal loss: {total_loss.item():.4f}")
    for task, loss in task_losses.items():
        print(f"  {task}: {loss.item():.4f}")

    print("\n✅ Binary tasks test passed!")


def test_masking_tasks():
    """Maskingタスクのテスト"""
    print("\n" + "=" * 80)
    print("Testing Masking Tasks (masking_channel, masking_time, masking_time_channel)")
    print("=" * 80)

    # パラメータ
    batch_size = 8
    n_channels = 3
    sequence_length = 150
    ssl_tasks = ["masking_channel", "masking_time", "masking_time_channel"]

    # モデル作成
    backbone = Resnet(n_channels=n_channels, foundationUK=False)
    model = IntegratedSSLModel(
        backbone=backbone,
        ssl_tasks=ssl_tasks,
        hidden_dim=256,
        n_channels=n_channels,
        sequence_length=sequence_length,
    )

    # 損失関数作成
    task_weights = {task: 1.0 for task in ssl_tasks}
    criterion = IntegratedSSLLoss(ssl_tasks=ssl_tasks, task_weights=task_weights)

    # ダミーデータ作成
    x_original = torch.randn(batch_size, n_channels, sequence_length)

    # マスキング拡張のテスト
    mask_ratio = 0.15
    channel_masking = ChannelMasking(mask_ratio=mask_ratio)
    time_masking = TimeMasking(mask_ratio=mask_ratio)
    time_channel_masking = TimeChannelMasking(
        time_mask_ratio=mask_ratio, channel_mask_ratio=mask_ratio
    )

    # 各タスクについて順伝播
    predictions = {}
    labels = {}

    # Channel masking
    x_channel_masked = x_original.clone()
    for i in range(batch_size):
        masked, _ = channel_masking(x_original[i].numpy())
        x_channel_masked[i] = torch.from_numpy(masked)

    pred_channel = model(x_channel_masked, "masking_channel")
    predictions["masking_channel"] = pred_channel
    labels["masking_channel"] = x_original  # 元データを再構成

    print(f"\nTask: masking_channel")
    print(f"  Masked input shape: {x_channel_masked.shape}")
    print(f"  Output shape: {pred_channel.shape}")
    print(f"  Expected: {x_original.shape}")
    assert pred_channel.shape == x_original.shape, f"Unexpected output shape: {pred_channel.shape}"

    # Time masking
    x_time_masked = x_original.clone()
    for i in range(batch_size):
        masked, _ = time_masking(x_original[i].numpy())
        x_time_masked[i] = torch.from_numpy(masked)

    pred_time = model(x_time_masked, "masking_time")
    predictions["masking_time"] = pred_time
    labels["masking_time"] = x_original

    print(f"\nTask: masking_time")
    print(f"  Masked input shape: {x_time_masked.shape}")
    print(f"  Output shape: {pred_time.shape}")
    print(f"  Expected: {x_original.shape}")
    assert pred_time.shape == x_original.shape, f"Unexpected output shape: {pred_time.shape}"

    # Time-channel masking
    x_tc_masked = x_original.clone()
    for i in range(batch_size):
        masked, _ = time_channel_masking(x_original[i].numpy())
        x_tc_masked[i] = torch.from_numpy(masked)

    pred_tc = model(x_tc_masked, "masking_time_channel")
    predictions["masking_time_channel"] = pred_tc
    labels["masking_time_channel"] = x_original

    print(f"\nTask: masking_time_channel")
    print(f"  Masked input shape: {x_tc_masked.shape}")
    print(f"  Output shape: {pred_tc.shape}")
    print(f"  Expected: {x_original.shape}")
    assert pred_tc.shape == x_original.shape, f"Unexpected output shape: {pred_tc.shape}"

    # 損失計算
    total_loss, task_losses = criterion(predictions, labels)

    print(f"\nTotal loss: {total_loss.item():.4f}")
    for task, loss in task_losses.items():
        print(f"  {task}: {loss.item():.4f}")

    print("\n✅ Masking tasks test passed!")


def test_mixed_tasks():
    """Binary + Maskingの混合タスクのテスト"""
    print("\n" + "=" * 80)
    print("Testing Mixed Tasks (binary_permute + masking_channel)")
    print("=" * 80)

    # パラメータ
    batch_size = 8
    n_channels = 3
    sequence_length = 150
    ssl_tasks = ["binary_permute", "masking_channel"]

    # モデル作成
    backbone = Resnet(n_channels=n_channels, foundationUK=False)
    model = IntegratedSSLModel(
        backbone=backbone,
        ssl_tasks=ssl_tasks,
        hidden_dim=256,
        n_channels=n_channels,
        sequence_length=sequence_length,
    )

    # 損失関数作成
    task_weights = {task: 1.0 for task in ssl_tasks}
    criterion = IntegratedSSLLoss(ssl_tasks=ssl_tasks, task_weights=task_weights)

    # ダミーデータ作成
    x_original = torch.randn(batch_size, n_channels, sequence_length)

    # Permute拡張を適用
    permutation = Permutation(n_segments=4)
    x_permuted = x_original.clone()
    for i in range(batch_size):
        x_permuted[i] = torch.from_numpy(permutation(x_original[i].numpy()))

    # Channel masking適用
    channel_masking = ChannelMasking(mask_ratio=0.15)
    x_masked = x_permuted.clone()
    for i in range(batch_size):
        masked, _ = channel_masking(x_permuted[i].numpy())
        x_masked[i] = torch.from_numpy(masked)

    # 各タスクについて順伝播
    predictions = {}
    labels = {}

    # Binary task
    pred_binary = model(x_masked, "binary_permute")
    predictions["binary_permute"] = pred_binary
    labels["binary_permute"] = torch.ones(batch_size, dtype=torch.long)  # Permute適用済み

    print(f"\nTask: binary_permute")
    print(f"  Input shape: {x_masked.shape}")
    print(f"  Output shape: {pred_binary.shape}")
    print(f"  Expected: ({batch_size}, 2)")
    assert pred_binary.shape == (batch_size, 2)

    # Masking task
    pred_masking = model(x_masked, "masking_channel")
    predictions["masking_channel"] = pred_masking
    labels["masking_channel"] = x_original  # 元データを再構成

    print(f"\nTask: masking_channel")
    print(f"  Input shape: {x_masked.shape}")
    print(f"  Output shape: {pred_masking.shape}")
    print(f"  Expected: {x_original.shape}")
    assert pred_masking.shape == x_original.shape

    # 損失計算
    total_loss, task_losses = criterion(predictions, labels)

    print(f"\nTotal loss: {total_loss.item():.4f}")
    for task, loss in task_losses.items():
        print(f"  {task}: {loss.item():.4f}")

    print("\n✅ Mixed tasks test passed!")


def test_augmentations():
    """拡張クラスのテスト"""
    print("\n" + "=" * 80)
    print("Testing Augmentation Classes")
    print("=" * 80)

    # サンプルデータ
    x = np.random.randn(3, 150).astype(np.float32)

    # Channel masking
    channel_masking = ChannelMasking(mask_ratio=0.33)
    x_masked, masked_channels = channel_masking(x)
    print(f"\nChannelMasking:")
    print(f"  Original shape: {x.shape}")
    print(f"  Masked shape: {x_masked.shape}")
    print(f"  Masked channels: {masked_channels}")
    print(f"  Number of masked channels: {len(masked_channels)}")
    assert len(masked_channels) == 1  # 33% of 3 channels = 1 channel

    # Time masking
    time_masking = TimeMasking(mask_ratio=0.2)
    x_masked, masked_timesteps = time_masking(x)
    print(f"\nTimeMasking:")
    print(f"  Original shape: {x.shape}")
    print(f"  Masked shape: {x_masked.shape}")
    print(f"  Number of masked timesteps: {len(masked_timesteps)}")
    assert len(masked_timesteps) == int(150 * 0.2)  # 20% of 150 = 30 timesteps

    # Time-channel masking
    tc_masking = TimeChannelMasking(time_mask_ratio=0.2, channel_mask_ratio=0.33)
    x_masked, (masked_timesteps, masked_channels) = tc_masking(x)
    print(f"\nTimeChannelMasking:")
    print(f"  Original shape: {x.shape}")
    print(f"  Masked shape: {x_masked.shape}")
    print(f"  Number of masked timesteps: {len(masked_timesteps)}")
    print(f"  Number of masked channels: {len(masked_channels)}")

    print("\n✅ Augmentation classes test passed!")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Multi-Task SSL Implementation Test")
    print("=" * 80)

    try:
        # 拡張クラスのテスト
        test_augmentations()

        # Binary拡張タスクのテスト
        test_binary_tasks()

        # Maskingタスクのテスト
        test_masking_tasks()

        # 混合タスクのテスト
        test_mixed_tasks()

        print("\n" + "=" * 80)
        print("🎉 All tests passed!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
