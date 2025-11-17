#!/bin/bash
# 全ウィンドウサイズの特徴抽出を一括実行

set -e  # エラー時に停止

echo "=========================================="
echo "Feature Extraction for All Window Sizes"
echo "=========================================="
echo ""

# 出力ディレクトリ
OUTPUT_DIR="analysis/embedding_explorer/data"
mkdir -p "$OUTPUT_DIR"

# 共通パラメータ
MAX_SAMPLES=100
MAX_USERS=20
DEVICE="cuda"

# モデルパス（実験ディレクトリから最新のepochを探す）
echo "🔍 Searching for model checkpoints..."
echo ""

# 5.0s (150 samples)
MODEL_5_0S=$(find experiments/pretrain/*/exp_2/models/ -name "checkpoint_epoch_*.pth" 2>/dev/null | grep -v "exp_[0-1]" | sort -V | tail -1)
if [ -z "$MODEL_5_0S" ]; then
    echo "❌ No model found for 5.0s (window_size=150)"
    MODEL_5_0S="experiments/pretrain/run_20251111_171703/exp_2/models/checkpoint_epoch_45.pth"
    echo "   Using fallback: $MODEL_5_0S"
fi
echo "✓ 5.0s model: $MODEL_5_0S"

# 2.0s (60 samples)
MODEL_2_0S=$(find experiments/pretrain/*/exp_0/models/ -name "checkpoint_epoch_*.pth" 2>/dev/null | grep -E "run_[0-9]{8}_[0-9]{6}" | sort -V | tail -1)
if [ -z "$MODEL_2_0S" ]; then
    echo "❌ No model found for 2.0s (window_size=60)"
    MODEL_2_0S="experiments/pretrain/run_20251112_192545/exp_0/models/checkpoint_epoch_40.pth"
    echo "   Using fallback: $MODEL_2_0S"
fi
echo "✓ 2.0s model: $MODEL_2_0S"

# 1.0s (30 samples)
MODEL_1_0S=$(find experiments/pretrain/*/exp_1/models/ -name "checkpoint_epoch_*.pth" 2>/dev/null | grep -E "run_[0-9]{8}_[0-9]{6}" | sort -V | tail -1)
if [ -z "$MODEL_1_0S" ]; then
    echo "❌ No model found for 1.0s (window_size=30)"
    MODEL_1_0S="experiments/pretrain/run_20251112_192545/exp_1/models/checkpoint_epoch_40.pth"
    echo "   Using fallback: $MODEL_1_0S"
fi
echo "✓ 1.0s model: $MODEL_1_0S"

# 0.5s (15 samples)
MODEL_0_5S=$(find experiments/pretrain/*/exp_2/models/ -name "checkpoint_epoch_*.pth" 2>/dev/null | grep -E "run_[0-9]{8}_[0-9]{6}" | grep -E "192545|later" | sort -V | tail -1)
if [ -z "$MODEL_0_5S" ]; then
    echo "❌ No model found for 0.5s (window_size=15)"
    MODEL_0_5S="experiments/pretrain/run_20251112_192545/exp_2/models/checkpoint_epoch_39.pth"
    echo "   Using fallback: $MODEL_0_5S"
fi
echo "✓ 0.5s model: $MODEL_0_5S"

echo ""
echo "=========================================="
echo "Starting feature extraction..."
echo "=========================================="
echo ""

# 5.0s (150 samples) の特徴抽出
echo "📊 [1/4] Extracting features for 5.0s (150 samples)..."
if [ -f "$MODEL_5_0S" ]; then
    python analysis/embedding_explorer/extract_features.py \
        --model "$MODEL_5_0S" \
        --max-samples $MAX_SAMPLES \
        --max-users $MAX_USERS \
        --output-dir "$OUTPUT_DIR" \
        --device "$DEVICE"
    echo "✓ 5.0s features extracted"
else
    echo "❌ Model file not found: $MODEL_5_0S"
    exit 1
fi
echo ""

# 2.0s (60 samples) の特徴抽出
echo "📊 [2/4] Extracting features for 2.0s (60 samples)..."
if [ -f "$MODEL_2_0S" ]; then
    python analysis/embedding_explorer/extract_features.py \
        --model "$MODEL_2_0S" \
        --max-samples $MAX_SAMPLES \
        --max-users $MAX_USERS \
        --output-dir "$OUTPUT_DIR" \
        --device "$DEVICE"
    echo "✓ 2.0s features extracted"
else
    echo "❌ Model file not found: $MODEL_2_0S"
    exit 1
fi
echo ""

# 1.0s (30 samples) の特徴抽出
echo "📊 [3/4] Extracting features for 1.0s (30 samples)..."
if [ -f "$MODEL_1_0S" ]; then
    python analysis/embedding_explorer/extract_features.py \
        --model "$MODEL_1_0S" \
        --max-samples $MAX_SAMPLES \
        --max-users $MAX_USERS \
        --output-dir "$OUTPUT_DIR" \
        --device "$DEVICE"
    echo "✓ 1.0s features extracted"
else
    echo "❌ Model file not found: $MODEL_1_0S"
    exit 1
fi
echo ""

# 0.5s (15 samples) の特徴抽出
echo "📊 [4/4] Extracting features for 0.5s (15 samples)..."
if [ -f "$MODEL_0_5S" ]; then
    python analysis/embedding_explorer/extract_features.py \
        --model "$MODEL_0_5S" \
        --max-samples $MAX_SAMPLES \
        --max-users $MAX_USERS \
        --output-dir "$OUTPUT_DIR" \
        --device "$DEVICE"
    echo "✓ 0.5s features extracted"
else
    echo "❌ Model file not found: $MODEL_0_5S"
    exit 1
fi
echo ""

echo "=========================================="
echo "✓ All features extracted successfully!"
echo "=========================================="
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"/*.npz "$OUTPUT_DIR"/*.json 2>/dev/null || echo "No files found"
echo ""
echo "Next step: Start the server with:"
echo "  python analysis/embedding_explorer/server.py --port 8050 --debug"
