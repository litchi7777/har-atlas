#!/bin/bash
# 4つのモデル（window_size × rotation）から特徴量を抽出
# 実験: run_20251120_070522

set -e  # エラーで停止

PROJECT_ROOT="/mnt/home/har-foundation"
RUN_DIR="$PROJECT_ROOT/experiments/pretrain/run_20251120_070522"
SCRIPT="$PROJECT_ROOT/analysis/embedding_explorer/extract_model_features.py"

# CUDA設定
export CUDA_VISIBLE_DEVICES=3

echo "=================================="
echo "Window Size Comparison Experiments"
echo "=================================="
echo "Run: run_20251120_070522"
echo "Device: cuda (GPU 3)"
echo "=================================="
echo ""

# exp_0: window_size=15 (0.5s), rotation=False
echo "[1/4] exp_0: window_size=15 (0.5s), rotation=False"
python "$SCRIPT" \
    --model-path "$RUN_DIR/exp_0/models/best_model.pth" \
    --model-name "ws15_0.5s_norot" \
    --max-samples 200 \
    --device cuda

echo ""

# exp_1: window_size=15 (0.5s), rotation=True
echo "[2/4] exp_1: window_size=15 (0.5s), rotation=True"
python "$SCRIPT" \
    --model-path "$RUN_DIR/exp_1/models/best_model.pth" \
    --model-name "ws15_0.5s_rot" \
    --max-samples 200 \
    --device cuda

echo ""

# exp_2: window_size=60 (2.0s), rotation=False
echo "[3/4] exp_2: window_size=60 (2.0s), rotation=False"
python "$SCRIPT" \
    --model-path "$RUN_DIR/exp_2/models/best_model.pth" \
    --model-name "ws60_2.0s_norot" \
    --max-samples 200 \
    --device cuda

echo ""

# exp_3: window_size=60 (2.0s), rotation=True
echo "[4/4] exp_3: window_size=60 (2.0s), rotation=True"
python "$SCRIPT" \
    --model-path "$RUN_DIR/exp_3/models/best_model.pth" \
    --model-name "ws60_2.0s_rot" \
    --max-samples 200 \
    --device cuda

echo ""
echo "=================================="
echo "✅ All features extracted!"
echo "=================================="
echo ""
echo "次のステップ:"
echo "1. Flaskサーバーを起動: cd analysis/embedding_explorer && python app.py"
echo "2. ブラウザで開く: http://localhost:5001"
echo "3. モデル選択で4つのモデルを比較:"
echo "   - ws15_0.5s_norot  (0.5秒窓、回転なし)"
echo "   - ws15_0.5s_rot    (0.5秒窓、回転あり)"
echo "   - ws60_2.0s_norot  (2.0秒窓、回転なし)"
echo "   - ws60_2.0s_rot    (2.0秒窓、回転あり)"
