#!/bin/bash
# 特徴抽出スクリプト
# Orientation Invarianceタスクあり/なしの両方のモデルから特徴抽出

set -e  # エラー時に停止

echo "=========================================="
echo "Feature Extraction (Multi-Model)"
echo "=========================================="
echo ""

# 共通パラメータ
MAX_SAMPLES=200
MAX_USERS=5
DEVICE="cuda"

# 実験ディレクトリ
PRETRAIN_DIR_WITH_ORI="experiments/pretrain/run_20251117_074851"  # Orientationあり
PRETRAIN_DIR_NO_ORI="experiments/pretrain/run_20251111_171703"     # Orientationなし

echo "Experiments:"
echo "  [WITH Orientation] $PRETRAIN_DIR_WITH_ORI"
echo "    - binary_permute, binary_reverse, binary_timewarp, invariant_orientation"
echo ""
echo "  [NO Orientation]   $PRETRAIN_DIR_NO_ORI"
echo "    - binary_permute, binary_reverse, binary_timewarp"
echo ""

# モデルとモデル名のマッピング
# WITH Orientation (run_20251117_074851):
#   exp_0: window_size=60 (2.0s) - 19.2MB
#   exp_1: window_size=30 (1.0s) - 19.2MB
#   exp_2: window_size=15 (0.5s) - 12.7MB
# NO Orientation (run_20251111_171703):
#   exp_0: window_size=60 (2.0s) - 55.6MB
#   exp_1: window_size=30 (1.0s) - 55.6MB
#   exp_2: window_size=15 (0.5s) - 55.6MB

declare -A MODEL_MAPPING=(
    # NO Orientation
    ["2.0s"]="$PRETRAIN_DIR_NO_ORI/exp_0/models/best_model.pth"
    ["1.0s"]="$PRETRAIN_DIR_NO_ORI/exp_1/models/best_model.pth"
    ["0.5s"]="$PRETRAIN_DIR_NO_ORI/exp_2/models/best_model.pth"

    # WITH Orientation
    ["2.0s_ori"]="$PRETRAIN_DIR_WITH_ORI/exp_0/models/best_model.pth"
    ["1.0s_ori"]="$PRETRAIN_DIR_WITH_ORI/exp_1/models/best_model.pth"
    ["0.5s_ori"]="$PRETRAIN_DIR_WITH_ORI/exp_2/models/best_model.pth"
)

declare -A MODEL_DESCRIPTIONS=(
    # NO Orientation
    ["2.0s"]="2.0s window NO orientation (run_20251111)"
    ["1.0s"]="1.0s window NO orientation (run_20251111)"
    ["0.5s"]="0.5s window NO orientation (run_20251111)"

    # WITH Orientation
    ["2.0s_ori"]="2.0s window WITH orientation (run_20251117)"
    ["1.0s_ori"]="1.0s window WITH orientation (run_20251117)"
    ["0.5s_ori"]="0.5s window WITH orientation (run_20251117)"
)

# 実行順序
MODEL_NAMES=("2.0s" "1.0s" "0.5s" "2.0s_ori" "1.0s_ori" "0.5s_ori")

echo "=========================================="
echo "Starting feature extraction..."
echo "=========================================="
echo ""

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    MODEL_PATH="${MODEL_MAPPING[$MODEL_NAME]}"
    DESCRIPTION="${MODEL_DESCRIPTIONS[$MODEL_NAME]}"
    NUM=$((i + 1))
    TOTAL=${#MODEL_NAMES[@]}

    echo "📊 [$NUM/$TOTAL] $DESCRIPTION"
    echo "---"

    # モデルファイルの存在確認
    if [ ! -f "$MODEL_PATH" ]; then
        echo "❌ Model file not found: $MODEL_PATH"
        echo ""
        continue
    fi

    # ファイル情報表示
    FILE_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    FILE_DATE=$(stat -c %y "$MODEL_PATH" | cut -d. -f1)
    echo "Model: $(basename $MODEL_PATH)"
    echo "Path:  $MODEL_PATH"
    echo "Size:  $FILE_SIZE"
    echo "Date:  $FILE_DATE"
    echo ""

    # 特徴抽出実行
    python analysis/embedding_explorer/extract_model_features.py \
        --model-path "$MODEL_PATH" \
        --model-name "$MODEL_NAME" \
        --max-samples $MAX_SAMPLES \
        --max-users $MAX_USERS \
        --device "$DEVICE"

    echo ""
    echo "✓ $MODEL_NAME features extracted successfully"
    echo ""
done

echo "=========================================="
echo "✓ All features extracted!"
echo "=========================================="
echo ""
echo "Output files:"
ls -lh analysis/embedding_explorer/data/*.npz analysis/embedding_explorer/data/*.json 2>/dev/null | \
    awk '{printf "  %s  %s\n", $9, $5}' || echo "  No files found"
echo ""
echo "Next step: Start the visualization server"
echo "  python analysis/embedding_explorer/server.py --port 5000"
