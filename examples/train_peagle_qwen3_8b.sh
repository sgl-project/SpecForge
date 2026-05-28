#!/bin/bash
# Example: P-EAGLE training on Qwen3-8B with ShareGPT dataset

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export TORCHINDUCTOR_CACHE_DIR="$ROOT_DIR/cache/compiled_kernels"
export FLASHINFER_WORKSPACE_BASE="$ROOT_DIR/cache/flashinfer"

# Model
TARGET_MODEL="${TARGET_MODEL:-/home/share/model_weight/qwen/qwen3-8b}"

# P-EAGLE parameters
NUM_DEPTHS=4
DOWN_SAMPLE_RATIO=0.7
DOWN_SAMPLE_RATIO_MIN=0.2
NUM_DRAFT_LAYERS=4

# Training
NUM_EPOCHS=10
BATCH_SIZE=1
LEARNING_RATE=6e-4
MAX_LENGTH=2048
WARMUP_RATIO=0.015
MAX_GRAD_NORM=0.5

# Data
TRAIN_DATA_PATH="$ROOT_DIR/cache/dataset/sharegpt_train.jsonl"
CHAT_TEMPLATE="qwen"

# Output
OUTPUT_DIR="$ROOT_DIR/outputs/peagle_qwen3_8b"
DRAFT_CONFIG="$ROOT_DIR/configs/qwen3-8b-peagle.json"
TORCHRUN="$ROOT_DIR/.venv/bin/torchrun"
if [ ! -x "$TORCHRUN" ]; then
    TORCHRUN="torchrun"
fi

"$TORCHRUN" \
    --standalone \
    --nproc_per_node=2 \
    "$ROOT_DIR/scripts/train_peagle.py" \
    --target-model-path "$TARGET_MODEL" \
    --draft-model-config "$DRAFT_CONFIG" \
    --train-data-path "$TRAIN_DATA_PATH" \
    --chat-template "$CHAT_TEMPLATE" \
    --output-dir "$OUTPUT_DIR" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --max-length "$MAX_LENGTH" \
    --warmup-ratio "$WARMUP_RATIO" \
    --max-grad-norm "$MAX_GRAD_NORM" \
    --num-depths "$NUM_DEPTHS" \
    --down-sample-ratio "$DOWN_SAMPLE_RATIO" \
    --down-sample-ratio-min "$DOWN_SAMPLE_RATIO_MIN" \
    --num-draft-layers "$NUM_DRAFT_LAYERS" \
    --no-norm-before-residual \
    --target-model-backend sglang \
    --save-interval 5000 \
    --eval-interval 5000 \
    --log-interval 1 \
    --report-to wandb
