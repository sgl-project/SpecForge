#!/bin/bash
# Domino training for Qwen3.5-4B on Ascend NPU
# Backend: HF + SDPA attention + HCCL distributed
#
# Required environment variables (override before invoking the script):
#   TARGET_MODEL_PATH  Path to Qwen3.5-4B weights
#   TRAIN_DATA_PATH    Path to the training jsonl

set -eu

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

: "${TARGET_MODEL_PATH:?Set TARGET_MODEL_PATH to the Qwen3.5-4B weights directory}"
: "${TRAIN_DATA_PATH:?Set TRAIN_DATA_PATH to the training jsonl file}"

NPU_DEVICES=${1:-0,1,2,3,4,5,6,7}
NUM_DEVICES=$(echo "$NPU_DEVICES" | tr ',' '\n' | wc -l)

export ASCEND_RT_VISIBLE_DEVICES=$NPU_DEVICES
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:32

torchrun \
    --standalone \
    --nproc_per_node "$NUM_DEVICES" \
    "$ROOT_DIR/scripts/train_domino.py" \
    --target-model-path "$TARGET_MODEL_PATH" \
    --draft-config-path "$ROOT_DIR/configs/qwen3.5-4b-domino.json" \
    --train-data-path "$TRAIN_DATA_PATH" \
    --output-dir "$ROOT_DIR/outputs/qwen3.5-4b-domino-npu" \
    --num-epochs 10 \
    --batch-size 1 \
    --accumulation-steps 4 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 1024 \
    --chat-template qwen3.5 \
    --attention-backend sdpa \
    --num-anchors 16 \
    --loss-decay-gamma 7.0 \
    --log-interval 50 \
    --save-interval 3000 \
    --report-to tensorboard \
    --target-model-backend hf \
    --block-size 16 \
    --embedding-key model.language_model.embed_tokens.weight \
    --trust-remote-code
