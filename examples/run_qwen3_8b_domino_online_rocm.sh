#!/bin/bash
# Domino online training for Qwen3-8B on AMD GPU (ROCm).
#
# ROCm-safe baseline: HF target backend + flex_attention, single GPU / data parallel.
# Domino needs no ROCm-specific code changes; its bf16 GRU projector runs on ROCm
# (MIOpen), so the NPU fp16-GRU workaround is not required. See
# docs/get_started/installation.md.
#
# Override the model / data paths via environment variables if needed:
#   TARGET_MODEL_PATH  Target model (default: Qwen/Qwen3-8B)
#   TRAIN_DATA_PATH    Training jsonl (default: cache/dataset/sharegpt_train.jsonl)

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

NUM_GPUS=${1:-1}
ATTENTION_BACKEND=${2:-flex_attention}
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-Qwen/Qwen3-8B}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-$ROOT_DIR/cache/dataset/sharegpt_train.jsonl}

python -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$NUM_GPUS" \
    "$ROOT_DIR/scripts/train_domino.py" \
    --target-model-path "$TARGET_MODEL_PATH" \
    --draft-config-path "$ROOT_DIR/configs/qwen3-8b-domino.json" \
    --train-data-path "$TRAIN_DATA_PATH" \
    --output-dir "$ROOT_DIR/outputs/qwen3-8b-domino-rocm" \
    --num-epochs 6 \
    --batch-size 1 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 3072 \
    --chat-template qwen \
    --cache-dir "$ROOT_DIR/cache" \
    --attention-backend "$ATTENTION_BACKEND" \
    --target-model-backend hf \
    --block-size 16 \
    --num-anchors 256 \
    --loss-decay-gamma 7.0 \
    --lambda-base-start 1.0 \
    --lambda-base-decay-ratio 1.0 \
    --log-interval 50 \
    --save-interval 2000 \
    --report-to tensorboard
