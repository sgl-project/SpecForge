#!/bin/bash
# EAGLE3 online training for Qwen3-8B on AMD GPU (ROCm).
#
# ROCm-safe baseline: HF target backend + flex_attention, single GPU / data parallel.
# The `fa` / `usp` attention backends and yunchang sequence parallel are not available
# on ROCm (they require a CUDA flash-attn build); see docs/get_started/installation.md.
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
    "$ROOT_DIR/scripts/train_eagle3.py" \
    --target-model-path "$TARGET_MODEL_PATH" \
    --draft-model-config "$ROOT_DIR/configs/qwen3-8b-eagle3.json" \
    --train-data-path "$TRAIN_DATA_PATH" \
    --output-dir "$ROOT_DIR/outputs/qwen3-8b-eagle3-rocm" \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template qwen \
    --cache-dir "$ROOT_DIR/cache" \
    --embedding-key model.embed_tokens.weight \
    --tp-size 1 \
    --attention-backend "$ATTENTION_BACKEND" \
    --target-model-backend hf \
    --report-to tensorboard
