#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")

NUM_GPUS=${NUM_GPUS:-${1:-8}}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-${2:-flex_attention}}
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-Qwen/Qwen3-8B}
DRAFT_CONFIG_PATH=${DRAFT_CONFIG_PATH:-$ROOT_DIR/configs/qwen3-8b-dflash.json}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-$ROOT_DIR/cache/dataset/perfectblend_qwen3-8b_regen.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/outputs/qwen3-8b-dpace}
CACHE_DIR=${CACHE_DIR:-$ROOT_DIR/cache}
DPACE_ALPHA=${DPACE_ALPHA:-0.5}

export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/cache/compiled_kernels}
export SPECFORGE_DATA_NUM_PROC=${SPECFORGE_DATA_NUM_PROC:-32}
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

torchrun \
    --standalone \
    --nproc_per_node "$NUM_GPUS" \
    "$ROOT_DIR/scripts/train_dflash.py" \
    --target-model-path "$TARGET_MODEL_PATH" \
    --target-model-backend sglang \
    --draft-config-path "$DRAFT_CONFIG_PATH" \
    --train-data-path "$TRAIN_DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --num-epochs 6 \
    --batch-size 4 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 3072 \
    --chat-template qwen \
    --attention-backend "$ATTENTION_BACKEND" \
    --block-size 16 \
    --num-draft-layers 1 \
    --num-anchors 512 \
    --loss-type dpace \
    --dpace-alpha "$DPACE_ALPHA" \
    --log-interval 50 \
    --save-interval 1000 \
    --report-to wandb \
    --wandb-project "${WANDB_PROJECT:-dpace-qwen3-8b}" \
    --wandb-name "${WANDB_NAME:-qwen3-8b-dpace}"
