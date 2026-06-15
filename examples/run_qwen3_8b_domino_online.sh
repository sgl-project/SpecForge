#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

export CUDA_HOME=/usr/local/cuda
export CONDA_PREFIX=${CONDA_PREFIX:-$HOME/.conda/envs/specforge}

export PATH=$CUDA_HOME/bin:$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export NCCL_DEBUG=WARN
export SPECFORGE_DATA_NUM_PROC=32

NUM_GPUS=${1:-8}
ATTENTION_BACKEND=${2:-flex_attention}
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-/path/to/Qwen3-8B}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-/path/to/sharegpt_train.jsonl}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
     $ROOT_DIR/scripts/train_domino.py \
    --target-model-path $TARGET_MODEL_PATH \
    --draft-config-path $ROOT_DIR/configs/qwen3-8b-domino.json \
    --train-data-path $TRAIN_DATA_PATH \
    --output-dir $ROOT_DIR/outputs/same_data/qwen3-8b-domino_sharegpt \
    --num-epochs 6 \
    --batch-size 2 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 3072 \
    --chat-template qwen \
    --attention-backend $ATTENTION_BACKEND \
    --num-anchors 256 \
    --loss-decay-gamma 7.0 \
    --log-interval 50 \
    --save-interval 2000 \
    --report-to wandb \
    --wandb-project specforge-qwen3-8b-domino \
    --target-model-backend sglang \
    --block-size 16 \
    --lambda-base-start 1.0 \
    --lambda-base-decay-ratio 1.0 \
    --wandb-name qwen3-8b-domino_sharegpt
