#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=32
NUM_GPUS=${1:-1}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path Qwen/Qwen3-8B \
    --draft-config-path $ROOT_DIR/configs/qwen3-8b-dflash.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --output-dir $ROOT_DIR/outputs/qwen3-8b-dflash-sharegpt \
    --num-epochs 20 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template qwen \
    --log-interval 50 \
    --save-interval 1000 \
    --report-to wandb \
    --wandb-project specforge-qwen3-8b-dflash \
    --wandb-name qwen3-8b-dflash-sharegpt
