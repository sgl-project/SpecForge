#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=32
NUM_GPUS=${1:-8}

ATTENTION_BACKEND=${2:-flex_attention}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path /disk3/wjp/pretrained_models/Qwen3-8B \
    --draft-config-path $ROOT_DIR/configs/qwen3-8b-dflash.json \
    --train-data-path /disk3/wjp/datasets/nes/all_1w.jsonl \
    --output-dir $ROOT_DIR/outputs/qwen3-8b-perfectblend \
    --num-epochs 6 \
    --batch-size 1 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 3072 \
    --chat-template qwen \
    --attention-backend $ATTENTION_BACKEND \
    --loss-decay-gamma 7.0 \
    --log-interval 50 \
    --save-interval 1000 \
    --report-to wandb \
    --wandb-project specforge-qwen3-8b-dflash \
    --target-model-backend sglang \
    --block-size 16 \
    --num-anchors 512 \
    --wandb-name qwen3-8b-dflash-perfectblend
