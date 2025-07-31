#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# support tp4/tp8 train eagle3 for Qwen3-30B-A3B
NUM_GPUS=${1:-8}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online.py \
    --target-model-path Qwen/Qwen3-30B-A3B \
    --draft-model-config $ROOT_DIR/configs/qwen3-30B-A3B-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt.jsonl \
    --output-dir $ROOT_DIR/outputs/Qwen3-30B-A3B-eagle3 \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $NUM_GPUS \
    --ttt-length 7
