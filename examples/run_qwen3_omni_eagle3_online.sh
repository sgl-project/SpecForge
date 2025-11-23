#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# support tp1 train eagle3 for Qwen3-Omni-30B-A3B-Instruct
NUM_GPUS=${1:-1}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online.py \
    --target-model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --draft-model-config $ROOT_DIR/configs/qwen3-omni-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/allava4v_train.jsonl \
    --output-dir $ROOT_DIR/outputs/Qwen3-Omni-30B-A3B-eagle3 \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 8192 \
    --dist-timeout 360 \
    --chat-template qwen3-omni \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key thinker.model.embed_tokens.weight \
    --tp-size 1 \
    --is-vlm \
    --target-model-backend hf \
    --min-pixels 50176 \
    --max-pixels 200704
