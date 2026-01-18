#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# support tp1 train eagle3 for qwen2.5-vl-7b-instruct
NUM_GPUS=${1:-1}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path Qwen/Qwen2.5-VL-32B-Instruct \
    --draft-model-config $ROOT_DIR/configs/qwen2.5-vl-32b-eagle3.json \
    --train-data-path $ROOT_DIR/cache/allava4v_train.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/qwen2.5-vl-32b-eagle3 \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --dist-timeout 360 \
    --chat-template qwen2-vl \
    --target-model-backend sglang \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size 4 \
    --sglang-mem-fraction-static 0.5 \
    --is-vlm \
    --torch-dtype bfloat16 \
    --min-pixels 200704 \
    --max-pixels 1003520 
