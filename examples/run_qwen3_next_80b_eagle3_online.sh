#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

NUM_GPUS=${1:-8}
TP_SIZE=4
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path $ROOT_DIR//Qwen/Qwen3-Next-80B-A3B-Instruct-FP8/\
    --draft-model-config $ROOT_DIR/configs/qwen3-next-80b-a3b-eagle3.json \
    --train-data-path $ROOT_DIR/data_qwen80b/qwen3_80b_perfectblend_train_regen.jsonl \
    --output-dir $ROOT_DIR/qwen3-80b-regen-blend \
    --num-epochs 2 \
    --batch-size 2 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $TP_SIZE \
    --sglang-mem-fraction-static 0.5 \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --target-model-backend sglang
