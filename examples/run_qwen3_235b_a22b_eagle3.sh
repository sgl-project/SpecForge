#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# support tp4/tp8 train eagle3 for Qwen3-30B-A3B
NUM_GPUS=8
TP_SIZE=4
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path /workdir/huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct-FP8/\
    --draft-model-config $ROOT_DIR/configs/qwen3-next-80b-a3b-eagle3.json \
    --train-data-path /workdir/data_qwen80b/qwen3_80b_perfectblend_train_regen.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir /workdir/qwen3-80b-regen-blend \
    --num-epochs 2 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template qwen \
    --cache-dir /workdir/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $TP_SIZE \
    --target-model-backend sglang
