#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

NUM_GPUS=${1:-1}
TP_SIZE=${2:-1}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path microsoft/phi-4 \
    --draft-model-config $ROOT_DIR/configs/phi4-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/phi4-eagle3-sharegpt \
    --num-epochs 10 \
    --batch-size 1 \
    --tp-size $TP_SIZE \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template phi4 \
    --cache-dir $ROOT_DIR/cache \
    --target-model-backend sglang \
    --embedding-key model.embed_tokens.weight
