#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# support tp4/tp8 train eagle3 for Qwen3-30B-A3B
NUM_GPUS=4 # ${1:-4}
TP_SIZE=4 # ${2:-4}
BUILD_DATASET_NUM_PROC=64 # ${BUILD_DATASET_NUM_PROC:-64}

GLOBAL_BATCH_SIZE=16
GA=$(($GLOBAL_BATCH_SIZE / 1 / $NUM_GPUS))
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --draft-model-config $ROOT_DIR/configs/qwen3-30B-A3B-eagle3-moe-draft.json \
    --train-data-path $ROOT_DIR/cache/dataset/qwen3-a30b-sharegpt-regen.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/qwen3-30b-a3b-instruct-eagle3-sharegpt-moe-draft \
    --num-epochs 10 \
    --batch-size 1 \
    --draft-accumulation-steps $GA \
    --learning-rate 5e-5 \
    --max-length 4096 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $TP_SIZE \
    --target-model-backend sglang --save-interval 30115 --eval-interval 30115
