#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

NUM_GPUS=${1:-8}
ATTENTION_BACKEND=${2:-flex_attention}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-16}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path Qwen/Qwen3-VL-8B-Instruct \
    --draft-model-config $ROOT_DIR/configs/qwen3-vl-8b-dflash.json \
    --target-model-backend hf \
    --is-vlm \
    --trust-remote-code \
    --train-data-path $ROOT_DIR/cache/dataset/allava4v-mix-20k_train.localimg_regen.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --min-pixels 50176 \
    --max-pixels 1003520 \
    --output-dir $ROOT_DIR/outputs/qwen3-vl-8b-allava4v20k-dflash \
    --cache-dir $ROOT_DIR/cache \
    --num-epochs 6 \
    --batch-size 2 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 4096 \
    --num-draft-layers 5 \
    --chat-template qwen3-vl \
    --attention-backend $ATTENTION_BACKEND \
    --block-size 16 \
    --num-anchors 512 \
    --loss-decay-gamma 7.0 \
    --log-interval 50 \
    --save-interval 1000