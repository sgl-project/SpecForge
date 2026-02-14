#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=${SPECFORGE_DATA_NUM_PROC:-64}

NUM_GPUS=${1:-1}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}
WANDB_MODE=offline
SGL_JIT_DEEPGEMM_PRECOMPILE=false
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path meituan-longcat/LongCat-Flash-Chat-FP8 \
    --target-model-backend sglang \
    --tp-size $NUM_GPUS \
    --sglang-attention-backend flashinfer \
    --sglang-mem-fraction-static 0.75 \
    --sglang-ep-size $NUM_GPUS \
    --draft-config-path $ROOT_DIR/configs/longcat-flash-dflash.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/longcat-flash-dflash-sharegpt \
    --num-epochs 6 \
    --batch-size 2 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 3072 \
    --chat-template longcat \
    --num-anchors 512 \
    --loss-decay-gamma 7.0 \
    --log-interval 50 \
    --save-interval 1000 \
    --report-to wandb \
    --wandb-project specforge-longcat-flash-dflash \
    --wandb-name longcat-flash-dflash-sharegpt \
    --mask-token-id 2
