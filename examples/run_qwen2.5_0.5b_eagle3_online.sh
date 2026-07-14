#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# Train an EAGLE3 draft for Qwen2.5-0.5B-Instruct with online data collection.
#
# This small model is the one used for the DataFlow-runtime-vs-main parity check.
# The comparison curve (100 steps, DataFlow colocated == disaggregated, tracking
# main) is in examples/assets/qwen2.5-0.5b-eagle3-vs-main.png. The disaggregated
# variant is examples/disagg/run_disagg_eagle3.py.

NUM_GPUS=${1:-1}
TP_SIZE=${2:-1}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path Qwen/Qwen2.5-0.5B-Instruct \
    --draft-model-config $ROOT_DIR/configs/qwen2.5-0.5b-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/qwen2.5-0.5b-eagle3-sharegpt \
    --num-epochs 10 \
    --batch-size 1 \
    --tp-size $TP_SIZE \
    --learning-rate 1e-4 \
    --ttt-length 7 \
    --max-length 512 \
    --chat-template qwen \
    --target-model-backend hf \
    --attention-backend flex_attention \
    --cache-dir $ROOT_DIR/cache \
    --log-interval 10
