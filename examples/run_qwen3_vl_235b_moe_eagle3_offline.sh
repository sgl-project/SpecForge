#!/bin/bash
export NCCL_IPV6=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo,eth0
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
MODEL_PATH=
DATA_PATH=

MODEL_NAME=Qwen3-VL-235B-A22B-Instruct-FP8
CACHE_DIR="$ROOT_DIR/cache"
HIDDEN_STATES_PATH="$CACHE_DIR/hidden_states/$MODEL_NAME"
OUTPUT_DIR="$ROOT_DIR/outputs/${MODEL_NAME}-offline"
DRAFT_MODEL_CONFIG="$ROOT_DIR/configs/qwen3-vl-235b-eagle3.json"

NUM_GPUS=4
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

if [ -z "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH is empty. Please set it to the target model path before running this script." >&2
    exit 1
fi

if [ -z "$DATA_PATH" ]; then
    echo "Error: DATA_PATH is empty. Please set it to the training dataset path before running this script." >&2
    exit 1
fi


torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/prepare_hidden_states.py \
    --target-model-path $MODEL_PATH \
    --enable-aux-hidden-states \
    --data-path "$DATA_PATH" \
    --output-path "$HIDDEN_STATES_PATH" \
    --chat-template qwen2-vl \
    --max-length 30000 \
    --tp-size 4 --is-vlm \
    --batch-size 1 \
    --sglang-mem-fraction-static 0.9


torchrun \
    --standalone \
    --nproc_per_node 1 \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path "$MODEL_PATH" \
    --draft-model-config "$DRAFT_MODEL_CONFIG" \
    --train-data-path "$DATA_PATH" \
    --train-hidden-states-path "$HIDDEN_STATES_PATH" \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir "$OUTPUT_DIR" \
    --num-epochs 10 \
    --batch-size 1 \
    --tp-size 1 \
    --target-model-backend sglang \
    --learning-rate 5e-5 \
    --max-length 30000 \
    --embedding-key model.language_model.embed_tokens.weight \
    --is-vlm \
    --chat-template qwen2-vl \
    --cache-dir "$CACHE_DIR" \
    --min-pixels 50176 \
    --max-pixels 802816
