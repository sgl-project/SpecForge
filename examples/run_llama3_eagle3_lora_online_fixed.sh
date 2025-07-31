#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

NUM_GPUS=${1:-8}
TARGET_LORA_PATH=${2:-/sgl-workspace/draft_cache/algoprog_fact-generation-llama-3.1-8b-instruct-lora}
DRAFT_LORA_CONFIG=${3:-$ROOT_DIR/configs/draft_lora_trainable_config.json}
BASE_DRAFT_MODEL_PATH=${4:-/sgl-workspace/draft_cache/jamesliu1_sglang-EAGLE3-Llama-3.1-Instruct-8B}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_lora_online.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config $ROOT_DIR/configs/llama3-8B-eagle3.json \
    --base-draft-model-path $BASE_DRAFT_MODEL_PATH \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt.jsonl \
    --output-dir $ROOT_DIR/outputs/llama3-8b-eagle3-lora-fixed \
    --use-lora \
    --lora-config $DRAFT_LORA_CONFIG \
    --target-lora-path $TARGET_LORA_PATH \
    --num-epochs 1 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template llama3 \
    --cache-dir $ROOT_DIR/cache