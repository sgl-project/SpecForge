#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# Train EAGLE3 draft model for Qwen3-Coder-30B-A3B-Instruct
# Uses the regenerated OPC dataset and TP=4 on GPUs 4,5,6,7

# GPU Configuration - Use the later 4 GPUs (4,5,6,7)
export CUDA_VISIBLE_DEVICES=4,5,6,7
NUM_GPUS=4
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --draft-model-config $ROOT_DIR/configs/qwen3-coder-30B-A3B-instruct-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/opc_regenerated.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/qwen3-coder-30b-a3b-instruct-eagle3-opc-regen \
    --num-epochs 2 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size 4 \
    --dist-timeout 60 \
    --log-interval 50 \
    --save-interval 5000 \
    --eval-interval 5000 \
    --report-to wandb \
    --wandb-project specforge-qwen3-coder \
    --wandb-name qwen3-coder-30b-eagle3-tp4-opc-regen
