#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

NUM_GPUS=${1:-1}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online.py \
    --target-model-path microsoft/phi-4 \
    --draft-model-config $ROOT_DIR/configs/phi4-mini-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt.jsonl \
    --output-dir $ROOT_DIR/outputs/Phi-4-mini-eagle3 \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template phi4-mini \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size 1 \
    --ttt-length 7
