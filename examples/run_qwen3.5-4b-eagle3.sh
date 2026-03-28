#!/bin/bash
# Example: Train EAGLE3 for Qwen3.5-4B

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# Generate hidden states
CUDA_VISIBLE_DEVICES=0 torchrun \
    --standalone \
    --nproc_per_node 1 \
    $ROOT_DIR/scripts/prepare_hidden_states.py \
    --target-model-path Qwen/Qwen3.5-4B \
    --enable-aux-hidden-states \
    --data-path $ROOT_DIR/cache/dataset/ultrachat_train.jsonl \
    --output-path $ROOT_DIR/cache/hidden_states/qwen3.5-4b \
    --chat-template qwen \
    --max-length 1024 \
    --batch-size 8 \
    --sglang-mem-fraction-static 0.6

# Train draft model
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --standalone \
    --nproc_per_node 4 \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path Qwen/Qwen3.5-4B \
    --draft-model-config $ROOT_DIR/configs/qwen3.5-4b-eagle3.json \
    --train-hidden-states-path $ROOT_DIR/cache/hidden_states/qwen3.5-4b \
    --output-path $ROOT_DIR/outputs/qwen3.5-4b-eagle3 \
    --embedding-key "model.language_model.embed_tokens.weight"
