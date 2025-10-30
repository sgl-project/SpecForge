#!/bin/bash

MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
MODEL_NAME=Qwen2.5-7B-Instruct
DATASET_PATH=/datasets
PERSIST_DIR=/specforge_output/$MODEL_NAME/50w # Please Change this to your own directory

CACHE_DIR=$PERSIST_DIR/cache
OUTPUT_DIR=$PERSIST_DIR/outputs
CHAT_TEMPLATE=qwen
MAX_LENGTH=16384


DRAFT_MODEL_CONFIG=./configs/qwen2.5-7b-eagle3.json

NUM_GPUS=8
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    scripts/train_eagle3_sgl_online.py \
    --target-model-path $MODEL_PATH \
    --model-path $MODEL_PATH \
    --draft-model-config $DRAFT_MODEL_CONFIG \
    --train-data-path $DATASET_PATH/all_50w.jsonl \
    --tp-size 2 \
    --output-dir $OUTPUT_DIR \
    --num-epochs 4 \
    --learning-rate 5e-5 \
    --draft-attention-backend flex_attention \
    --draft-global-batch-size 16 \
    --max-length $MAX_LENGTH \
    --chat-template $CHAT_TEMPLATE \
    --cache-dir $CACHE_DIR \
    --mem-frac=0.4 \
    --warmup-ratio=0.015 \
    --dist-timeout=10000 \
    --resume \
    --report-to tensorboard \
    --target-model-backend hf
