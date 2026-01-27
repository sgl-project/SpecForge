#!/bin/bash
# MiniCPM-V-4 EAGLE3 Multi-Node Training Script
#
# Environment Variables (auto-configured by scheduler):
#   MASTER_ADDR:        Master node IP
#   MASTER_PORT:        Communication port
#   RANK:               Node rank (0, 1, 2, ...)
#   WORLD_SIZE:         Number of nodes
#   GPUS_PER_NODE:      GPUs per node
#   TARGET_MODEL_PATH:  Model path
#   TRAIN_DATA_PATH:    Training data path
#   OUTPUT_DIR:         Output directory

set -e

export FLASHINFER_DISABLE_VERSION_CHECK=1
export TORCHDYNAMO_DISABLE=1

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# Defaults (override with environment variables)
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-"$ROOT_DIR/models/MiniCPM-V-4"}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-"$ROOT_DIR/datasets/minicpm_v4_train.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT_DIR/outputs/MiniCPM-V-4-eagle3"}
LOGGING_DIR=${LOGGING_DIR:-"$ROOT_DIR/logs/tensorboard/MiniCPM-V-4-eagle3"}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-32}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

TOTAL_GPUS=$((WORLD_SIZE * GPUS_PER_NODE))

echo "=========================================="
echo "EAGLE3 Multi-Node Training"
echo "=========================================="
echo "Nodes: $WORLD_SIZE x $GPUS_PER_NODE GPUs = $TOTAL_GPUS total"
echo "Node $RANK -> $MASTER_ADDR:$MASTER_PORT"
echo "Model: $TARGET_MODEL_PATH"
echo "Data:  $TRAIN_DATA_PATH"
echo "=========================================="

torchrun \
    --nnodes $WORLD_SIZE \
    --nproc_per_node $GPUS_PER_NODE \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --node_rank $RANK \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path $TARGET_MODEL_PATH \
    --draft-model-config $ROOT_DIR/configs/minicpm-v4-eagle3.json \
    --train-data-path $TRAIN_DATA_PATH \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --dataloader-num-workers 0 \
    --output-dir $OUTPUT_DIR \
    --num-epochs 25 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --dist-timeout 360 \
    --chat-template minicpm-v \
    --cache-dir $ROOT_DIR/cache \
    --tp-size 1 \
    --is-vlm \
    --trust-remote-code \
    --embedding-key llm.model.embed_tokens.weight \
    --lm-head-key llm.lm_head.weight \
    --debug-first-sample \
    --report-to tensorboard \
    --logging-dir $LOGGING_DIR \
    --attention-backend fa
