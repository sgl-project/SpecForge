#!/bin/bash
# MiniCPM-V-4 EAGLE3 Online Training Script
#
# Usage:
#   bash run_minicpm_v4_eagle3_online.sh [NUM_GPUS] [TARGET_MODEL_PATH] [TRAIN_DATA_PATH] [OUTPUT_DIR]
#
# Examples:
#   bash run_minicpm_v4_eagle3_online.sh 1
#   bash run_minicpm_v4_eagle3_online.sh 2 /path/to/MiniCPM-V-4
#   bash run_minicpm_v4_eagle3_online.sh 4 /path/to/model /path/to/data.jsonl /path/to/output

export FLASHINFER_DISABLE_VERSION_CHECK=1
export TORCHDYNAMO_DISABLE=1

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# Script parameters with defaults
NUM_GPUS=${1:-1}
TARGET_MODEL_PATH=${2:-"$ROOT_DIR/models/MiniCPM-V-4"}
TRAIN_DATA_PATH=${3:-"$ROOT_DIR/datasets/minicpm_v4_train.jsonl"}
OUTPUT_DIR=${4:-"$ROOT_DIR/outputs/MiniCPM-V-4-eagle3"}

# Environment variables
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-32}

echo "=========================================="
echo "MiniCPM-V-4 EAGLE3 Training Configuration"
echo "=========================================="
echo "NUM_GPUS:          $NUM_GPUS"
echo "TARGET_MODEL_PATH: $TARGET_MODEL_PATH"
echo "TRAIN_DATA_PATH:   $TRAIN_DATA_PATH"
echo "OUTPUT_DIR:        $OUTPUT_DIR"
echo "=========================================="

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path $TARGET_MODEL_PATH \
    --draft-model-config $ROOT_DIR/configs/minicpm-v4-eagle3.json \
    --train-data-path $TRAIN_DATA_PATH \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --dataloader-num-workers 0 \
    --output-dir $OUTPUT_DIR \
    --num-epochs 23 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --dist-timeout 360 \
    --chat-template minicpm-v \
    --cache-dir $ROOT_DIR/cache \
    --tp-size 1 \
    --is-vlm \
    --trust-remote-code \
    --embedding-key llm.model.embed_tokens.weight \
    --lm-head-key llm.lm_head.weight \
    --save-per-epoch \
    --debug-first-sample \
    --attention-backend fa
