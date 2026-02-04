#!/bin/bash
# GLM-4.7-Flash EAGLE3 Training Script
# Usage: ./examples/run_glm4_flash_eagle3_online.sh [NUM_GPUS] [TP_SIZE]

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# Load wandb API key from persistent storage
if [ -f /gustavo/.wandb_key ]; then
    export WANDB_API_KEY=$(cat /gustavo/.wandb_key)
fi

NUM_GPUS=${1:-1}
TP_SIZE=${2:-1}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

echo "========================================"
echo "GLM-4.7-Flash EAGLE3 Training"
echo "========================================"
echo "NUM_GPUS: $NUM_GPUS"
echo "TP_SIZE: $TP_SIZE"
echo "BUILD_DATASET_NUM_PROC: $BUILD_DATASET_NUM_PROC"
echo "========================================"

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path zai-org/GLM-4.7-Flash \
    --trust-remote-code \
    --draft-model-config $ROOT_DIR/configs/glm4-flash-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/glm4-flash-eagle3-sharegpt \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template glm4 \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-mem-fraction-static 0.4
