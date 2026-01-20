#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# Remote backend: separate inference and training GPUs
# GPU 0: Training process
# GPU 1: Inference worker (SGLang engine)
TRAIN_GPU=4
INFERENCE_GPU=5
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

MODEL_PATH=Qwen/Qwen3-8B
DRAFT_CONFIG=$ROOT_DIR/configs/qwen3-8b-eagle3.json
TRAIN_DATA=$ROOT_DIR/cache/dataset/sharegpt_train.jsonl

TASK_QUEUE_ADDR=tcp://127.0.0.1:5555
NOTIFY_ADDR=tcp://127.0.0.1:5556
MOONCAKE_MASTER=127.0.0.1:50051

cleanup() {
    echo "Cleaning up..."
    jobs -p | xargs -r kill 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting training on GPU $TRAIN_GPU..."
CUDA_VISIBLE_DEVICES=$TRAIN_GPU torchrun \
    --standalone \
    --nproc_per_node 1 \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path $MODEL_PATH \
    --draft-model-config $DRAFT_CONFIG \
    --train-data-path $TRAIN_DATA \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/qwen3-8b-eagle3-remote \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --target-model-backend remote \
    --task-queue-addr $TASK_QUEUE_ADDR \
    --notify-addr $NOTIFY_ADDR \
    --mooncake-master-addr $MOONCAKE_MASTER \
    --training-method disagg \
    --prefetch-depth 4
