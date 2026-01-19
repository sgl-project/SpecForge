#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
TP_SIZE=${TP_SIZE:-1}
INFERENCE_GPUS=${INFERENCE_GPUS:-"0"}
TASK_QUEUE_ADDR=${TASK_QUEUE_ADDR:-tcp://0.0.0.0:5555}
NOTIFY_ADDR=${NOTIFY_ADDR:-tcp://0.0.0.0:5556}
MOONCAKE_MASTER=${MOONCAKE_MASTER:-127.0.0.1:50051}
MOONCAKE_METADATA_PORT=${MOONCAKE_METADATA_PORT:-8090}
MOONCAKE_PROTOCOL=${MOONCAKE_PROTOCOL:-tcp}
DTYPE=${DTYPE:-bfloat16}
USE_ZERO_COPY=${USE_ZERO_COPY:-true}

cleanup() {
    echo "Cleaning up..."
    jobs -p | xargs -r kill 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting inference worker with TP=$TP_SIZE on GPUs: $INFERENCE_GPUS"
echo "Model: $MODEL_PATH"

if [ "$TP_SIZE" -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=$INFERENCE_GPUS python -m specforge.modeling.target.remote_backend \
        --model-path $MODEL_PATH \
        --tp-size 1 \
        --task-queue-addr $TASK_QUEUE_ADDR \
        --notify-addr $NOTIFY_ADDR \
        --mooncake-master-addr $MOONCAKE_MASTER \
        --mooncake-metadata-port $MOONCAKE_METADATA_PORT \
        --mooncake-protocol $MOONCAKE_PROTOCOL \
        --mooncake-device-name "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_12,mlx5_13" \
        --dtype $DTYPE \
        --use-zero-copy $USE_ZERO_COPY
else
    CUDA_VISIBLE_DEVICES=$INFERENCE_GPUS torchrun \
        --standalone \
        --nproc_per_node=$TP_SIZE \
        -m specforge.modeling.target.remote_backend \
        --model-path $MODEL_PATH \
        --tp-size $TP_SIZE \
        --task-queue-addr $TASK_QUEUE_ADDR \
        --notify-addr $NOTIFY_ADDR \
        --mooncake-master-addr $MOONCAKE_MASTER \
        --mooncake-metadata-port $MOONCAKE_METADATA_PORT \
        --mooncake-protocol $MOONCAKE_PROTOCOL \
        --dtype $DTYPE \
        --use-zero-copy $USE_ZERO_COPY
fi