#!/bin/bash
#
# Compare local sglang backend inference with remote backend inference.
#
# Prerequisites:
#   1. Start the Mooncake metadata server and master (see scripts/mooncake/)
#   2. Start the remote inference worker: ./examples/run_remote_inference.sh
#   3. Run this script on a SEPARATE GPU (not used by the remote worker)
#
# Usage:
#   LOCAL_GPU=1 ./examples/run_comparison_test.sh
#

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
LOCAL_GPU=${LOCAL_GPU:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
SEQ_LEN=${SEQ_LEN:-128}

TASK_QUEUE_ADDR=${TASK_QUEUE_ADDR:-tcp://127.0.0.1:5555}
NOTIFY_ADDR=${NOTIFY_ADDR:-tcp://127.0.0.1:5556}

MOONCAKE_LOCAL_HOSTNAME=${MOONCAKE_LOCAL_HOSTNAME:-10.173.2.69}
MOONCAKE_METADATA_SERVER=${MOONCAKE_METADATA_SERVER:-http://localhost:8090/metadata}
MOONCAKE_MASTER_ADDR=${MOONCAKE_MASTER_ADDR:-127.0.0.1:50051}
MOONCAKE_PROTOCOL=${MOONCAKE_PROTOCOL:-rdma}

echo "=============================================="
echo "Local vs Remote Inference Comparison Test"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Local GPU: $LOCAL_GPU"
echo "Batch size: $BATCH_SIZE"
echo "Sequence length: $SEQ_LEN"
echo "Task queue: $TASK_QUEUE_ADDR"
echo "Notify address: $NOTIFY_ADDR"
echo "Mooncake protocol: $MOONCAKE_PROTOCOL"
echo "=============================================="

CUDA_VISIBLE_DEVICES=$LOCAL_GPU python $ROOT_DIR/tests/test_modeling/test_target/test_remote_backend/compare_local_remote_inference.py \
    --model-path $MODEL_PATH \
    --batch-size $BATCH_SIZE \
    --seq-len $SEQ_LEN \
    --task-queue-addr $TASK_QUEUE_ADDR \
    --notify-addr $NOTIFY_ADDR \
    --mooncake-local-hostname $MOONCAKE_LOCAL_HOSTNAME \
    --mooncake-metadata-server $MOONCAKE_METADATA_SERVER \
    --mooncake-master-addr $MOONCAKE_MASTER_ADDR \
    --mooncake-protocol $MOONCAKE_PROTOCOL
