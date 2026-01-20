#!/bin/bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels


TRAIN_GPU=${TRAIN_GPU:-0}
INFER_GPU=${INFER_GPU:-1}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

# Auto-detect InfiniBand device name
IB_DEVICE=$(ls /sys/class/infiniband/ 2>/dev/null | head -n 1)
MOONCAKE_DEVICE_NAME=${MOONCAKE_DEVICE_NAME:-$IB_DEVICE}


DRAFT_CONFIG=${DRAFT_CONFIG:-$ROOT_DIR/configs/qwen3-8b-eagle3.json}
TRAIN_DATA=${TRAIN_DATA:-$ROOT_DIR/cache/dataset/sharegpt_train.jsonl}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/outputs/qwen3-8b-eagle3-remote}

MC_PORT=${MC_PORT:-50051}
MC_HOST=${MC_HOST:-127.0.0.1}
MC_HTTP_PORT=${MC_HTTP_PORT:-8090}
TASK_QUEUE_ADDR=${TASK_QUEUE_ADDR:-tcp://127.0.0.1:5555}
NOTIFY_ADDR=${NOTIFY_ADDR:-tcp://127.0.0.1:5556}
MC_PROTOCOL=${MC_PROTOCOL:-rdma}

cleanup() {
    echo "=== Stopping all services ==="
    jobs -p | xargs -r kill 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

run_master() {
    local BIN
    if [ -n "$MOONCAKE_BUILD_DIR" ]; then
        BIN="$MOONCAKE_BUILD_DIR/mooncake-store/src/mooncake_master"
    elif command -v mooncake_master &> /dev/null; then
        BIN="mooncake_master"
    else
        BIN="$ROOT_DIR/build/mooncake-store/src/mooncake_master"
    fi

    echo ">>> [Master] Starting on Port $MC_PORT..."
    if [ ! -x "$(command -v $BIN)" ] && [ ! -f "$BIN" ]; then
        echo "Error: Mooncake master binary not found at $BIN"
        return 1
    fi

    "$BIN" \
        --port="$MC_PORT" \
        --http_metadata_server_port="$MC_HTTP_PORT" \
        --http_metadata_server_host="0.0.0.0" \
        --enable_http_metadata_server=true \
        "$@"
}

run_worker() {
    echo ">>> [Worker] Starting on GPU $INFER_GPU..."

    local EXTRA_ARGS=""
    [ -n "$MOONCAKE_DEVICE_NAME" ] && EXTRA_ARGS="--mooncake-device-name $MOONCAKE_DEVICE_NAME"

    CUDA_VISIBLE_DEVICES="$INFER_GPU" torchrun \
        --standalone \
        --nproc_per_node=1 \
        -m specforge.modeling.target.remote_backend \
        --model-path "$MODEL_PATH" \
        --tp-size 1 \
        --task-queue-addr "$TASK_QUEUE_ADDR" \
        --notify-addr "$NOTIFY_ADDR" \
        --mooncake-master-addr "$MC_HOST:$MC_PORT" \
        --mooncake-metadata-port "$MC_HTTP_PORT" \
        --mooncake-protocol "$MC_PROTOCOL" \
        --dtype bfloat16 \
        $EXTRA_ARGS \
        "$@"
}

run_trainer() {
    echo ">>> [Trainer] Starting on GPU $TRAIN_GPU..."

    local EXTRA_ARGS=""
    [ -n "$LOCAL_HOSTNAME" ] && EXTRA_ARGS="--mooncake-local-hostname $LOCAL_HOSTNAME"
    [ -n "$MOONCAKE_DEVICE_NAME" ] && EXTRA_ARGS="$EXTRA_ARGS --mooncake-device-name $MOONCAKE_DEVICE_NAME"

    CUDA_VISIBLE_DEVICES="$TRAIN_GPU" torchrun \
        --standalone \
        --nproc_per_node 1 \
        "$ROOT_DIR/scripts/train_eagle3.py" \
        --target-model-path "$MODEL_PATH" \
        --draft-model-config "$DRAFT_CONFIG" \
        --train-data-path "$TRAIN_DATA" \
        --build-dataset-num-proc "$BUILD_DATASET_NUM_PROC" \
        --output-dir "$OUTPUT_DIR" \
        --num-epochs 10 \
        --batch-size 1 \
        --learning-rate 1e-4 \
        --max-length 4096 \
        --chat-template qwen \
        --cache-dir "$ROOT_DIR/cache" \
        --embedding-key model.embed_tokens.weight \
        --target-model-backend remote \
        --task-queue-addr "$TASK_QUEUE_ADDR" \
        --notify-addr "$NOTIFY_ADDR" \
        --mooncake-master-addr "$MC_HOST:$MC_PORT" \
        --mooncake-metadata-port "$MC_HTTP_PORT" \
        --mooncake-protocol "$MC_PROTOCOL" \
        --training-method disagg \
        --prefetch-depth 4 \
        $EXTRA_ARGS \
        "$@"
}

ROLE=${1:-all}
shift 1 || true

case "$ROLE" in
    master)
        run_master "$@"
        ;;
    worker)
        run_worker "$@"
        ;;
    trainer)
        run_trainer "$@"
        ;;
    all)
        echo "=== Launching Full System ==="

        run_master > master.log 2>&1 &
        MASTER_PID=$!
        echo "Master PID: $MASTER_PID"
        sleep 2
        kill -0 $MASTER_PID 2>/dev/null || { echo "Master died"; exit 1; }

        run_worker > worker.log 2>&1 &
        WORKER_PID=$!
        echo "Worker PID: $WORKER_PID. Waiting for model loading..."
        sleep 10
        kill -0 $WORKER_PID 2>/dev/null || { echo "Worker died"; exit 1; }

        echo "Starting Trainer (output: trainer.log)..."
        run_trainer "$@" > trainer.log 2>&1 &
        TRAINER_PID=$!
        echo "Trainer PID: $TRAINER_PID"

        echo "All services started. Tailing trainer.log (Ctrl+C to stop)..."
        tail -f trainer.log
        ;;
    *)
        echo "Usage: $0 {all|master|worker|trainer} [extra_args...]"
        exit 1
        ;;
esac
