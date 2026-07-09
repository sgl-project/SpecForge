#!/bin/bash
# Qwen3-8B Domino, online disaggregated training on one multi-GPU node:
#   Mooncake master           -> CPU
#   patched SGLang server     -> SERVER_GPUS
#   producer HTTP driver      -> CPU
#   consumer Domino trainer   -> CONSUMER_GPUS, DP=TRAIN_DP
#
# Domino reuses DFlash server-side feature capture. The draft config must set
# dflash_config.projector_type="domino", and AUX_LAYER_IDS must match
# dflash_config.target_layer_ids in that config.
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")
cd "$ROOT_DIR"

# -----------------------------------------------------------------------------
# Local runtime paths
# -----------------------------------------------------------------------------
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/scripts:${PYTHONPATH:-}"

LAUNCHER=$SCRIPT_DIR/run_disagg_domino.py

# -----------------------------------------------------------------------------
# Model, data, and Domino capture config
# -----------------------------------------------------------------------------
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-/disk3/wjp/pretrained_models/Qwen3-8B}
DRAFT_CONFIG_PATH=${DRAFT_CONFIG_PATH:-$ROOT_DIR/configs/qwen3-8b-domino.json}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-/disk3/wjp/datasets/perfectblend/qwen3-8b/perfectblend_train_regen_temperature0_no_think.jsonl}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-qwen}
AUX_LAYER_IDS=${AUX_LAYER_IDS:-"1 9 17 25 33"}

# -----------------------------------------------------------------------------
# GPU topology
# -----------------------------------------------------------------------------
SERVER_GPUS=${SERVER_GPUS:-${SERVER0_GPUS:-"2"}}
SERVER_TP=${SERVER_TP:-1}
SERVER_PORT=${SERVER_PORT:-${SERVER0_PORT:-30000}}

TRAIN_DP=${TRAIN_DP:-4}
CONSUMER_GPUS=${CONSUMER_GPUS:-"3,4,5,6"}

# -----------------------------------------------------------------------------
# Mooncake object-store config
# -----------------------------------------------------------------------------
MOONCAKE_HOST=${MOONCAKE_HOST:-127.0.0.1}
MOONCAKE_RPC_PORT=${MOONCAKE_RPC_PORT:-35551}
MOONCAKE_HTTP_PORT=${MOONCAKE_HTTP_PORT:-35880}
MOONCAKE_METRICS_PORT=${MOONCAKE_METRICS_PORT:-35903}

export MOONCAKE_LOCAL_HOSTNAME=$MOONCAKE_HOST
export MOONCAKE_MASTER_SERVER_ADDR=$MOONCAKE_HOST:$MOONCAKE_RPC_PORT
export MOONCAKE_METADATA_SERVER=http://$MOONCAKE_HOST:$MOONCAKE_HTTP_PORT/metadata
export MOONCAKE_PROTOCOL=${MOONCAKE_PROTOCOL:-rdma}
export MOONCAKE_GLOBAL_SEGMENT_SIZE=${MOONCAKE_GLOBAL_SEGMENT_SIZE:-$((32 << 30))}

# Only long-lived SGLang servers should own segments; producer/consumer are
# clients. If the producer owns feature objects, they can disappear when it exits.
export DISAGG_CLIENT_SEGMENT_SIZE=${DISAGG_CLIENT_SEGMENT_SIZE:-0}
export DISAGG_CLIENT_BUFFER_SIZE=${DISAGG_CLIENT_BUFFER_SIZE:-$((256 << 20))}
if [[ "$DISAGG_CLIENT_SEGMENT_SIZE" -ne 0 ]]; then
    echo "DISAGG_CLIENT_SEGMENT_SIZE must be 0 for server-owned captures" >&2
    exit 2
fi

# -----------------------------------------------------------------------------
# Disaggregated runtime channels and limits
# -----------------------------------------------------------------------------
export DISAGG_STORE_ID=${DISAGG_STORE_ID:-qwen3-8b-domino-disagg-1srv}
export DISAGG_SERVER_URLS="http://127.0.0.1:$SERVER_PORT"
export DISAGG_REF_CHANNEL=${DISAGG_REF_CHANNEL:-$ROOT_DIR/outputs/$DISAGG_STORE_ID/refs.jsonl}
export DISAGG_MAX_PROMPTS=${DISAGG_MAX_PROMPTS:-400000}
export DISAGG_MAX_STEPS=${DISAGG_MAX_STEPS:-0}
export DISAGG_TOTAL_STEPS=${DISAGG_TOTAL_STEPS:-100000}
export DISAGG_LOG_INTERVAL=${DISAGG_LOG_INTERVAL:-1}

DISAGG_DB=${DISAGG_DB:-$ROOT_DIR/outputs/$DISAGG_STORE_ID/run.db}
DISAGG_INBOX_DIR=${DISAGG_INBOX_DIR:-$ROOT_DIR/outputs/$DISAGG_STORE_ID/inboxes}
RUN_OUTPUT_DIR=$(dirname "$DISAGG_REF_CHANNEL")
case "$RUN_OUTPUT_DIR" in
    "$ROOT_DIR"/outputs/*) ;;
    *)
        echo "DISAGG_REF_CHANNEL must live under $ROOT_DIR/outputs" >&2
        exit 2
        ;;
esac
case "$DISAGG_DB" in
    "$RUN_OUTPUT_DIR"/*) ;;
    *)
        echo "DISAGG_DB must live under $RUN_OUTPUT_DIR" >&2
        exit 2
        ;;
esac
MOONCAKE_LOG=${MOONCAKE_LOG:-$RUN_OUTPUT_DIR/mooncake.log}

PRODUCER_OUTPUT_DIR=$ROOT_DIR/outputs/qwen3-8b-domino-disagg-1p4c-producer
CONSUMER_OUTPUT_DIR=$ROOT_DIR/outputs/qwen3-8b-domino-disagg-1p4c-consumer
REPORT_TO=${REPORT_TO:-none}

# -----------------------------------------------------------------------------
# Common training arguments
# -----------------------------------------------------------------------------
ARGS=(
    # Target and draft model
    --target-model-path "$TARGET_MODEL_PATH"
    --target-model-backend hf
    --trust-remote-code
    --draft-config-path "$DRAFT_CONFIG_PATH"

    # Data and tokenization
    --mask-token-id 151669
    --train-data-path "$TRAIN_DATA_PATH"
    --chat-template "$CHAT_TEMPLATE"
    --max-length 3072

    # Optimization
    --batch-size 2
    --learning-rate 6e-4
    --warmup-ratio 0.04
    --max-grad-norm 1.0
    --num-epochs 6
    --seed 42
    --save-interval 2000

    # Domino / DFlash draft head
    --attention-backend flex_attention
    --block-size 16
    --num-anchors 256
    --loss-decay-gamma 7.0
    --lambda-base-start 1.0
    --lambda-base-decay-ratio 1.0
)

SGLANG_ARGS=(
    --model-path "$TARGET_MODEL_PATH"
    --trust-remote-code
    --skip-tokenizer-init
    --tp-size "$SERVER_TP"
    --mem-fraction-static 0.85
    --chunked-prefill-size -1
    --disable-radix-cache
    --enable-spec-capture
    --spec-capture-method dflash
    --spec-capture-aux-layer-ids $AUX_LAYER_IDS
)


rm -rf "$RUN_OUTPUT_DIR" "$DISAGG_DB"
mkdir -p "$RUN_OUTPUT_DIR"
: > "$DISAGG_REF_CHANNEL"

cleanup() {
    kill "${SERVER_PID:-}" "${PRODUCER_PID:-}" 2>/dev/null || true
    if [[ -n "${MASTER_PID:-}" ]]; then
        kill -- "-$MASTER_PID" 2>/dev/null || kill "$MASTER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# -----------------------------------------------------------------------------
# Mooncake master
# -----------------------------------------------------------------------------
setsid mooncake_master \
    --enable_http_metadata_server=true \
    --rpc_port="$MOONCAKE_RPC_PORT" \
    --http_metadata_server_port="$MOONCAKE_HTTP_PORT" \
    --metrics_port="$MOONCAKE_METRICS_PORT" \
    >"$MOONCAKE_LOG" 2>&1 &
MASTER_PID=$!

wait_for_mooncake() {
    for _ in $(seq 1 30); do
        if ! kill -0 "$MASTER_PID" 2>/dev/null; then
            echo "Mooncake master exited during startup; see $MOONCAKE_LOG" >&2
            tail -n 100 "$MOONCAKE_LOG" >&2 || true
            return 1
        fi

        if curl -sS --max-time 1 -o /dev/null \
            "$MOONCAKE_METADATA_SERVER?key=specforge-health-check" \
            && timeout 1 bash -c \
                "</dev/tcp/$MOONCAKE_HOST/$MOONCAKE_RPC_PORT" 2>/dev/null; then
            return 0
        fi

        sleep 1
    done

    echo "Mooncake master did not become ready; see $MOONCAKE_LOG" >&2
    tail -n 100 "$MOONCAKE_LOG" >&2 || true
    return 1
}
wait_for_mooncake

# -----------------------------------------------------------------------------
# SGLang capture servers
# -----------------------------------------------------------------------------
launch_server() {
    local gpus=$1
    local port=$2

    CUDA_VISIBLE_DEVICES=$gpus \
        python -m sglang.launch_server "${SGLANG_ARGS[@]}" --port "$port" &
}

wait_for_server() {
    local port=$1
    local pid=$2

    until curl -sf "http://127.0.0.1:$port/health" > /dev/null; do
        if ! kill -0 "$MASTER_PID" 2>/dev/null; then
            echo "Mooncake master died while SGLang servers were starting" >&2
            tail -n 100 "$MOONCAKE_LOG" >&2 || true
            exit 1
        fi

        if ! kill -0 "$pid" 2>/dev/null; then
            echo "server on :$port died" >&2
            exit 1
        fi

        sleep 5
    done
}

launch_server "$SERVER_GPUS" "$SERVER_PORT"
SERVER_PID=$!

wait_for_server "$SERVER_PORT" "$SERVER_PID"

# -----------------------------------------------------------------------------
# Producer and consumer
# -----------------------------------------------------------------------------
DISAGG_ROLE=producer CUDA_VISIBLE_DEVICES="" \
    python "$LAUNCHER" "${ARGS[@]}" \
        --output-dir "$PRODUCER_OUTPUT_DIR" &
PRODUCER_PID=$!

CUDA_VISIBLE_DEVICES=$CONSUMER_GPUS DISAGG_ROLE=consumer \
    DISAGG_DB=$DISAGG_DB DISAGG_INBOX_DIR=$DISAGG_INBOX_DIR \
    torchrun --rdzv-backend c10d --rdzv-endpoint localhost:29702 \
        --nnodes 1 --nproc_per_node "$TRAIN_DP" "$LAUNCHER" "${ARGS[@]}" \
        --output-dir "$CONSUMER_OUTPUT_DIR" \
        --report-to "$REPORT_TO" \
        --wandb-project qwen3-8b-domino-disagg \
        --wandb-name qwen3-8b-domino-1srv-dp$TRAIN_DP

wait "$PRODUCER_PID"
echo "QWEN3-8B-DOMINO-DISAGG-1SRV-DONE"
