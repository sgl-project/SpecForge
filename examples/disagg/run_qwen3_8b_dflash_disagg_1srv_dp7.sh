#!/bin/bash
# Qwen3-8B Domino, ONLINE disaggregated, SINGLE inference server + DP=7 trainer:
#   mooncake master          -> CPU
#   patched SGLang server 0   -> SERVER0_GPUS (1 GPU: frozen Qwen3-8B -> mooncake)
#   producer (HTTP driver)    -> CPU
#   consumer (Domino trainer) -> CONSUMER_GPUS (DP=TRAIN_DP, 7 GPUs)
# Epochs: producer replicates the prompt pool x NUM_EPOCHS (servers are idle, so
# re-capturing is free) because the streaming channel is consume-once.
set -euxo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/scripts:${PYTHONPATH:-}"
cd "$ROOT_DIR"

# --- topology: 1 server x TP=1 + DP=7 trainer (override via env) ---
SERVER0_GPUS=${SERVER0_GPUS:-"0"}
SERVER_TP=${SERVER_TP:-1}
SERVER0_PORT=${SERVER0_PORT:-30000}
TRAIN_DP=${TRAIN_DP:-7}
CONSUMER_GPUS=${CONSUMER_GPUS:-"1,2,3,4,5,6,7"}

TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-Qwen/Qwen3-8B}
DRAFT_CONFIG_PATH=${DRAFT_CONFIG_PATH:-$ROOT_DIR/configs/qwen3-8b-dflash.json}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-/sgl-workspace/SpecForge/cache/dataset/perfectblend_train_regen_temperature0_no_think_20w.jsonl}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-qwen}

AUX_LAYER_IDS=${AUX_LAYER_IDS:-"1 9 17 25 33"}

MOONCAKE_HOST=${MOONCAKE_HOST:-127.0.0.1}
MOONCAKE_RPC_PORT=${MOONCAKE_RPC_PORT:-35551}
MOONCAKE_HTTP_PORT=${MOONCAKE_HTTP_PORT:-35880}
MOONCAKE_METRICS_PORT=${MOONCAKE_METRICS_PORT:-35903}

export MOONCAKE_LOCAL_HOSTNAME=$MOONCAKE_HOST
export MOONCAKE_MASTER_SERVER_ADDR=$MOONCAKE_HOST:$MOONCAKE_RPC_PORT
export MOONCAKE_METADATA_SERVER=http://$MOONCAKE_HOST:$MOONCAKE_HTTP_PORT/metadata
export MOONCAKE_PROTOCOL=${MOONCAKE_PROTOCOL:-tcp}
export MOONCAKE_GLOBAL_SEGMENT_SIZE=${MOONCAKE_GLOBAL_SEGMENT_SIZE:-$((32 << 30))}
export DISAGG_CLIENT_SEGMENT_SIZE=${DISAGG_CLIENT_SEGMENT_SIZE:-0}
export DISAGG_CLIENT_BUFFER_SIZE=${DISAGG_CLIENT_BUFFER_SIZE:-$((256 << 20))}
if [[ "$DISAGG_CLIENT_SEGMENT_SIZE" -ne 0 ]]; then
    echo "DISAGG_CLIENT_SEGMENT_SIZE must be 0 for server-owned captures" >&2
    exit 2
fi

export DISAGG_STORE_ID=${DISAGG_STORE_ID:-qwen3-8b-dflash-1srv-dp7}
export DISAGG_SERVER_URLS="${DISAGG_SERVER_URLS:-http://127.0.0.1:$SERVER0_PORT}"
export DISAGG_REF_CHANNEL=${DISAGG_REF_CHANNEL:-$ROOT_DIR/outputs/$DISAGG_STORE_ID/refs.jsonl}
DISAGG_DB=${DISAGG_DB:-$ROOT_DIR/outputs/$DISAGG_STORE_ID/run.db}
DISAGG_INBOX_DIR=${DISAGG_INBOX_DIR:-$ROOT_DIR/outputs/$DISAGG_STORE_ID/inboxes}
# 0 = no cap; with NUM_EPOCHS the producer replicates the pool, so leave this
# uncapped and let epochs control the sample volume.
export DISAGG_MAX_PROMPTS=${DISAGG_MAX_PROMPTS:-0}
export DISAGG_MAX_STEPS=${DISAGG_MAX_STEPS:-0}
export DISAGG_LOG_INTERVAL=${DISAGG_LOG_INTERVAL:-10}
NUM_EPOCHS=${NUM_EPOCHS:-10}
SAVE_INTERVAL=${SAVE_INTERVAL:-800}
BATCH_SIZE=${BATCH_SIZE:-2}
# LR-schedule horizon = ceil(NUM_EPOCHS * prompts / (TRAIN_DP * BATCH_SIZE)).
# Auto-derived from the dataset line count when DISAGG_TOTAL_STEPS is unset (a
# slight over-estimate vs the post-filter prompt count, which only makes the LR
# decay a hair slower -- safe). Set DISAGG_TOTAL_STEPS to override. A too-small
# value decays the LR to ~0 before training finishes.
if [[ -z "${DISAGG_TOTAL_STEPS:-}" ]]; then
    _n_prompts=$(wc -l < "$TRAIN_DATA_PATH")
    DISAGG_TOTAL_STEPS=$(python3 -c "import math,sys; print(math.ceil($NUM_EPOCHS*int(sys.argv[1])/($TRAIN_DP*$BATCH_SIZE)))" "$_n_prompts")
    echo "[launcher] auto DISAGG_TOTAL_STEPS=$DISAGG_TOTAL_STEPS (epochs=$NUM_EPOCHS prompts=$_n_prompts dp=$TRAIN_DP batch=$BATCH_SIZE)"
fi
export DISAGG_TOTAL_STEPS
REPORT_TO=${REPORT_TO:-none}
WANDB_PROJECT=${WANDB_PROJECT:-qwen3-8b-dflash-disagg}

python - <<'PY'
import torch
try:
    from mooncake.store import MooncakeDistributedStore, ReplicateConfig
except Exception as exc:
    raise SystemExit(f"Mooncake preflight failed: {type(exc).__name__}: {exc}") from exc
PY
if [[ "$REPORT_TO" == "wandb" ]]; then
    python - <<'PY'
import wandb
required = ("login", "init", "log", "finish")
if not all(callable(getattr(wandb, name, None)) for name in required):
    raise SystemExit("REPORT_TO=wandb requires a complete W&B client: pip install wandb")
PY
fi

rm -rf "$(dirname "$DISAGG_REF_CHANNEL")" "$DISAGG_DB"
mkdir -p "$(dirname "$DISAGG_REF_CHANNEL")"
: > "$DISAGG_REF_CHANNEL"
MOONCAKE_LOG=${MOONCAKE_LOG:-$(dirname "$DISAGG_REF_CHANNEL")/mooncake.log}

cleanup() {
    kill "${SERVER0_PID:-}" "${PRODUCER_PID:-}" 2>/dev/null || true
    if [[ -n "${MASTER_PID:-}" ]]; then
        kill -- "-$MASTER_PID" 2>/dev/null || kill "$MASTER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# --- mooncake master ---
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

# --- single patched SGLang server: frozen target, spec-capture on ---
launch_server() { # $1=gpus $2=port
    CUDA_VISIBLE_DEVICES=$1 MOONCAKE_LOCAL_HOSTNAME=$MOONCAKE_LOCAL_HOSTNAME \
        python -m sglang.launch_server \
            --model-path "$TARGET_MODEL_PATH" \
            --trust-remote-code \
            --skip-tokenizer-init \
            --tp-size "$SERVER_TP" \
            --mem-fraction-static "${SERVER_MEM_FRACTION:-0.85}" \
            --chunked-prefill-size -1 \
            --disable-radix-cache \
            --enable-spec-capture \
            --spec-capture-method dflash \
            --spec-capture-aux-layer-ids $AUX_LAYER_IDS \
            --port "$2" &
}
launch_server "$SERVER0_GPUS" "$SERVER0_PORT"
SERVER0_PID=$!

until curl -sf "http://127.0.0.1:$SERVER0_PORT/health" > /dev/null; do
    if ! kill -0 "$MASTER_PID" 2>/dev/null; then
        echo "Mooncake master died while SGLang server was starting" >&2
        tail -n 100 "$MOONCAKE_LOG" >&2 || true
        exit 1
    fi
    if ! kill -0 "$SERVER0_PID" 2>/dev/null; then echo "server on :$SERVER0_PORT died"; exit 1; fi
    sleep 5
done

ARGS=(
    --target-model-path "$TARGET_MODEL_PATH"
    --target-model-backend hf
    --trust-remote-code
    --draft-config-path "$DRAFT_CONFIG_PATH"
    --mask-token-id 151669
    --train-data-path "$TRAIN_DATA_PATH"
    --chat-template "$CHAT_TEMPLATE"
    --max-length 3072
    --batch-size ${BATCH_SIZE}
    --accumulation-steps ${ACCUM:-1}
    --learning-rate 6e-4
    --warmup-ratio 0.04
    --max-grad-norm 1.0
    --attention-backend flex_attention
    --block-size 16
    --num-anchors ${NUM_ANCHORS:-512}
    --loss-decay-gamma 7.0
    --num-epochs ${NUM_EPOCHS}
    --seed 42
    --save-interval ${SAVE_INTERVAL}
)

LAUNCHER=$SCRIPT_DIR/run_disagg_dflash.py

DISAGG_ROLE=producer CUDA_VISIBLE_DEVICES="" \
    python "$LAUNCHER" "${ARGS[@]}" \
        --output-dir "$ROOT_DIR/outputs/$DISAGG_STORE_ID-producer" &
PRODUCER_PID=$!

CUDA_VISIBLE_DEVICES=$CONSUMER_GPUS DISAGG_ROLE=consumer \
    DISAGG_DB=$DISAGG_DB DISAGG_INBOX_DIR=$DISAGG_INBOX_DIR \
    torchrun --rdzv-backend c10d --rdzv-endpoint localhost:29702 \
        --nnodes 1 --nproc_per_node "$TRAIN_DP" "$LAUNCHER" "${ARGS[@]}" \
        --output-dir "$ROOT_DIR/outputs/$DISAGG_STORE_ID-consumer" \
        --report-to "$REPORT_TO" \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-name "${WANDB_NAME:-qwen3-8b-dflash-1srv-dp$TRAIN_DP}"

wait $PRODUCER_PID
echo "QWEN3-8B-DFLASH-1SRV-DP7-DONE"
