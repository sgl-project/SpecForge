#!/bin/bash
# Qwen3-8B Domino, ONLINE disaggregated, MULTI-SERVER on one node:
#   mooncake master           -> CPU  (RDMA/TCP object store)
#   patched SGLang server 0   -> SERVER0_GPUS (frozen Qwen3-8B -> mooncake)
#   patched SGLang server 1   -> SERVER1_GPUS (same model + capture flags)
#   producer (HTTP driver)    -> CPU  (fans prompts out to BOTH servers)
#   consumer (Domino trainer) -> CONSUMER_GPUS (DP=TRAIN_DP)
#
# Domino uses the same server-side DFlash feature capture as DFlash. The draft
# config must have dflash_config.projector_type="domino"; the trainer consumes
# the captured hidden_states with strategy="domino".
set -euxo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/scripts:${PYTHONPATH:-}"
cd "$ROOT_DIR"

# --- topology: 2 servers x TP=1 + DP=2 trainer (override via env) ---
SERVER0_GPUS=${SERVER0_GPUS:-"2"}
SERVER1_GPUS=${SERVER1_GPUS:-"3"}
SERVER_TP=${SERVER_TP:-1}
SERVER0_PORT=${SERVER0_PORT:-30000}
SERVER1_PORT=${SERVER1_PORT:-30001}
TRAIN_DP=${TRAIN_DP:-2}
CONSUMER_GPUS=${CONSUMER_GPUS:-"4,5"}

TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-/disk3/wjp/pretrained_models/Qwen3-8B}
DRAFT_CONFIG_PATH=${DRAFT_CONFIG_PATH:-$ROOT_DIR/configs/qwen3-8b-domino.json}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-/disk3/wjp/datasets/perfectblend/qwen3-8b/perfectblend_train_regen_temperature0_no_think.jsonl}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-qwen}

# Must match dflash_config.target_layer_ids in configs/qwen3-8b-domino.json.
AUX_LAYER_IDS=${AUX_LAYER_IDS:-"1 9 17 25 33"}

# --- mooncake connection (server sinks + producer + consumer share these) ---
# export MOONCAKE_LOCAL_HOSTNAME=${MOONCAKE_LOCAL_HOSTNAME:-127.0.0.1}
# export MOONCAKE_MASTER_SERVER_ADDR=${MOONCAKE_MASTER_SERVER_ADDR:-127.0.0.1:50051}
# export MOONCAKE_METADATA_SERVER=${MOONCAKE_METADATA_SERVER:-http://127.0.0.1:8080/metadata}

MOONCAKE_HOST=${MOONCAKE_HOST:-127.0.0.1}
MOONCAKE_RPC_PORT=${MOONCAKE_RPC_PORT:-50051}
MOONCAKE_HTTP_PORT=${MOONCAKE_HTTP_PORT:-18080}
MOONCAKE_METRICS_PORT=${MOONCAKE_METRICS_PORT:-19003}

export MOONCAKE_LOCAL_HOSTNAME=$MOONCAKE_HOST
export MOONCAKE_MASTER_SERVER_ADDR=$MOONCAKE_HOST:$MOONCAKE_RPC_PORT
export MOONCAKE_METADATA_SERVER=http://$MOONCAKE_HOST:$MOONCAKE_HTTP_PORT/metadata
export MOONCAKE_PROTOCOL=${MOONCAKE_PROTOCOL:-tcp}
export MOONCAKE_GLOBAL_SEGMENT_SIZE=${MOONCAKE_GLOBAL_SEGMENT_SIZE:-$((32 << 30))}
export DISAGG_CLIENT_SEGMENT_SIZE=${DISAGG_CLIENT_SEGMENT_SIZE:-$((256 << 20))}
export DISAGG_CLIENT_BUFFER_SIZE=${DISAGG_CLIENT_BUFFER_SIZE:-$((256 << 20))}

export DISAGG_STORE_ID=${DISAGG_STORE_ID:-qwen3-8b-domino-disagg-2srv}
export DISAGG_SERVER_URLS="http://127.0.0.1:$SERVER0_PORT,http://127.0.0.1:$SERVER1_PORT"
export DISAGG_REF_CHANNEL=${DISAGG_REF_CHANNEL:-$ROOT_DIR/outputs/$DISAGG_STORE_ID/refs.jsonl}
DISAGG_DB=${DISAGG_DB:-$ROOT_DIR/outputs/$DISAGG_STORE_ID/run.db}
DISAGG_INBOX_DIR=${DISAGG_INBOX_DIR:-$ROOT_DIR/outputs/$DISAGG_STORE_ID/inboxes}
export DISAGG_MAX_PROMPTS=${DISAGG_MAX_PROMPTS:-400000}
export DISAGG_MAX_STEPS=${DISAGG_MAX_STEPS:-0}
export DISAGG_TOTAL_STEPS=${DISAGG_TOTAL_STEPS:-10000}
export DISAGG_LOG_INTERVAL=${DISAGG_LOG_INTERVAL:-1}
REPORT_TO=${REPORT_TO:-wandb}  # set REPORT_TO=none to run without W&B

rm -rf "$(dirname "$DISAGG_REF_CHANNEL")" "$DISAGG_DB"
mkdir -p "$(dirname "$DISAGG_REF_CHANNEL")"
: > "$DISAGG_REF_CHANNEL"

cleanup() { kill "${MASTER_PID:-}" "${SERVER0_PID:-}" "${SERVER1_PID:-}" "${PRODUCER_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT

# --- mooncake master ---
mooncake_master \
    --enable_http_metadata_server=true \
    --rpc_port="$MOONCAKE_RPC_PORT" \
    --http_metadata_server_port="$MOONCAKE_HTTP_PORT" \
    --metrics_port="$MOONCAKE_METRICS_PORT" &
MASTER_PID=$!
sleep 3

# --- patched SGLang servers: frozen target, spec-capture on ---
launch_server() { # $1=gpus $2=port
    CUDA_VISIBLE_DEVICES=$1 MOONCAKE_LOCAL_HOSTNAME=$MOONCAKE_LOCAL_HOSTNAME \
        python -m sglang.launch_server \
            --model-path "$TARGET_MODEL_PATH" \
            --trust-remote-code \
            --skip-tokenizer-init \
            --tp-size "$SERVER_TP" \
            --mem-fraction-static 0.85 \
            --chunked-prefill-size -1 \
            --disable-radix-cache \
            --enable-spec-capture \
            --spec-capture-method dflash \
            --spec-capture-aux-layer-ids $AUX_LAYER_IDS \
            --port "$2" &
}
launch_server "$SERVER0_GPUS" "$SERVER0_PORT"
SERVER0_PID=$!
launch_server "$SERVER1_GPUS" "$SERVER1_PORT"
SERVER1_PID=$!

for port_pid in "$SERVER0_PORT:$SERVER0_PID" "$SERVER1_PORT:$SERVER1_PID"; do
    port=${port_pid%%:*}; pid=${port_pid##*:}
    until curl -sf "http://127.0.0.1:$port/health" > /dev/null; do
        if ! kill -0 "$pid" 2>/dev/null; then echo "server on :$port died"; exit 1; fi
        sleep 5
    done
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
    --batch-size 2
    --learning-rate 6e-4
    --warmup-ratio 0.04
    --max-grad-norm 1.0
    --attention-backend flex_attention
    --block-size 16
    --num-anchors 256
    --loss-decay-gamma 7.0
    --num-epochs 6
    --seed 42
    --save-interval 2000
    --lambda-base-start 1.0
    --lambda-base-decay-ratio 1.0
)

LAUNCHER=$SCRIPT_DIR/run_disagg_domino.py

DISAGG_ROLE=producer CUDA_VISIBLE_DEVICES="" \
    python "$LAUNCHER" "${ARGS[@]}" \
        --output-dir "$ROOT_DIR/outputs/qwen3-8b-domino-disagg-2srv-producer" &
PRODUCER_PID=$!

CUDA_VISIBLE_DEVICES=$CONSUMER_GPUS DISAGG_ROLE=consumer \
    DISAGG_DB=$DISAGG_DB DISAGG_INBOX_DIR=$DISAGG_INBOX_DIR \
    torchrun --rdzv-backend c10d --rdzv-endpoint localhost:29702 \
        --nnodes 1 --nproc_per_node "$TRAIN_DP" "$LAUNCHER" "${ARGS[@]}" \
        --output-dir "$ROOT_DIR/outputs/qwen3-8b-domino-disagg-2srv-consumer" \
        --report-to "$REPORT_TO" \
        --wandb-project qwen3-8b-domino-disagg \
        --wandb-name qwen3-8b-domino-2srv-dp$TRAIN_DP

wait $PRODUCER_PID
echo "QWEN3-8B-DOMINO-DISAGG-2SRV-DONE"
