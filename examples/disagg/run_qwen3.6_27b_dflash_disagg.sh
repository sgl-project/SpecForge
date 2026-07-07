#!/bin/bash
# Qwen3.6-27B DFlash, ONLINE disaggregated via server-capture on one 8-GPU node:
#   mooncake master          -> CPU (RDMA/TCP object store)
#   patched SGLang server    -> SERVER_GPU     (frozen 27B target -> mooncake)
#   producer (HTTP driver)   -> CPU            (prompts -> server, streams refs)
#   consumer (DFlash trainer)-> CONSUMER_GPUS  (DP=TRAIN_DP over per-rank inboxes)
#
# The consumer is DATA-PARALLEL: torchrun spawns TRAIN_DP ranks; rank 0 hosts the
# RefDistributor — the run's single book-keeper (one SQLite ledger, one channel
# reader) — and round-robin dispatches refs to per-rank inboxes; gradients sync
# via FSDP. The producer never touches the ledger (trainer-side only).
#
# The server is sglang 0.5.14 patched with patches/sglang/v0.5.14/spec-capture.patch
# (apply with scripts/apply_sglang_spec_capture_patch.sh). Feature tensors travel
# server -> mooncake -> trainer with no re-copy; only tiny SampleRefs cross the
# ref channel. Disaggregated sibling of run_qwen3.6_27b_dflash_online.sh.
#
# Prereqs: Qwen/Qwen3.6-27B weights + a chat dataset at $ROOT_DIR/cache/dataset/;
# the `mooncake` package + `mooncake_master` on PATH; export WANDB_API_KEY for
# consumer W&B logging (or pass --report-to none).
set -uxo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/scripts:${PYTHONPATH:-}"
cd "$ROOT_DIR"

SERVER_GPU=${SERVER_GPU:-0}
TRAIN_DP=${TRAIN_DP:-2}
CONSUMER_GPUS=${CONSUMER_GPUS:-"1,2"}
SERVER_PORT=${SERVER_PORT:-30000}
# DFlash target capture layers — must match dflash_config.target_layer_ids in
# the draft config below.
AUX_LAYER_IDS=${AUX_LAYER_IDS:-"1 16 31 46 61"}

# --- mooncake connection (server sink + producer + consumer share these) ---
export MOONCAKE_LOCAL_HOSTNAME=${MOONCAKE_LOCAL_HOSTNAME:-127.0.0.1}
export MOONCAKE_MASTER_SERVER_ADDR=${MOONCAKE_MASTER_SERVER_ADDR:-127.0.0.1:50051}
export MOONCAKE_METADATA_SERVER=${MOONCAKE_METADATA_SERVER:-http://127.0.0.1:8080/metadata}
export MOONCAKE_PROTOCOL=${MOONCAKE_PROTOCOL:-tcp}
# The server sink's segment is the ONLY object storage (features are
# hard-pinned: undersizing fails puts rather than evicting). Size it above the
# producer watermark: 256 in-flight x ~100MB/sample at seq 2048 ~= 26GB.
export MOONCAKE_GLOBAL_SEGMENT_SIZE=${MOONCAKE_GLOBAL_SEGMENT_SIZE:-$((64 << 30))}
# Producer/consumer are pure clients (no segment): objects must not live in a
# process that exits before training finishes.
export DISAGG_CLIENT_SEGMENT_SIZE=${DISAGG_CLIENT_SEGMENT_SIZE:-0}

export DISAGG_STORE_ID=${DISAGG_STORE_ID:-qwen36-dflash-disagg}
export DISAGG_SERVER_URL=http://127.0.0.1:$SERVER_PORT
export DISAGG_REF_CHANNEL=${DISAGG_REF_CHANNEL:-$ROOT_DIR/outputs/$DISAGG_STORE_ID/refs.jsonl}
# Trainer-side ONLY (rank-0 ledger + per-rank inboxes); the producer must not
# see the db — the launcher below passes it just to the consumer.
DISAGG_DB=${DISAGG_DB:-$ROOT_DIR/outputs/$DISAGG_STORE_ID/run.db}
DISAGG_INBOX_DIR=${DISAGG_INBOX_DIR:-$ROOT_DIR/outputs/$DISAGG_STORE_ID/inboxes}
export DISAGG_MAX_PROMPTS=${DISAGG_MAX_PROMPTS:-400}
export DISAGG_MAX_STEPS=${DISAGG_MAX_STEPS:-0}
export DISAGG_LOG_INTERVAL=${DISAGG_LOG_INTERVAL:-1}

rm -rf "$(dirname "$DISAGG_REF_CHANNEL")" "$DISAGG_DB"
mkdir -p "$(dirname "$DISAGG_REF_CHANNEL")"
: > "$DISAGG_REF_CHANNEL"

cleanup() { kill "${MASTER_PID:-}" "${SERVER_PID:-}" "${PRODUCER_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT

# --- mooncake master ---
mooncake_master --enable-http-metadata-server=true &
MASTER_PID=$!
sleep 3

# --- patched SGLang server: frozen 27B target, spec-capture on ---
CUDA_VISIBLE_DEVICES=$SERVER_GPU MOONCAKE_LOCAL_HOSTNAME=$MOONCAKE_LOCAL_HOSTNAME \
    python -m sglang.launch_server \
        --model-path Qwen/Qwen3.6-27B \
        --trust-remote-code \
        --skip-tokenizer-init \
        --mem-fraction-static 0.85 \
        --chunked-prefill-size -1 \
        --disable-radix-cache \
        --enable-spec-capture \
        --spec-capture-method dflash \
        --spec-capture-aux-layer-ids $AUX_LAYER_IDS \
        --port $SERVER_PORT &
SERVER_PID=$!

# wait for the server to come up before driving prompts
until curl -sf "http://127.0.0.1:$SERVER_PORT/health" > /dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "server died"; exit 1; fi
    sleep 5
done

# recipe matches run_qwen3.6_27b_dflash_online.sh; max-length 2048 for a bounded
# single-node demo. batch-size is PER RANK: global batch = batch_size * TRAIN_DP.
ARGS=(
    --target-model-path Qwen/Qwen3.6-27B
    --target-model-backend hf
    --trust-remote-code
    --draft-config-path "$ROOT_DIR/configs/qwen3.6-27b-dflash.json"
    --embedding-key model.language_model.embed_tokens.weight
    --lm-head-key lm_head.weight
    --mask-token-id 248070
    --train-data-path "$ROOT_DIR/cache/dataset/nemotron_v2_train.jsonl"
    --chat-template qwen3.5
    --max-length 2048
    --batch-size 1
    --learning-rate 6e-4
    --warmup-ratio 0.04
    --max-grad-norm 1.0
    --attention-backend flex_attention
    --block-size 16
    --num-anchors 512
    --loss-decay-gamma 7.0
    --num-epochs 1
    --seed 42
    --save-interval 1000000
)

LAUNCHER=$SCRIPT_DIR/run_disagg_dflash.py

# --- producer: CPU-only HTTP driver (no torch model, no process group) ---
DISAGG_ROLE=producer CUDA_VISIBLE_DEVICES="" \
    python "$LAUNCHER" "${ARGS[@]}" \
        --output-dir "$ROOT_DIR/outputs/qwen36-disagg-producer" &
PRODUCER_PID=$!

# --- consumer: DP=$TRAIN_DP trainer pool; rank 0 = distributor + ledger ---
CUDA_VISIBLE_DEVICES=$CONSUMER_GPUS DISAGG_ROLE=consumer \
    DISAGG_DB=$DISAGG_DB DISAGG_INBOX_DIR=$DISAGG_INBOX_DIR \
    torchrun --rdzv-backend c10d --rdzv-endpoint localhost:29702 \
        --nnodes 1 --nproc_per_node "$TRAIN_DP" "$LAUNCHER" "${ARGS[@]}" \
        --output-dir "$ROOT_DIR/outputs/qwen36-disagg-consumer" \
        --report-to wandb \
        --wandb-project qwen36-dflash-disagg \
        --wandb-name qwen36-27b-dflash-server-capture-dp$TRAIN_DP

wait $PRODUCER_PID
echo "DISAGG36-DONE"
