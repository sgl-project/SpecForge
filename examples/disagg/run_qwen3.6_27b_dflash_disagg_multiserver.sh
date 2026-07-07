#!/bin/bash
# Qwen3.6-27B DFlash, ONLINE disaggregated, MULTI-SERVER on one 8-GPU node:
#   mooncake master           -> CPU  (RDMA/TCP object store)
#   patched SGLang server 0   -> GPUs 0,1 (TP=2, frozen 27B target -> mooncake)
#   patched SGLang server 1   -> GPUs 2,3 (TP=2, same model + capture flags)
#   producer (HTTP driver)    -> CPU  (fans prompts out to BOTH servers)
#   consumer (DFlash trainer) -> GPUs 4,5 (DP=2 over per-rank inboxes)
#
# Multi-server sibling of run_qwen3.6_27b_dflash_disagg.sh. The producer builds
# one SGLangServerCaptureAdapter per URL in DISAGG_SERVER_URLS; each adapter is
# driven by its own RolloutWorker on its own thread, leasing DISJOINT prompts
# from the one controller — both servers prefill concurrently. Every server
# registers a segment with the ONE mooncake master, so the trainer fetches any
# sample by key regardless of which server captured it. A server that dies is
# dropped after its in-flight prompts are returned to the pool; the survivor
# finishes the run.
#
# Both servers MUST be launched with identical model + capture flags (the
# producer's FeatureContract check fails loudly per-sample otherwise).
#
# Prereqs: identical to run_qwen3.6_27b_dflash_disagg.sh (patched sglang 0.5.14,
# mooncake_master on PATH, dataset + weights cached, WANDB_API_KEY or
# --report-to none).
set -uxo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/scripts:${PYTHONPATH:-}"
cd "$ROOT_DIR"

# --- topology: 2 servers x TP=2 + DP=2 trainer (override via env) ---
SERVER0_GPUS=${SERVER0_GPUS:-"0,1"}
SERVER1_GPUS=${SERVER1_GPUS:-"2,3"}
SERVER_TP=${SERVER_TP:-2}
SERVER0_PORT=${SERVER0_PORT:-30000}
SERVER1_PORT=${SERVER1_PORT:-30001}
TRAIN_DP=${TRAIN_DP:-2}
CONSUMER_GPUS=${CONSUMER_GPUS:-"4,5"}
# DFlash target capture layers — must match dflash_config.target_layer_ids in
# the draft config below, on BOTH servers.
AUX_LAYER_IDS=${AUX_LAYER_IDS:-"1 16 31 46 61"}

# --- mooncake connection (server sinks + producer + consumer share these) ---
export MOONCAKE_LOCAL_HOSTNAME=${MOONCAKE_LOCAL_HOSTNAME:-127.0.0.1}
export MOONCAKE_MASTER_SERVER_ADDR=${MOONCAKE_MASTER_SERVER_ADDR:-127.0.0.1:50051}
export MOONCAKE_METADATA_SERVER=${MOONCAKE_METADATA_SERVER:-http://127.0.0.1:8080/metadata}
export MOONCAKE_PROTOCOL=${MOONCAKE_PROTOCOL:-tcp}
# EACH server contributes a segment of this size to the one master; objects are
# hard-pinned (undersizing fails puts rather than evicting). The producer
# watermark's in-flight bytes now spread across N segments, but placement is
# not balanced — keep per-server headroom above watermark/N (skew margin).
export MOONCAKE_GLOBAL_SEGMENT_SIZE=${MOONCAKE_GLOBAL_SEGMENT_SIZE:-$((48 << 30))}
# Producer/consumer are pure clients (no segment): objects must not live in a
# process that exits before training finishes.
export DISAGG_CLIENT_SEGMENT_SIZE=${DISAGG_CLIENT_SEGMENT_SIZE:-0}

export DISAGG_STORE_ID=${DISAGG_STORE_ID:-qwen36-dflash-disagg-2srv}
export DISAGG_SERVER_URLS="http://127.0.0.1:$SERVER0_PORT,http://127.0.0.1:$SERVER1_PORT"
export DISAGG_REF_CHANNEL=${DISAGG_REF_CHANNEL:-$ROOT_DIR/outputs/$DISAGG_STORE_ID/refs.jsonl}
# Trainer-side ONLY (rank-0 ledger + per-rank inboxes); the producer must not
# see the db — the launcher below passes it just to the consumer.
DISAGG_DB=${DISAGG_DB:-$ROOT_DIR/outputs/$DISAGG_STORE_ID/run.db}
DISAGG_INBOX_DIR=${DISAGG_INBOX_DIR:-$ROOT_DIR/outputs/$DISAGG_STORE_ID/inboxes}
export DISAGG_MAX_PROMPTS=${DISAGG_MAX_PROMPTS:-400}
export DISAGG_MAX_STEPS=${DISAGG_MAX_STEPS:-0}
export DISAGG_LOG_INTERVAL=${DISAGG_LOG_INTERVAL:-1}
REPORT_TO=${REPORT_TO:-wandb}  # set REPORT_TO=none to run without W&B

rm -rf "$(dirname "$DISAGG_REF_CHANNEL")" "$DISAGG_DB"
mkdir -p "$(dirname "$DISAGG_REF_CHANNEL")"
: > "$DISAGG_REF_CHANNEL"

cleanup() { kill "${MASTER_PID:-}" "${SERVER0_PID:-}" "${SERVER1_PID:-}" "${PRODUCER_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT

# --- mooncake master ---
mooncake_master --enable-http-metadata-server=true &
MASTER_PID=$!
sleep 3

# --- patched SGLang servers: frozen 27B target, TP=2 each, spec-capture on ---
launch_server() { # $1=gpus $2=port
    CUDA_VISIBLE_DEVICES=$1 MOONCAKE_LOCAL_HOSTNAME=$MOONCAKE_LOCAL_HOSTNAME \
        python -m sglang.launch_server \
            --model-path Qwen/Qwen3.6-27B \
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

# wait for BOTH servers before driving prompts (die fast if either dies)
for port_pid in "$SERVER0_PORT:$SERVER0_PID" "$SERVER1_PORT:$SERVER1_PID"; do
    port=${port_pid%%:*}; pid=${port_pid##*:}
    until curl -sf "http://127.0.0.1:$port/health" > /dev/null; do
        if ! kill -0 "$pid" 2>/dev/null; then echo "server on :$port died"; exit 1; fi
        sleep 5
    done
done

# recipe matches run_qwen3.6_27b_dflash_disagg.sh; max-length 2048 for a bounded
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

# --- producer: CPU-only HTTP driver fanning out to both servers ---
DISAGG_ROLE=producer CUDA_VISIBLE_DEVICES="" \
    python "$LAUNCHER" "${ARGS[@]}" \
        --output-dir "$ROOT_DIR/outputs/qwen36-disagg-2srv-producer" &
PRODUCER_PID=$!

# --- consumer: DP=$TRAIN_DP trainer pool; rank 0 = distributor + ledger ---
CUDA_VISIBLE_DEVICES=$CONSUMER_GPUS DISAGG_ROLE=consumer \
    DISAGG_DB=$DISAGG_DB DISAGG_INBOX_DIR=$DISAGG_INBOX_DIR \
    torchrun --rdzv-backend c10d --rdzv-endpoint localhost:29702 \
        --nnodes 1 --nproc_per_node "$TRAIN_DP" "$LAUNCHER" "${ARGS[@]}" \
        --output-dir "$ROOT_DIR/outputs/qwen36-disagg-2srv-consumer" \
        --report-to "$REPORT_TO" \
        --wandb-project qwen36-dflash-disagg \
        --wandb-name qwen36-27b-dflash-2srv-tp2-dp$TRAIN_DP

wait $PRODUCER_PID
echo "DISAGG36-2SRV-DONE"
