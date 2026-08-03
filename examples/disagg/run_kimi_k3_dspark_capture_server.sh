#!/usr/bin/env bash
# Launch one Kimi-K3 TP8 DSpark capture replica for the pipelined recipes.

set -euo pipefail

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the node-local Kimi-K3 snapshot}
CAPTURE_IP=${CAPTURE_IP:?set CAPTURE_IP to this node routable address}
MOONCAKE_MASTER_IP=${MOONCAKE_MASTER_IP:?set MOONCAKE_MASTER_IP}
SGLANG_ROOT=${SGLANG_ROOT:?set SGLANG_ROOT to the patched SGLang checkout}
SERVER_PORT=${SERVER_PORT:-30000}
AUX_LAYER_IDS=${AUX_LAYER_IDS:-"11 23 47 71 83"}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-16}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-73728}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-40960}
MAX_MAMBA_CACHE_SIZE=${MAX_MAMBA_CACHE_SIZE:-80}

[[ -f "$MODEL_PATH/config.json" ]] || {
    printf 'missing target config: %s/config.json\n' "$MODEL_PATH" >&2
    exit 1
}
[[ -f "$SGLANG_ROOT/python/sglang/srt/spec_capture_sink.py" ]] || {
    printf 'SGLang checkout is not patched for spec capture: %s\n' "$SGLANG_ROOT" >&2
    exit 1
}

export PYTHONPATH="$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export MOONCAKE_MASTER_SERVER_ADDR="$MOONCAKE_MASTER_IP:35551"
export MOONCAKE_METADATA_SERVER="http://$MOONCAKE_MASTER_IP:35880/metadata"
export MOONCAKE_LOCAL_HOSTNAME="$CAPTURE_IP"
export MC_TCP_BIND_ADDRESS="$CAPTURE_IP"
export MC_TRANSFER_TIMEOUT=${MC_TRANSFER_TIMEOUT:-300}
export MOONCAKE_PROTOCOL=${MOONCAKE_PROTOCOL:-tcp}
export MOONCAKE_GLOBAL_SEGMENT_SIZE=${MOONCAKE_GLOBAL_SEGMENT_SIZE:-1099511627776}
export MOONCAKE_LOCAL_BUFFER_SIZE=${MOONCAKE_LOCAL_BUFFER_SIZE:-1073741824}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
# Retain at most two host-side capture batches: Mooncake publishes batch N on
# its background writer while TP8 computes prefill N+1.  Larger queues retain
# tens of GiB per batch without increasing a single writer's throughput.
export SGLANG_SPEC_CAPTURE_MAX_PENDING_BATCHES=${SGLANG_SPEC_CAPTURE_MAX_PENDING_BATCHES:-2}

# AUX_LAYER_IDS is a trusted operator-provided whitespace-separated integer
# list and intentionally expands to separate CLI values. Capture workers spend
# almost all of their time in full-prompt prefill, so skip the long K3 decode
# CUDA-graph compile that cannot accelerate the one token completing /generate.
# Leave SGLang symmetric memory off: the latest K3 branch can then auto-enable
# its fused CustomAllReduceV2 path; symmetric memory disables that faster path.
# shellcheck disable=SC2086
exec python3 -m sglang.launch_server \
    --host 0.0.0.0 \
    --port "$SERVER_PORT" \
    --model-path "$MODEL_PATH" \
    --trust-remote-code \
    --skip-tokenizer-init \
    --tp-size 8 \
    --mem-fraction-static 0.76 \
    --context-length 4608 \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --max-total-tokens "$MAX_TOTAL_TOKENS" \
    --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
    --prefill-attention-backend flashinfer \
    --decode-attention-backend trtllm_mla \
    --moe-runner-backend marlin \
    --mamba-radix-cache-strategy extra_buffer \
    --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE" \
    --disable-cuda-graph \
    --chunked-prefill-size -1 \
    --enable-spec-capture \
    --spec-capture-method dspark \
    --spec-capture-aux-layer-ids $AUX_LAYER_IDS
