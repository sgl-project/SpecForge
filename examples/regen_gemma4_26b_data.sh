#!/usr/bin/env bash
# Regenerate training data for Gemma4-26B Eagle3.
#
# This script:
#   1. Launches SGLang server(s) for Gemma4-26B on available GPUs.
#   2. Waits for the server(s) to become healthy.
#   3. Runs regenerate_train_data.py with thinking-ratio support.
#   4. Shuts down the server(s) on exit.
#
# Usage:
#   bash examples/regen_gemma4_26b_data.sh
#
# Environment variables (override defaults):
#   MODEL            - HuggingFace model ID       (default: google/gemma-4-26b-a4b-it)
#   TP_SIZE          - Tensor-parallel size        (default: 2)
#   NUM_SERVERS      - Number of server instances  (default: 1)
#   BASE_PORT        - First server port           (default: 30000)
#   CONCURRENCY      - Requests per server         (default: 128)
#   MAX_TOKENS       - Max generation tokens       (default: 8192)
#   TEMPERATURE      - Sampling temperature        (default: 0.8)
#   THINKING_RATIO   - Fraction with thinking      (default: 0.7)
#   INPUT_FILE       - Input JSONL path            (required)
#   OUTPUT_FILE      - Output JSONL path           (required)
#   NUM_SAMPLES      - Max samples to process      (default: all)

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# ── Configurable defaults ────────────────────────────────────────────────────
MODEL="${MODEL:-google/gemma-4-26b-a4b-it}"
TP_SIZE="${TP_SIZE:-1}"
NUM_SERVERS="${NUM_SERVERS:-8}"
BASE_PORT="${BASE_PORT:-30000}"
CONCURRENCY="${CONCURRENCY:-128}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-1}"
THINKING_RATIO="${THINKING_RATIO:-0.7}"
INPUT_FILE="${INPUT_FILE:-$ROOT_DIR/cache/dataset/ultrachat_train.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-$ROOT_DIR/outputs/dataset/ultrachat_regen_gemma4.jsonl}"
NUM_SAMPLES="${NUM_SAMPLES:-}"

# ── Derived ──────────────────────────────────────────────────────────────────
TOTAL_GPUS=$(( TP_SIZE * NUM_SERVERS ))
AVAIL_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)

if [ "$AVAIL_GPUS" -lt "$TOTAL_GPUS" ]; then
    echo "Error: Need ${TOTAL_GPUS} GPUs (${NUM_SERVERS} servers x TP ${TP_SIZE}) but only ${AVAIL_GPUS} available."
    exit 1
fi

echo "============================================================"
echo "  Gemma4-26B Data Regeneration"
echo "============================================================"
echo "  Model:           ${MODEL}"
echo "  TP size:         ${TP_SIZE}"
echo "  Servers:         ${NUM_SERVERS}"
echo "  Ports:           ${BASE_PORT}..$(( BASE_PORT + (NUM_SERVERS - 1) * 10 ))"
echo "  Concurrency:     ${CONCURRENCY} per server"
echo "  Max tokens:      ${MAX_TOKENS}"
echo "  Temperature:     ${TEMPERATURE}"
echo "  Thinking ratio:  ${THINKING_RATIO}"
echo "  Input:           ${INPUT_FILE}"
echo "  Output:          ${OUTPUT_FILE}"
echo "============================================================"

# ── Cleanup on exit ──────────────────────────────────────────────────────────
SERVER_PIDS=()

cleanup() {
    echo ""
    echo "Shutting down SGLang server(s)..."
    for pid in "${SERVER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    # Wait briefly then force-kill stragglers
    sleep 2
    for pid in "${SERVER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    echo "All servers stopped."
}
trap cleanup EXIT

# ── Launch servers ───────────────────────────────────────────────────────────
SERVER_ADDRESSES=()

for i in $(seq 0 $(( NUM_SERVERS - 1 ))); do
    PORT=$(( BASE_PORT + i * 10 ))
    GPU_START=$(( i * TP_SIZE ))
    GPU_END=$(( GPU_START + TP_SIZE - 1 ))
    CUDA_DEVICES=$(seq -s, "$GPU_START" "$GPU_END")

    echo "Starting server $((i+1))/${NUM_SERVERS} on GPUs ${CUDA_DEVICES}, port ${PORT}..."

    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" /home/pyc_google_com/dev/gemma/.venv/bin/python -m sglang.launch_server \
        --model "${MODEL}" \
        --tp "${TP_SIZE}" \
        --port "${PORT}" \
        --host 0.0.0.0 \
        --cuda-graph-max-bs 128 \
        --trust-remote-code --enable-torch-compile \
        > "${ROOT_DIR}/cache/sglang_server_${PORT}.log" 2>&1 &

    SERVER_PIDS+=($!)
    SERVER_ADDRESSES+=("localhost:${PORT}")
done

# ── Wait for servers to be healthy ───────────────────────────────────────────
echo ""
echo "Waiting for servers to become healthy..."

wait_for_server() {
    local addr=$1
    local max_wait=600  # 10 minutes
    local elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        if curl -sf "http://${addr}/health" > /dev/null 2>&1; then
            return 0
        fi
        sleep 5
        elapsed=$(( elapsed + 5 ))
    done
    return 1
}

for addr in "${SERVER_ADDRESSES[@]}"; do
    if wait_for_server "$addr"; then
        echo "  ${addr} is healthy."
    else
        echo "Error: ${addr} did not become healthy within 10 minutes."
        echo "Check logs at: ${ROOT_DIR}/cache/sglang_server_*.log"
        exit 1
    fi
done

echo "All ${NUM_SERVERS} server(s) are ready."
echo "------------------------------------------------------------"

# ── Build regen command ──────────────────────────────────────────────────────
REGEN_ARGS=(
    python3 "${ROOT_DIR}/scripts/regenerate_train_data.py"
    --model "${MODEL}"
    --is-reasoning-model
    --thinking-ratio "${THINKING_RATIO}"
    --concurrency "${CONCURRENCY}"
    --max-tokens "${MAX_TOKENS}"
    --temperature "${TEMPERATURE}"
    --server-address "${SERVER_ADDRESSES[@]}"
    --input-file-path "${INPUT_FILE}"
    --output-file-path "${OUTPUT_FILE}"
    --resume
)

if [ -n "${NUM_SAMPLES}" ]; then
    REGEN_ARGS+=(--num-samples "${NUM_SAMPLES}")
fi

# ── Run regeneration ─────────────────────────────────────────────────────────
echo "Starting data regeneration..."
echo ""

mkdir -p "$(dirname "${OUTPUT_FILE}")"
"${REGEN_ARGS[@]}"

echo ""
echo "============================================================"
echo "  Done! Output saved to: ${OUTPUT_FILE}"
echo "============================================================"
