#!/usr/bin/env bash
# Disaggregated offline EAGLE3 (Qwen2.5-7B) across two pools sharing a mount.
#
# Drive both nodes at once with rcli (node 0 ingests features into the shared
# store; node 1 trains the draft from it):
#
#   rcli exec --per-node <job> 'bash examples/disagg/run_qwen2.5_7b_eagle3_disagg.sh'
#
# Prereq: features already generated on node 0 at $FEATURES_DIR (see README).
set -euo pipefail

: "${SF_HOME:=/root/SpecForge}"
: "${TARGET_MODEL:=Qwen/Qwen2.5-7B-Instruct}"
: "${DRAFT_CONFIG:=$SF_HOME/configs/qwen2.5-7b-eagle3.json}"
: "${PROMPTS:=/root/disagg/prompts.jsonl}"
: "${FEATURES_DIR:=/root/disagg/features}"
: "${OUTPUT_DIR:=/root/disagg/out}"
: "${MAX_STEPS:=200}"
: "${NUM_EPOCHS:=10}"
: "${TTT_LENGTH:=7}"
: "${NPROC:=1}"
: "${CHAT_TEMPLATE:=qwen}"
: "${LEARNING_RATE:=5e-5}"
: "${CACHE_DIR:=$SF_HOME/cache}"
: "${DISAGG_STORE_ROOT:=/workspace/disagg_store}"
: "${DISAGG_MANIFEST:=/workspace/disagg_store/refs.json}"

# shared store + auth (both pools), and image/cache env
export DISAGG_STORE_ROOT DISAGG_MANIFEST
export DISAGG_STORE_ID="${DISAGG_STORE_ID:-eagle3-disagg}"
export DISAGG_AUTH_TOKEN="${DISAGG_AUTH_TOKEN:-disagg-secret}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HOME=/root HF_HOME=/root/.cache/huggingface TRITON_CACHE_DIR=/root/.triton
export PYTHONPATH="$SF_HOME:$SF_HOME/scripts:${PYTHONPATH:-}"
cd "$SF_HOME"

COMMON=(
  --target-model-path "$TARGET_MODEL"
  --target-model-backend hf
  --draft-model-config "$DRAFT_CONFIG"
  --train-data-path "$PROMPTS"
  --train-hidden-states-path "$FEATURES_DIR"
  --output-dir "$OUTPUT_DIR"
  --chat-template "$CHAT_TEMPLATE"
  --cache-dir "$CACHE_DIR"
  --attention-backend flex_attention
  --ttt-length "$TTT_LENGTH"
  --max-num-steps "$MAX_STEPS"
  --num-epochs "$NUM_EPOCHS"
  --batch-size 1
  --learning-rate "$LEARNING_RATE"
  --seed 0
)

if [ "${RCLI_NODE_RANK:-0}" = "0" ]; then
  echo "[node0] PRODUCER: ingest $FEATURES_DIR -> $DISAGG_STORE_ROOT"
  DISAGG_ROLE=producer python examples/disagg/run_disagg_eagle3.py "${COMMON[@]}"
else
  echo "[node1] CONSUMER: train from shared store ($NPROC gpu)"
  DISAGG_ROLE=consumer torchrun --standalone --nproc_per_node "$NPROC" \
    examples/disagg/run_disagg_eagle3.py "${COMMON[@]}"
fi
