#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Regenerate Qwen3.5-122B-A10B SFT data with quality-aligned settings.
#
# Background: the v1 regen (run_regenerate_data_test.sh) silently produced
# ~28.6% truncated (content=None) samples and ~84.3% /no_think violations,
# which broke MTP training (mean accept length stuck at 2.7). This v2 fixes:
#
#   1. --respect-no-think    : pass {enable_thinking: False} when the user
#                              prompt contains '/no_think', so the SFT model
#                              actually obeys the directive at regen time.
#   2. --drop-truncated      : drop samples that hit max_tokens (no final
#                              answer) instead of writing them as success.
#   3. --inline-reasoning-into-content : merge reasoning_content back into a
#                              <think>...</think> block inside content, so the
#                              regen output exactly matches the original SFT
#                              data layout (a single string in 'content',
#                              no standalone reasoning_content field).
#   4. Larger --max-tokens   : 16384 to leave room for any real reasoning.
#
# Usage:
#   bash run_regenerate_data_v2.sh sanity   # 1000 samples + validate (fail-fast)
#   bash run_regenerate_data_v2.sh full     # full regen + validate
#
# Prerequisite: sglang server already running at $SERVER_ADDRESS.
# Recommended sglang launch (separate terminal, on the 122B target):
#
#   python3 -m sglang.launch_server \
#     --model /mnt/nj-larc/dataset/xiaowen/model/qwen35-122b \
#     --tp 8 --dtype bfloat16 \
#     --host 0.0.0.0 --port 8081 \
#     --reasoning-parser qwen3
#
# (Keeping --reasoning-parser qwen3 is fine: this script will inline
#  reasoning_content back into content downstream.)
# ----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")
export https_proxy=10.140.24.177:3128

MODE=${1:-sanity}

INPUT=/mnt/nj-larc/dataset/xiaowen/data/w1w/train.jsonl
SERVER_ADDRESS=${SERVER_ADDRESS:-127.0.0.1:8081}
MODEL_PATH=${MODEL_PATH:-/mnt/nj-larc/dataset/xiaowen/model/qwen35-122b}
CONCURRENCY=${CONCURRENCY:-32}
MAX_TOKENS=${MAX_TOKENS:-16384}

case "$MODE" in
  sanity)
    OUTPUT=/mnt/tidal-alsh01/dataset/xiaowen/data/w1w/train_regen_v2_sanity.jsonl
    NUM_SAMPLES=1000
    STRICT=--strict
    ;;
  full)
    OUTPUT=/mnt/tidal-alsh01/dataset/xiaowen/data/w1w/train_regen_v2.jsonl
    NUM_SAMPLES=100000
    STRICT=--strict
    ;;
  *)
    echo "Usage: bash $0 [sanity|full]"
    echo "  sanity : regen 1000 samples then run validator (fail-fast)"
    echo "  full   : full regen then run validator"
    exit 1
    ;;
esac

echo "=========================================================================="
echo "  MODE         : $MODE"
echo "  INPUT        : $INPUT"
echo "  OUTPUT       : $OUTPUT"
echo "  MODEL_PATH   : $MODEL_PATH"
echo "  SERVER       : $SERVER_ADDRESS"
echo "  CONCURRENCY  : $CONCURRENCY"
echo "  MAX_TOKENS   : $MAX_TOKENS"
echo "  NUM_SAMPLES  : $NUM_SAMPLES"
echo "=========================================================================="

# ---- Step 1: regen ----
python3 \
    "$ROOT_DIR/scripts/regenerate_train_data.py" \
    --model "$MODEL_PATH" \
    --input-file-path "$INPUT" \
    --output-file-path "$OUTPUT" \
    --concurrency "$CONCURRENCY" \
    --max-tokens "$MAX_TOKENS" \
    --num-samples "$NUM_SAMPLES" \
    --temperature 0.2 \
    --top-p 0.3 \
    --server-address "$SERVER_ADDRESS" \
    --resume \
    --is-reasoning-model \
    --respect-no-think \
    --drop-truncated \
    --inline-reasoning-into-content

# ---- Step 2: validate ----
echo
echo "=========================================================================="
echo "  Validating regen output against original SFT distribution"
echo "=========================================================================="
python3 "$ROOT_DIR/scripts/validate_regen_data.py" \
    --regen-file "$OUTPUT" \
    --reference-file "$INPUT" \
    --max-null-content-rate 0.0 \
    --max-empty-content-rate 0.0 \
    --max-rc-field-rate 0.0 \
    --max-no-think-violation-rate 0.05 \
    $STRICT

echo
echo "Done. If validator passed, next steps:"
echo "  1. Re-run scripts/prepare_hidden_states.py with --data-path $OUTPUT"
echo "  2. Point train script's --train-data-path to $OUTPUT"
echo "  3. Retrain MTP and watch wandb acc_0 vs the previous 0.82 baseline"
