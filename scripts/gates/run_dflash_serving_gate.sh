#!/usr/bin/env bash
# Export one DFlash-family checkpoint and gate it through real SGLang serving.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")
# shellcheck source=scripts/gates/_e2e_common.sh
source "$SCRIPT_DIR/_e2e_common.sh"

usage() {
    cat <<'EOF'
Export a DFlash-family runtime checkpoint and run the strict real-serving gate.

Required environment:
  CHECKPOINT_PATH         Complete SpecForge runtime checkpoint directory.
  TARGET_MODEL_PATH       Target model path or Hugging Face repository.
  DRAFT_CONFIG_PATH       Matching DFlash-family draft JSON.
  PROMPT_ARTIFACT_PATH    Artifact written by select_overfit_sample.py.

Common overrides:
  WORK_DIR                Fresh export/server/result directory.
  SERVING_GPUS            CUDA devices for SGLang (default: 0).
  SERVING_TP              SGLang tensor parallel size (default: 1).
  SERVING_PORT            Local serving port (default: 30001).
  GATE_DRY_RUN            Print commands without exporting or starting a server.
EOF
}

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    usage
    exit 0
fi
[[ $# -eq 0 ]] || gate_fail "this gate accepts configuration through environment variables"

gate_install_cleanup_traps
gate_is_true "${GATE_DRY_RUN:-0}" GATE_DRY_RUN || true

CHECKPOINT_PATH=${CHECKPOINT_PATH:-}
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-}
DRAFT_CONFIG_PATH=${DRAFT_CONFIG_PATH:-}
PROMPT_ARTIFACT_PATH=${PROMPT_ARTIFACT_PATH:-}
PYTHON=${PYTHON:-python}
SGLANG_PYTHON=${SGLANG_PYTHON:-$PYTHON}
SPECFORGE_BIN=${SPECFORGE_BIN:-specforge}

gate_require_value "$CHECKPOINT_PATH" CHECKPOINT_PATH
gate_require_directory "$CHECKPOINT_PATH" CHECKPOINT_PATH
gate_require_value "$TARGET_MODEL_PATH" TARGET_MODEL_PATH
gate_require_value "$DRAFT_CONFIG_PATH" DRAFT_CONFIG_PATH
gate_require_file "$DRAFT_CONFIG_PATH" DRAFT_CONFIG_PATH
gate_require_value "$PROMPT_ARTIFACT_PATH" PROMPT_ARTIFACT_PATH
gate_require_file "$PROMPT_ARTIFACT_PATH" PROMPT_ARTIFACT_PATH
gate_require_command "$PYTHON" PYTHON

RUN_TAG=${RUN_TAG:-$(date +%Y%m%dT%H%M%S)}
WORK_DIR=${WORK_DIR:-$ROOT_DIR/outputs/dflash-serving-gate-$RUN_TAG}
EXPORT_DIR=$WORK_DIR/draft_hf
RESULT_PATH=$WORK_DIR/serving-gate.json
SERVER_LOG=$WORK_DIR/server.log
[[ ! -e "$WORK_DIR" ]] || gate_fail "WORK_DIR must be fresh: $WORK_DIR"

draft_metadata=$(
    "$PYTHON" -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
method = config.get("dflash_config") or {}
block_size = config.get("block_size")
projector = method.get("projector_type", "dflash")
if not isinstance(block_size, int) or block_size < 1:
    raise SystemExit("draft config must define a positive block_size")
print(f"{block_size}|{projector}")
' "$DRAFT_CONFIG_PATH"
)
IFS='|' read -r CONFIG_BLOCK_SIZE PROJECTOR_TYPE <<EOF
$draft_metadata
EOF
BLOCK_SIZE=${BLOCK_SIZE:-$CONFIG_BLOCK_SIZE}
gate_require_positive_integer "$BLOCK_SIZE" BLOCK_SIZE
[[ "$BLOCK_SIZE" == "$CONFIG_BLOCK_SIZE" ]] || {
    gate_fail "BLOCK_SIZE=$BLOCK_SIZE does not match draft config value $CONFIG_BLOCK_SIZE"
}

SERVING_GPUS=${SERVING_GPUS:-0}
SERVING_TP=${SERVING_TP:-1}
SERVING_PORT=${SERVING_PORT:-30001}
SERVING_HOST=${SERVING_HOST:-127.0.0.1}
SERVED_MODEL=${SERVED_MODEL:-specforge-target}
SERVING_CONTEXT_LENGTH=${SERVING_CONTEXT_LENGTH:-512}
SERVING_MAX_TOTAL_TOKENS=${SERVING_MAX_TOTAL_TOKENS:-1024}
SERVING_MEM_FRACTION=${SERVING_MEM_FRACTION:-0.8}
SERVING_ATTENTION_BACKEND=${SERVING_ATTENTION_BACKEND:-triton}
SERVING_MAX_TOKENS=${SERVING_MAX_TOKENS:-$BLOCK_SIZE}
REASONING_PARSER=${REASONING_PARSER:-}
SGLANG_ROOT=${SGLANG_ROOT:-}
SERVING_SERVER_EXTRA_ARGS=${SERVING_SERVER_EXTRA_ARGS:-}
EMBEDDING_KEY=${EMBEDDING_KEY:-model.embed_tokens.weight}

gate_require_positive_integer "$SERVING_TP" SERVING_TP
gate_require_positive_integer "$SERVING_PORT" SERVING_PORT
((SERVING_PORT <= 65535)) || gate_fail "SERVING_PORT must be <= 65535"
gate_require_positive_integer "$SERVING_CONTEXT_LENGTH" SERVING_CONTEXT_LENGTH
gate_require_positive_integer "$SERVING_MAX_TOTAL_TOKENS" SERVING_MAX_TOTAL_TOKENS
gate_require_positive_integer "$SERVING_MAX_TOKENS" SERVING_MAX_TOKENS

EXPORT_COMMAND=(
    "$SPECFORGE_BIN" export --to hf
    --checkpoint "$CHECKPOINT_PATH"
    --draft-config "$DRAFT_CONFIG_PATH"
    --output-dir "$EXPORT_DIR"
    --embedding-source "$TARGET_MODEL_PATH"
    --embedding-key "$EMBEDDING_KEY"
)

# SGLang's DFLASH loader dispatches both DFlash and Domino-compatible artifacts
# through DFlashDraftModel. Preserve method-specific fields in dflash_config.
NORMALIZE_COMMAND=(
    "$PYTHON" "$SCRIPT_DIR/normalize_dflash_export.py"
    --config "$EXPORT_DIR/config.json"
    --block-size "$BLOCK_SIZE"
)

SERVER_ENV=(
    env
    "CUDA_VISIBLE_DEVICES=$SERVING_GPUS"
    PYTHONUNBUFFERED=1
)
if [[ -n "$SGLANG_ROOT" ]]; then
    SERVER_ENV+=("PYTHONPATH=$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}")
fi

SERVER_COMMAND=(
    "${SERVER_ENV[@]}"
    "$SGLANG_PYTHON" -m sglang.launch_server
    --model-path "$TARGET_MODEL_PATH"
    --served-model-name "$SERVED_MODEL"
    --tp-size "$SERVING_TP"
    --dtype bfloat16
    --attention-backend "$SERVING_ATTENTION_BACKEND"
    --mem-fraction-static "$SERVING_MEM_FRACTION"
    --context-length "$SERVING_CONTEXT_LENGTH"
    --max-running-requests 1
    --max-total-tokens "$SERVING_MAX_TOTAL_TOKENS"
    --chunked-prefill-size -1
    --disable-radix-cache
    --disable-cuda-graph
    --trust-remote-code
    --host "$SERVING_HOST"
    --port "$SERVING_PORT"
    --speculative-algorithm DFLASH
    --speculative-draft-model-path "$EXPORT_DIR"
    --speculative-num-draft-tokens "$BLOCK_SIZE"
    --speculative-dflash-block-size "$BLOCK_SIZE"
)
if [[ -n "$REASONING_PARSER" ]]; then
    SERVER_COMMAND+=(--reasoning-parser "$REASONING_PARSER")
fi
if [[ -n "$SERVING_SERVER_EXTRA_ARGS" ]]; then
    read -r -a SERVING_EXTRA_ARGS <<EOF
$SERVING_SERVER_EXTRA_ARGS
EOF
    SERVER_COMMAND+=("${SERVING_EXTRA_ARGS[@]}")
fi

CHECK_COMMAND=(
    "$SGLANG_PYTHON" "$SCRIPT_DIR/run_dflash_chat_serving_gate.py"
    --server-url "http://$SERVING_HOST:$SERVING_PORT"
    --model-path "$TARGET_MODEL_PATH"
    --served-model "$SERVED_MODEL"
    --prompt-json-path "$PROMPT_ARTIFACT_PATH"
    --output-path "$RESULT_PATH"
    --block-size "$BLOCK_SIZE"
    --max-tokens "$SERVING_MAX_TOKENS"
)

if ! gate_dry_run; then
    gate_require_command "$SPECFORGE_BIN" SPECFORGE_BIN
    gate_require_command "$SGLANG_PYTHON" SGLANG_PYTHON
    gate_require_command curl curl
    gate_require_tcp_port_free \
        "$PYTHON" "$SERVING_HOST" "$SERVING_PORT" SERVING_PORT
    mkdir -p "$WORK_DIR"
fi

gate_run "${EXPORT_COMMAND[@]}"
gate_run "${NORMALIZE_COMMAND[@]}"
gate_start_service dflash-serving "$SERVER_LOG" "${SERVER_COMMAND[@]}"
SERVER_PID=$GATE_LAST_PID
gate_wait_http "$SERVER_PID" dflash-serving \
    "http://$SERVING_HOST:$SERVING_PORT/v1/models" "$SERVER_LOG" false
gate_run "${CHECK_COMMAND[@]}"

if gate_dry_run; then
    printf 'DFLASH-SERVING-GATE-PLAN: %s\n' "$RESULT_PATH"
else
    printf 'DFLASH-SERVING-GATE-PASSED: %s\n' "$RESULT_PATH"
fi
