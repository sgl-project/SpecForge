#!/usr/bin/env bash
# Export one Domino overfit checkpoint and gate it through real SGLang DFLASH.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")

: "${CHECKPOINT_PATH:?set CHECKPOINT_PATH to the overfit checkpoint directory}"
MODEL_PROFILE=${MODEL_PROFILE:-qwen3-8b}
case "$MODEL_PROFILE" in
    qwen3-8b)
        TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-Qwen/Qwen3-8B}
        DRAFT_CONFIG_PATH=${DRAFT_CONFIG_PATH:-$ROOT_DIR/configs/qwen3-8b-domino.json}
        EMBEDDING_KEY=${EMBEDDING_KEY:-model.embed_tokens.weight}
        SERVED_MODEL=${SERVED_MODEL:-qwen3-8b}
        SERVING_GPUS=${SERVING_GPUS:-${SERVING_GPU:-0}}
        SERVING_TP=${SERVING_TP:-1}
        REASONING_PARSER=${REASONING_PARSER:-}
        SERVING_CONTEXT_LENGTH=${SERVING_CONTEXT_LENGTH:-512}
        SERVING_MAX_TOTAL_TOKENS=${SERVING_MAX_TOTAL_TOKENS:-1024}
        ;;
    qwen3.6-27b)
        TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-Qwen/Qwen3.6-27B}
        DRAFT_CONFIG_PATH=${DRAFT_CONFIG_PATH:-$ROOT_DIR/configs/qwen3.6-27b-domino-full-attention.json}
        EMBEDDING_KEY=${EMBEDDING_KEY:-model.language_model.embed_tokens.weight}
        SERVED_MODEL=${SERVED_MODEL:-qwen3.6-27b}
        SERVING_GPUS=${SERVING_GPUS:-${SERVING_GPU:-0}}
        SERVING_TP=${SERVING_TP:-1}
        REASONING_PARSER=${REASONING_PARSER:-qwen3}
        SERVING_CONTEXT_LENGTH=${SERVING_CONTEXT_LENGTH:-2048}
        SERVING_MAX_TOTAL_TOKENS=${SERVING_MAX_TOTAL_TOKENS:-4096}
        ;;
    *)
        echo "Unknown MODEL_PROFILE=$MODEL_PROFILE (expected qwen3-8b or qwen3.6-27b)" >&2
        exit 2
        ;;
esac
: "${PROMPT_ARTIFACT_PATH:?set PROMPT_ARTIFACT_PATH to prompt_artifact.json}"
SPECFORGE_PYTHON=${SPECFORGE_PYTHON:-python}
SGLANG_PYTHON=${SGLANG_PYTHON:-python}
SGLANG_ROOT=${SGLANG_ROOT:-}
SERVING_PORT=${SERVING_PORT:-30001}
WORK_DIR=${WORK_DIR:-$ROOT_DIR/outputs/$MODEL_PROFILE-domino-real-serving-$(date +%Y%m%dT%H%M%S)}
EXPORT_DIR=$WORK_DIR/draft_hf
RESULT_PATH=$WORK_DIR/serving_gate.json
SERVER_LOG=$WORK_DIR/server.log

for path in "$CHECKPOINT_PATH" "$PROMPT_ARTIFACT_PATH" "$DRAFT_CONFIG_PATH"; do
    if [[ ! -e "$path" ]]; then
        echo "Required path does not exist: $path" >&2
        exit 2
    fi
done
if timeout 1 bash -c "</dev/tcp/127.0.0.1/$SERVING_PORT" 2>/dev/null; then
    echo "SERVING_PORT is already occupied; refusing to probe an existing server: $SERVING_PORT" >&2
    exit 2
fi
if [[ -e "$WORK_DIR" ]]; then
    echo "WORK_DIR already exists; serving gates require a fresh output: $WORK_DIR" >&2
    exit 2
fi
mkdir -p "$WORK_DIR"

mapfile -t DRAFT_VALUES < <("$SPECFORGE_PYTHON" - "$DRAFT_CONFIG_PATH" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
dflash = config.get("dflash_config") or {}
if dflash.get("projector_type") != "domino":
    raise SystemExit("draft config must set dflash_config.projector_type='domino'")
print(config["block_size"])
PY
)
BLOCK_SIZE=${BLOCK_SIZE:-${DRAFT_VALUES[0]}}

PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$SPECFORGE_PYTHON" -m specforge.export.to_hf \
    --checkpoint "$CHECKPOINT_PATH" \
    --draft-config "$DRAFT_CONFIG_PATH" \
    --output-dir "$EXPORT_DIR" \
    --embedding-source "$TARGET_MODEL_PATH" \
    --embedding-key "$EMBEDDING_KEY"

# SGLang dispatches this compatible HF artifact through its DFLASH loader.
"$SPECFORGE_PYTHON" - "$EXPORT_DIR/config.json" "$BLOCK_SIZE" <<'PY'
import json
import sys

path = sys.argv[1]
block_size = int(sys.argv[2])
with open(path, encoding="utf-8") as handle:
    config = json.load(handle)
config["architectures"] = ["DFlashDraftModel"]
config.pop("auto_map", None)
with open(path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
    handle.write("\n")
dflash = config.get("dflash_config") or {}
if config.get("block_size") != block_size or dflash.get("projector_type") != "domino":
    raise SystemExit(f"exported checkpoint is not a block-{block_size} Domino draft")
PY

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

SGLANG_PYTHONPATH=()
if [[ -n "$SGLANG_ROOT" ]]; then
    SGLANG_PYTHONPATH=(PYTHONPATH="$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}")
fi
REASONING_PARSER_ARGS=()
if [[ -n "$REASONING_PARSER" ]]; then
    REASONING_PARSER_ARGS=(--reasoning-parser "$REASONING_PARSER")
fi
setsid env \
    CUDA_VISIBLE_DEVICES="$SERVING_GPUS" \
    PYTHONUNBUFFERED=1 \
    "${SGLANG_PYTHONPATH[@]}" \
    "$SGLANG_PYTHON" -m sglang.launch_server \
    --model-path "$TARGET_MODEL_PATH" \
    --served-model-name "$SERVED_MODEL" \
    --tp-size "$SERVING_TP" \
    --dtype bfloat16 \
    --attention-backend "${SERVING_ATTENTION_BACKEND:-triton}" \
    --mem-fraction-static "${SERVING_MEM_FRACTION:-0.8}" \
    --context-length "$SERVING_CONTEXT_LENGTH" \
    --max-running-requests 1 \
    --max-total-tokens "$SERVING_MAX_TOTAL_TOKENS" \
    --chunked-prefill-size -1 \
    --disable-radix-cache \
    --disable-cuda-graph \
    --trust-remote-code \
    --host 127.0.0.1 \
    --port "$SERVING_PORT" \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path "$EXPORT_DIR" \
    --speculative-num-draft-tokens "$BLOCK_SIZE" \
    --speculative-dflash-block-size "$BLOCK_SIZE" \
    "${REASONING_PARSER_ARGS[@]}" \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 300); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "SGLang server exited during startup; see $SERVER_LOG" >&2
        exit 1
    fi
    if curl -fsS "http://127.0.0.1:$SERVING_PORT/v1/models" >/dev/null; then
        break
    fi
    sleep 2
done
if ! curl -fsS "http://127.0.0.1:$SERVING_PORT/v1/models" >/dev/null; then
    echo "SGLang server did not become ready; see $SERVER_LOG" >&2
    exit 1
fi

PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$SGLANG_PYTHON" "$ROOT_DIR/scripts/run_qwen3_8b_dflash_serving_gate.py" \
    --server-url "http://127.0.0.1:$SERVING_PORT" \
    --model-path "$TARGET_MODEL_PATH" \
    --served-model "$SERVED_MODEL" \
    --prompt-json-path "$PROMPT_ARTIFACT_PATH" \
    --output-path "$RESULT_PATH" \
    --block-size "$BLOCK_SIZE" \
    --max-tokens "$BLOCK_SIZE"

echo "DOMINO-REAL-DFLASH-SERVING-PASSED profile=$MODEL_PROFILE: $RESULT_PATH"
