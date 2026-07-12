#!/bin/bash
# Strict one-sample Qwen3/Qwen3.6 Domino disaggregated overfit gate:
# one SGLang capture server and one Domino trainer.
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/scripts:${PYTHONPATH:-}"
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/cache/compiled_kernels}
export FLASHINFER_DISABLE_VERSION_CHECK=1

PYTHON=${PYTHON:-python}
MODEL_PROFILE=${MODEL_PROFILE:-qwen3-8b}
case "$MODEL_PROFILE" in
    qwen3-8b)
        TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-Qwen/Qwen3-8B}
        DRAFT_CONFIG_PATH=${DRAFT_CONFIG_PATH:-$ROOT_DIR/configs/qwen3-8b-domino.json}
        SOURCE_DATA_PATH=${SOURCE_DATA_PATH:-$ROOT_DIR/cache/dataset/sharegpt_train_regen_qwen3_8b_temperature0_non_reasoning.jsonl}
        CHAT_TEMPLATE=${CHAT_TEMPLATE:-qwen}
        EMBEDDING_KEY=${EMBEDDING_KEY:-model.embed_tokens.weight}
        LM_HEAD_KEY=${LM_HEAD_KEY:-lm_head.weight}
        REASONING_POLICY=${REASONING_POLICY:-forbidden}
        ENABLE_THINKING=${ENABLE_THINKING:-0}
        MAX_LENGTH=${MAX_LENGTH:-512}
        SERVER_GPUS=${SERVER_GPUS:-0}
        SERVER_TP=${SERVER_TP:-1}
        REAL_SERVING_GPUS=${REAL_SERVING_GPUS:-0}
        REAL_SERVING_TP=${REAL_SERVING_TP:-1}
        ;;
    qwen3.6-27b)
        TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-Qwen/Qwen3.6-27B}
        DRAFT_CONFIG_PATH=${DRAFT_CONFIG_PATH:-$ROOT_DIR/configs/qwen3.6-27b-domino-full-attention.json}
        SOURCE_DATA_PATH=${SOURCE_DATA_PATH:-$ROOT_DIR/cache/dataset/sharegpt_train_regen_qwen3.6-27b_temperature0_reasoning.jsonl}
        CHAT_TEMPLATE=${CHAT_TEMPLATE:-qwen3.5}
        EMBEDDING_KEY=${EMBEDDING_KEY:-model.language_model.embed_tokens.weight}
        LM_HEAD_KEY=${LM_HEAD_KEY:-lm_head.weight}
        REASONING_POLICY=${REASONING_POLICY:-required}
        ENABLE_THINKING=${ENABLE_THINKING:-1}
        MAX_LENGTH=${MAX_LENGTH:-2048}
        SERVER_GPUS=${SERVER_GPUS:-0}
        SERVER_TP=${SERVER_TP:-1}
        REAL_SERVING_GPUS=${REAL_SERVING_GPUS:-0}
        REAL_SERVING_TP=${REAL_SERVING_TP:-1}
        ;;
    *)
        echo "Unknown MODEL_PROFILE=$MODEL_PROFILE (expected qwen3-8b or qwen3.6-27b)" >&2
        exit 2
        ;;
esac

SERVER_PORT=${SERVER_PORT:-30000}
TRAIN_DP=${TRAIN_DP:-7}
CONSUMER_GPUS=${CONSUMER_GPUS:-"1,2,3,4,5,6,7"}

RUN_TAG=${RUN_TAG:-$(date +%Y%m%dT%H%M%S)}
export DISAGG_STORE_ID=${DISAGG_STORE_ID:-$MODEL_PROFILE-domino-overfit-$RUN_TAG}
WORK_DIR=${WORK_DIR:-$ROOT_DIR/outputs/$DISAGG_STORE_ID}
PRODUCER_OUTPUT_DIR=$WORK_DIR/producer
CONSUMER_OUTPUT_DIR=$WORK_DIR/consumer
SINGLE_DATA_PATH=$WORK_DIR/single_sample.jsonl
PROMPT_ARTIFACT_PATH=$WORK_DIR/prompt_artifact.json
TRAIN_LOG=$WORK_DIR/train.log

for path in "$SOURCE_DATA_PATH" "$DRAFT_CONFIG_PATH"; do
    if [[ ! -e "$path" ]]; then
        echo "Required path does not exist: $path" >&2
        exit 2
    fi
done
if timeout 1 bash -c "</dev/tcp/127.0.0.1/$SERVER_PORT" 2>/dev/null; then
    echo "SERVER_PORT is already occupied; refusing to use an existing capture server: $SERVER_PORT" >&2
    exit 2
fi
if [[ -e "$WORK_DIR" ]]; then
    echo "WORK_DIR already exists; overfit gates require a fresh output: $WORK_DIR" >&2
    exit 2
fi
mkdir -p "$WORK_DIR"

mapfile -t DRAFT_VALUES < <("$PYTHON" - "$DRAFT_CONFIG_PATH" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
dflash = config.get("dflash_config") or {}
if dflash.get("projector_type") != "domino":
    raise SystemExit("draft config must set dflash_config.projector_type='domino'")
print(dflash["mask_token_id"])
print(config["block_size"])
print(" ".join(str(layer) for layer in dflash["target_layer_ids"]))
PY
)
MASK_TOKEN_ID=${MASK_TOKEN_ID:-${DRAFT_VALUES[0]}}
BLOCK_SIZE=${BLOCK_SIZE:-${DRAFT_VALUES[1]}}
AUX_LAYER_IDS=${AUX_LAYER_IDS:-${DRAFT_VALUES[2]}}
MIN_LOSS_TOKENS=$((2 * BLOCK_SIZE))

SELECT_ARGS=()
if [[ "$ENABLE_THINKING" == "1" || "$ENABLE_THINKING" == "true" ]]; then
    SELECT_ARGS+=(--enable-thinking)
fi
"$PYTHON" "$ROOT_DIR/scripts/create_qwen3_8b_overfit_data.py" \
    --input-data-path "$SOURCE_DATA_PATH" \
    --output-data-path "$SINGLE_DATA_PATH" \
    --model-path "$TARGET_MODEL_PATH" \
    --chat-template "$CHAT_TEMPLATE" \
    --max-length "$MAX_LENGTH" \
    --min-loss-tokens "$MIN_LOSS_TOKENS" \
    --min-assistant-chars "${MIN_ASSISTANT_CHARS:-128}" \
    --train-only-last-turn \
    --reasoning-policy "$REASONING_POLICY" \
    --prompt-output-path "$PROMPT_ARTIFACT_PATH" \
    --require-untruncated \
    "${SELECT_ARGS[@]}"

export DISAGG_MAX_PROMPTS=${DISAGG_MAX_PROMPTS:-1}
export DISAGG_MAX_STEPS=${DISAGG_MAX_STEPS:-400}
if [[ "$DISAGG_MAX_PROMPTS" -ne 1 || "$DISAGG_MAX_STEPS" -le 0 ]]; then
    echo "overfit requires DISAGG_MAX_PROMPTS=1 and DISAGG_MAX_STEPS>0" >&2
    exit 2
fi
# Each optimizer step consumes one batch on every DP rank. Replaying the one
# prompt MAX_STEPS*TRAIN_DP times keeps the producer alive for the whole gate.
NUM_EPOCHS=$((DISAGG_MAX_STEPS * TRAIN_DP))
export DISAGG_TOTAL_STEPS=$DISAGG_MAX_STEPS
export DISAGG_LOG_INTERVAL=${DISAGG_LOG_INTERVAL:-1}

MAX_LOSS=${MAX_LOSS:-0.0001}
MIN_ACCURACY=${MIN_ACCURACY:-1.0}

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

export DISAGG_SERVER_URLS=http://127.0.0.1:$SERVER_PORT
export DISAGG_REF_CHANNEL=$WORK_DIR/refs.jsonl
DISAGG_DB=$WORK_DIR/run.db
DISAGG_INBOX_DIR=$WORK_DIR/inboxes
MOONCAKE_LOG=$WORK_DIR/mooncake.log
: > "$DISAGG_REF_CHANNEL"

cleanup() {
    kill "${SERVER_PID:-}" "${PRODUCER_PID:-}" 2>/dev/null || true
    if [[ -n "${MASTER_PID:-}" ]]; then
        kill -- "-$MASTER_PID" 2>/dev/null || kill "$MASTER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

setsid mooncake_master \
    --enable_http_metadata_server=true \
    --rpc_port="$MOONCAKE_RPC_PORT" \
    --http_metadata_server_port="$MOONCAKE_HTTP_PORT" \
    --metrics_port="$MOONCAKE_METRICS_PORT" \
    >"$MOONCAKE_LOG" 2>&1 &
MASTER_PID=$!

for _ in $(seq 1 30); do
    if ! kill -0 "$MASTER_PID" 2>/dev/null; then
        echo "Mooncake master exited during startup; see $MOONCAKE_LOG" >&2
        exit 1
    fi
    if curl -sS --max-time 1 -o /dev/null \
        "$MOONCAKE_METADATA_SERVER?key=specforge-health-check" \
        && timeout 1 bash -c \
            "</dev/tcp/$MOONCAKE_HOST/$MOONCAKE_RPC_PORT" 2>/dev/null; then
        break
    fi
    sleep 1
done
if ! timeout 1 bash -c "</dev/tcp/$MOONCAKE_HOST/$MOONCAKE_RPC_PORT" 2>/dev/null; then
    echo "Mooncake master did not become ready; see $MOONCAKE_LOG" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES=$SERVER_GPUS \
    "$PYTHON" -m sglang.launch_server \
        --model-path "$TARGET_MODEL_PATH" \
        --trust-remote-code \
        --skip-tokenizer-init \
        --tp-size "$SERVER_TP" \
        --context-length "$MAX_LENGTH" \
        --mem-fraction-static "${SERVER_MEM_FRACTION:-0.85}" \
        --chunked-prefill-size -1 \
        --disable-radix-cache \
        --enable-spec-capture \
        --spec-capture-method dflash \
        --spec-capture-aux-layer-ids $AUX_LAYER_IDS \
        --port "$SERVER_PORT" &
SERVER_PID=$!

until curl -sf "http://127.0.0.1:$SERVER_PORT/health" >/dev/null; do
    if ! kill -0 "$MASTER_PID" 2>/dev/null; then
        echo "Mooncake master died while SGLang was starting" >&2
        exit 1
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "SGLang server on :$SERVER_PORT died" >&2
        exit 1
    fi
    sleep 5
done

ARGS=(
    --target-model-path "$TARGET_MODEL_PATH"
    --target-model-backend hf
    --trust-remote-code
    --draft-config-path "$DRAFT_CONFIG_PATH"
    --embedding-key "$EMBEDDING_KEY"
    --lm-head-key "$LM_HEAD_KEY"
    --mask-token-id "$MASK_TOKEN_ID"
    --train-data-path "$SINGLE_DATA_PATH"
    --chat-template "$CHAT_TEMPLATE"
    --train-only-last-turn
    --max-length "$MAX_LENGTH"
    --batch-size 1
    --accumulation-steps 1
    --learning-rate "${LEARNING_RATE:-6e-4}"
    --warmup-ratio "${WARMUP_RATIO:-0.04}"
    --max-grad-norm 1.0
    --attention-backend "${ATTENTION_BACKEND:-flex_attention}"
    --block-size "$BLOCK_SIZE"
    --num-anchors "${NUM_ANCHORS:-256}"
    --loss-decay-gamma 7.0
    --num-epochs "$NUM_EPOCHS"
    --seed 42
    --save-interval "$DISAGG_MAX_STEPS"
    --lambda-base-start 1.0
    --lambda-base-decay-ratio 1.0
)
LAUNCHER=$SCRIPT_DIR/run_disagg_domino.py

DISAGG_ROLE=producer CUDA_VISIBLE_DEVICES="" \
    "$PYTHON" "$LAUNCHER" "${ARGS[@]}" \
        --output-dir "$PRODUCER_OUTPUT_DIR" &
PRODUCER_PID=$!

CUDA_VISIBLE_DEVICES=$CONSUMER_GPUS DISAGG_ROLE=consumer \
    DISAGG_DB=$DISAGG_DB DISAGG_INBOX_DIR=$DISAGG_INBOX_DIR \
    torchrun --rdzv-backend c10d --rdzv-endpoint localhost:29702 \
        --nnodes 1 --nproc_per_node "$TRAIN_DP" "$LAUNCHER" "${ARGS[@]}" \
        --output-dir "$CONSUMER_OUTPUT_DIR" \
        --report-to none 2>&1 | tee "$TRAIN_LOG"

wait "$PRODUCER_PID"
"$PYTHON" "$ROOT_DIR/scripts/check_domino_overfit.py" \
    --log-path "$TRAIN_LOG" \
    --checkpoint-root "$CONSUMER_OUTPUT_DIR" \
    --expected-step "$DISAGG_MAX_STEPS" \
    --max-loss "$MAX_LOSS" \
    --min-accuracy "$MIN_ACCURACY"

LATEST_CHECKPOINT="$CONSUMER_OUTPUT_DIR/$DISAGG_STORE_ID-step$DISAGG_MAX_STEPS"
if [[ ! -f "$LATEST_CHECKPOINT/training_state.pt" ]]; then
    echo "Expected final checkpoint is missing: $LATEST_CHECKPOINT/training_state.pt" >&2
    exit 1
fi
if [[ "${RUN_REAL_SERVING_GATE:-0}" == "1" ]]; then
    # Release the capture/training stack before loading target + draft serving.
    kill "$SERVER_PID" 2>/dev/null || true
    kill -- "-$MASTER_PID" 2>/dev/null || kill "$MASTER_PID" 2>/dev/null || true
    wait "$SERVER_PID" "$MASTER_PID" 2>/dev/null || true
    SERVER_PID=""
    MASTER_PID=""
    CHECKPOINT_PATH="$LATEST_CHECKPOINT" \
    PROMPT_ARTIFACT_PATH="$PROMPT_ARTIFACT_PATH" \
    MODEL_PROFILE="$MODEL_PROFILE" \
    TARGET_MODEL_PATH="$TARGET_MODEL_PATH" \
    DRAFT_CONFIG_PATH="$DRAFT_CONFIG_PATH" \
    EMBEDDING_KEY="$EMBEDDING_KEY" \
    SPECFORGE_PYTHON="$PYTHON" \
    SGLANG_PYTHON="${SERVING_PYTHON:-$PYTHON}" \
    SGLANG_ROOT="${SERVING_SGLANG_ROOT:-}" \
    SERVING_GPUS="$REAL_SERVING_GPUS" \
    SERVING_TP="$REAL_SERVING_TP" \
    WORK_DIR="$WORK_DIR/real_serving" \
        bash "$ROOT_DIR/examples/disagg/run_qwen3_8b_domino_real_serving_gate.sh"
else
    echo "Real serving is a separate strict gate. Run it with:"
    echo "MODEL_PROFILE=$MODEL_PROFILE CHECKPOINT_PATH=$LATEST_CHECKPOINT \\"
    echo "  PROMPT_ARTIFACT_PATH=$PROMPT_ARTIFACT_PATH \\"
    echo "  bash examples/disagg/run_qwen3_8b_domino_real_serving_gate.sh"
fi

echo "DOMINO-DISAGG-OVERFIT-PASSED profile=$MODEL_PROFILE: $WORK_DIR"
