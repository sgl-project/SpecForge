#!/usr/bin/env bash
# Run one strict DFlash-family disaggregated overfit and optional serving gate.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")
# shellcheck source=scripts/gates/_e2e_common.sh
source "$SCRIPT_DIR/_e2e_common.sh"

usage() {
    cat <<'EOF'
Run a one-sample online-disaggregated overfit through the unified SpecForge CLI.

Required environment:
  CONFIG                 Typed online-disaggregated YAML config.
  TARGET_MODEL_PATH      Target model path or Hugging Face repository.
  DRAFT_CONFIG_PATH      DFlash-family draft JSON (Domino by default).
  SOURCE_DATA_PATH       Validated conversation JSONL used to select one sample.

Common overrides:
  WORK_DIR               Fresh output directory.
  RUN_ID                 Attempt/run id (default: disagg-overfit-<timestamp>).
  NPROC_PER_NODE         Consumer DP width (default: 1).
  CAPTURE_GPUS           Capture-server CUDA devices (default: 0).
  CONSUMER_GPUS          Trainer CUDA devices (default: 1).
  MAX_STEPS              Strict optimizer-step target (default: 400).
  RUN_SERVING_GATE       Chain the real DFLASH serving gate (default: true).
  GATE_DRY_RUN           Print the complete command plan without side effects.

The script starts and owns Mooncake, the patched SGLang capture server, the
canonical producer/consumer wrappers, metric verification, export, real serving,
and cleanup. Set START_MOONCAKE=false or START_CAPTURE_SERVER=false only when an
externally supervised service is already available at the configured address.
EOF
}

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    usage
    exit 0
fi
[[ $# -eq 0 ]] || gate_fail "this gate accepts configuration through environment variables"

gate_install_cleanup_traps
gate_is_true "${GATE_DRY_RUN:-0}" GATE_DRY_RUN || true
gate_is_true "${START_MOONCAKE:-true}" START_MOONCAKE || true
gate_is_true "${START_CAPTURE_SERVER:-true}" START_CAPTURE_SERVER || true
gate_is_true "${RUN_SERVING_GATE:-true}" RUN_SERVING_GATE || true

CONFIG=${CONFIG:-}
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-}
DRAFT_CONFIG_PATH=${DRAFT_CONFIG_PATH:-}
SOURCE_DATA_PATH=${SOURCE_DATA_PATH:-}
PYTHON=${PYTHON:-python}
SGLANG_PYTHON=${SGLANG_PYTHON:-$PYTHON}
MOONCAKE_BIN=${MOONCAKE_BIN:-mooncake_master}
ONLINE_LAUNCHER=$ROOT_DIR/examples/disagg/run_online.sh
SERVING_GATE=$SCRIPT_DIR/run_dflash_serving_gate.sh

gate_require_value "$CONFIG" CONFIG
gate_require_file "$CONFIG" CONFIG
gate_require_value "$TARGET_MODEL_PATH" TARGET_MODEL_PATH
gate_require_value "$DRAFT_CONFIG_PATH" DRAFT_CONFIG_PATH
gate_require_file "$DRAFT_CONFIG_PATH" DRAFT_CONFIG_PATH
gate_require_value "$SOURCE_DATA_PATH" SOURCE_DATA_PATH
gate_require_file "$SOURCE_DATA_PATH" SOURCE_DATA_PATH
gate_require_file "$ONLINE_LAUNCHER" online_launcher
gate_require_file "$SERVING_GATE" serving_gate
gate_require_command "$PYTHON" PYTHON

RUN_TAG=${RUN_TAG:-$(date +%Y%m%dT%H%M%S)}
RUN_ID=${RUN_ID:-disagg-overfit-$RUN_TAG}
WORK_DIR=${WORK_DIR:-$ROOT_DIR/outputs/$RUN_ID}
PRODUCER_OUTPUT_DIR=$WORK_DIR/producer
CONSUMER_OUTPUT_DIR=$WORK_DIR/consumer
SINGLE_DATA_PATH=$WORK_DIR/single_sample.jsonl
PROMPT_ARTIFACT_PATH=$WORK_DIR/prompt.json
TRAIN_LOG=$WORK_DIR/train.log
PRODUCER_LOG=$WORK_DIR/producer.log
MOONCAKE_LOG=$WORK_DIR/mooncake.log
CAPTURE_LOG=$WORK_DIR/capture-server.log

[[ ! -e "$WORK_DIR" ]] || gate_fail "WORK_DIR must be fresh: $WORK_DIR"

NPROC_PER_NODE=${NPROC_PER_NODE:-1}
NNODES=${NNODES:-1}
MAX_STEPS=${MAX_STEPS:-400}
MAX_LENGTH=${MAX_LENGTH:-512}
CAPTURE_TP=${CAPTURE_TP:-1}
CAPTURE_PORT=${CAPTURE_PORT:-30000}
MOONCAKE_RPC_PORT=${MOONCAKE_RPC_PORT:-35551}
MOONCAKE_HTTP_PORT=${MOONCAKE_HTTP_PORT:-35880}
MOONCAKE_METRICS_PORT=${MOONCAKE_METRICS_PORT:-35903}
gate_require_positive_integer "$NPROC_PER_NODE" NPROC_PER_NODE
gate_require_positive_integer "$NNODES" NNODES
[[ "$NNODES" == 1 ]] || {
    gate_fail "the one-command gate owns one local consumer node; set NNODES=1"
}
gate_require_positive_integer "$MAX_STEPS" MAX_STEPS
gate_require_positive_integer "$MAX_LENGTH" MAX_LENGTH
gate_require_positive_integer "$CAPTURE_TP" CAPTURE_TP
for port_name in CAPTURE_PORT MOONCAKE_RPC_PORT MOONCAKE_HTTP_PORT MOONCAKE_METRICS_PORT; do
    port_value=${!port_name}
    gate_require_positive_integer "$port_value" "$port_name"
    ((port_value <= 65535)) || gate_fail "$port_name must be <= 65535"
done

EXPECTED_PROJECTOR_TYPE=${EXPECTED_PROJECTOR_TYPE:-domino}
draft_metadata=$(
    "$PYTHON" -c '
import json
import sys

path, expected = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    config = json.load(handle)
method = config.get("dflash_config") or {}
projector = method.get("projector_type", "dflash")
if expected and projector != expected:
    raise SystemExit(
        f"draft config projector_type={projector!r}, expected {expected!r}"
    )
block_size = config.get("block_size")
layers = method.get("target_layer_ids")
mask_token_id = method.get("mask_token_id")
if not isinstance(block_size, int) or block_size < 1:
    raise SystemExit("draft config must define a positive block_size")
if not isinstance(layers, list) or not layers or not all(isinstance(x, int) for x in layers):
    raise SystemExit("draft config must define nonempty integer target_layer_ids")
if not isinstance(mask_token_id, int):
    raise SystemExit("draft config must define integer mask_token_id")
print(f"{block_size}|{mask_token_id}|{projector}|" + " ".join(map(str, layers)))
' "$DRAFT_CONFIG_PATH" "$EXPECTED_PROJECTOR_TYPE"
)
IFS='|' read -r BLOCK_SIZE MASK_TOKEN_ID PROJECTOR_TYPE DRAFT_CAPTURE_LAYER_IDS <<EOF
$draft_metadata
EOF

CAPTURE_LAYER_IDS=${CAPTURE_LAYER_IDS:-$DRAFT_CAPTURE_LAYER_IDS}
CAPTURE_LAYER_IDS=${CAPTURE_LAYER_IDS//,/ }
read -r -a CAPTURE_LAYER_ARGS <<EOF
$CAPTURE_LAYER_IDS
EOF
[[ ${#CAPTURE_LAYER_ARGS[@]} -gt 0 ]] || gate_fail "CAPTURE_LAYER_IDS is empty"

MIN_LOSS_TOKENS=${MIN_LOSS_TOKENS:-$((2 * BLOCK_SIZE))}
gate_require_positive_integer "$MIN_LOSS_TOKENS" MIN_LOSS_TOKENS
PROMPT_EPOCHS=$((MAX_STEPS * NPROC_PER_NODE))

IN_FLIGHT_HIGH_WATERMARK=${IN_FLIGHT_HIGH_WATERMARK:-}
if [[ -z "$IN_FLIGHT_HIGH_WATERMARK" ]]; then
    IN_FLIGHT_HIGH_WATERMARK=$((NPROC_PER_NODE * 4))
    if ((IN_FLIGHT_HIGH_WATERMARK < 64)); then
        IN_FLIGHT_HIGH_WATERMARK=64
    fi
fi
IN_FLIGHT_LOW_WATERMARK=${IN_FLIGHT_LOW_WATERMARK:-$((IN_FLIGHT_HIGH_WATERMARK / 2))}
gate_require_positive_integer "$IN_FLIGHT_HIGH_WATERMARK" IN_FLIGHT_HIGH_WATERMARK
gate_require_nonnegative_integer "$IN_FLIGHT_LOW_WATERMARK" IN_FLIGHT_LOW_WATERMARK
((IN_FLIGHT_HIGH_WATERMARK >= NPROC_PER_NODE)) || {
    gate_fail "IN_FLIGHT_HIGH_WATERMARK must cover the consumer quantum $NPROC_PER_NODE"
}
((IN_FLIGHT_LOW_WATERMARK <= IN_FLIGHT_HIGH_WATERMARK)) || {
    gate_fail "IN_FLIGHT_LOW_WATERMARK must not exceed the high watermark"
}

CHAT_TEMPLATE=${CHAT_TEMPLATE:-qwen}
REASONING_POLICY=${REASONING_POLICY:-allow}
ENABLE_THINKING=${ENABLE_THINKING:-false}
EMBEDDING_KEY=${EMBEDDING_KEY:-model.embed_tokens.weight}
MAX_LOSS=${MAX_LOSS:-0.0001}
MIN_ACCURACY=${MIN_ACCURACY:-1.0}
CAPTURE_METHOD=${CAPTURE_METHOD:-dflash}
CAPTURE_GPUS=${CAPTURE_GPUS:-0}
CONSUMER_GPUS=${CONSUMER_GPUS:-1}
CAPTURE_HOST=${CAPTURE_HOST:-127.0.0.1}
CAPTURE_MEM_FRACTION=${CAPTURE_MEM_FRACTION:-0.85}
CAPTURE_ATTENTION_BACKEND=${CAPTURE_ATTENTION_BACKEND:-}
CAPTURE_SERVER_EXTRA_ARGS=${CAPTURE_SERVER_EXTRA_ARGS:-}
gate_is_true "$ENABLE_THINKING" ENABLE_THINKING || true

MOONCAKE_HOST=${MOONCAKE_HOST:-127.0.0.1}
export MOONCAKE_LOCAL_HOSTNAME=${MOONCAKE_LOCAL_HOSTNAME:-$MOONCAKE_HOST}
export MOONCAKE_MASTER_SERVER_ADDR=${MOONCAKE_MASTER_SERVER_ADDR:-$MOONCAKE_HOST:$MOONCAKE_RPC_PORT}
export MOONCAKE_METADATA_SERVER=${MOONCAKE_METADATA_SERVER:-http://$MOONCAKE_HOST:$MOONCAKE_HTTP_PORT/metadata}
export MOONCAKE_PROTOCOL=${MOONCAKE_PROTOCOL:-tcp}
export MOONCAKE_GLOBAL_SEGMENT_SIZE=${MOONCAKE_GLOBAL_SEGMENT_SIZE:-$((32 << 30))}
export DISAGG_CLIENT_SEGMENT_SIZE=${DISAGG_CLIENT_SEGMENT_SIZE:-0}
export DISAGG_CLIENT_BUFFER_SIZE=${DISAGG_CLIENT_BUFFER_SIZE:-$((256 << 20))}
[[ "$DISAGG_CLIENT_SEGMENT_SIZE" == 0 ]] || {
    gate_fail "DISAGG_CLIENT_SEGMENT_SIZE must be 0 for server-owned captures"
}

export DISAGG_STORE_ID=${DISAGG_STORE_ID:-$RUN_ID}
export DISAGG_SERVER_URL=${DISAGG_SERVER_URL:-http://$CAPTURE_HOST:$CAPTURE_PORT}
export DISAGG_REF_CHANNEL=${DISAGG_REF_CHANNEL:-$WORK_DIR/refs.jsonl}
export DISAGG_DB=${DISAGG_DB:-$WORK_DIR/consumer.sqlite}
export DISAGG_INBOX_DIR=${DISAGG_INBOX_DIR:-$WORK_DIR/inboxes}
export DISAGG_IDLE_TIMEOUT=${DISAGG_IDLE_TIMEOUT:-600}

SELECT_ARGS=(
    "$PYTHON" "$SCRIPT_DIR/select_overfit_sample.py"
    --input-data-path "$SOURCE_DATA_PATH"
    --output-data-path "$SINGLE_DATA_PATH"
    --prompt-output-path "$PROMPT_ARTIFACT_PATH"
    --model-path "$TARGET_MODEL_PATH"
    --chat-template "$CHAT_TEMPLATE"
    --max-length "$MAX_LENGTH"
    --min-loss-tokens "$MIN_LOSS_TOKENS"
    --reasoning-policy "$REASONING_POLICY"
    --require-untruncated
)
if gate_is_true "$ENABLE_THINKING" ENABLE_THINKING; then
    SELECT_ARGS+=(--enable-thinking)
fi

COMMON_OVERRIDES=(
    "model.target_model_path=$TARGET_MODEL_PATH"
    "model.draft_model_config=$DRAFT_CONFIG_PATH"
    model.target_backend=sglang
    "model.embedding_key=$EMBEDDING_KEY"
    "model.mask_token_id=$MASK_TOKEN_ID"
    model.shard_target_output=false
    "data.train_data_path=$SINGLE_DATA_PATH"
    'data.prompts_path=""'
    'data.hidden_states_path=""'
    'data.eval_data_path=""'
    'data.eval_hidden_states_path=""'
    data.max_prompts=1
    "data.max_length=$MAX_LENGTH"
    "data.chat_template=$CHAT_TEMPLATE"
    "training.num_epochs=$PROMPT_EPOCHS"
    "training.max_steps=$MAX_STEPS"
    "training.total_steps=$MAX_STEPS"
    training.batch_size=1
    training.accumulation_steps=1
    training.eval_interval=0
    "training.save_interval=$MAX_STEPS"
    training.log_interval=1
    training.deployment_mode=disaggregated
    training.tp_size=1
    training.sp_ulysses_size=1
    training.sp_ring_size=1
    tracking.report_to=none
    "runtime.in_flight_high_watermark=$IN_FLIGHT_HIGH_WATERMARK"
    "runtime.in_flight_low_watermark=$IN_FLIGHT_LOW_WATERMARK"
    runtime.producer_lease=1
    "run_id=$RUN_ID"
)

CAPTURE_COMMAND=(
    env "CUDA_VISIBLE_DEVICES=$CAPTURE_GPUS"
    "$SGLANG_PYTHON" -m sglang.launch_server
    --model-path "$TARGET_MODEL_PATH"
    --trust-remote-code
    --skip-tokenizer-init
    --tp-size "$CAPTURE_TP"
    --context-length "$MAX_LENGTH"
    --mem-fraction-static "$CAPTURE_MEM_FRACTION"
    --chunked-prefill-size -1
    --disable-radix-cache
    --enable-spec-capture
    --spec-capture-method "$CAPTURE_METHOD"
    --spec-capture-aux-layer-ids "${CAPTURE_LAYER_ARGS[@]}"
    --host "$CAPTURE_HOST"
    --port "$CAPTURE_PORT"
)
if [[ -n "$CAPTURE_ATTENTION_BACKEND" ]]; then
    CAPTURE_COMMAND+=(--attention-backend "$CAPTURE_ATTENTION_BACKEND")
fi
if [[ -n "$CAPTURE_SERVER_EXTRA_ARGS" ]]; then
    read -r -a CAPTURE_EXTRA_ARGS <<EOF
$CAPTURE_SERVER_EXTRA_ARGS
EOF
    CAPTURE_COMMAND+=("${CAPTURE_EXTRA_ARGS[@]}")
fi

if ! gate_dry_run; then
    gate_require_command "$SGLANG_PYTHON" SGLANG_PYTHON
    gate_require_command curl curl
    if gate_is_true "${START_MOONCAKE:-true}" START_MOONCAKE; then
        gate_require_command "$MOONCAKE_BIN" MOONCAKE_BIN
        gate_require_tcp_port_free \
            "$PYTHON" "$MOONCAKE_HOST" "$MOONCAKE_RPC_PORT" MOONCAKE_RPC_PORT
        gate_require_tcp_port_free \
            "$PYTHON" "$MOONCAKE_HOST" "$MOONCAKE_HTTP_PORT" MOONCAKE_HTTP_PORT
        gate_require_tcp_port_free \
            "$PYTHON" "$MOONCAKE_HOST" "$MOONCAKE_METRICS_PORT" MOONCAKE_METRICS_PORT
    fi
    if gate_is_true "${START_CAPTURE_SERVER:-true}" START_CAPTURE_SERVER; then
        gate_require_tcp_port_free \
            "$PYTHON" "$CAPTURE_HOST" "$CAPTURE_PORT" CAPTURE_PORT
    fi
    mkdir -p "$PRODUCER_OUTPUT_DIR" "$CONSUMER_OUTPUT_DIR" "$DISAGG_INBOX_DIR"
    : >"$DISAGG_REF_CHANNEL"
fi

gate_run "${SELECT_ARGS[@]}"

MOONCAKE_PID=""
if gate_is_true "${START_MOONCAKE:-true}" START_MOONCAKE; then
    gate_start_service mooncake "$MOONCAKE_LOG" \
        "$MOONCAKE_BIN" \
        --enable_http_metadata_server=true \
        --rpc_port="$MOONCAKE_RPC_PORT" \
        --http_metadata_server_port="$MOONCAKE_HTTP_PORT" \
        --metrics_port="$MOONCAKE_METRICS_PORT"
    MOONCAKE_PID=$GATE_LAST_PID
fi
gate_wait_http "$MOONCAKE_PID" mooncake \
    "$MOONCAKE_METADATA_SERVER?key=specforge-health-check" "$MOONCAKE_LOG" true

CAPTURE_PID=""
if gate_is_true "${START_CAPTURE_SERVER:-true}" START_CAPTURE_SERVER; then
    gate_start_service capture-server "$CAPTURE_LOG" "${CAPTURE_COMMAND[@]}"
    CAPTURE_PID=$GATE_LAST_PID
fi
gate_wait_http "$CAPTURE_PID" capture-server \
    "$DISAGG_SERVER_URL/health" "$CAPTURE_LOG" false

gate_start_service producer "$PRODUCER_LOG" \
    env CUDA_VISIBLE_DEVICES= \
    "$ONLINE_LAUNCHER" producer \
    "${COMMON_OVERRIDES[@]}" \
    "output_dir=$PRODUCER_OUTPUT_DIR"
PRODUCER_PID=$GATE_LAST_PID

gate_run_with_tee "$TRAIN_LOG" \
    env "CUDA_VISIBLE_DEVICES=$CONSUMER_GPUS" \
    "NPROC_PER_NODE=$NPROC_PER_NODE" NNODES=1 \
    "DISAGG_DB=$DISAGG_DB" "DISAGG_INBOX_DIR=$DISAGG_INBOX_DIR" \
    "$ONLINE_LAUNCHER" consumer \
    "${COMMON_OVERRIDES[@]}" \
    "output_dir=$CONSUMER_OUTPUT_DIR"
gate_wait_pid "$PRODUCER_PID" producer "$PRODUCER_LOG"

gate_run "$PYTHON" "$SCRIPT_DIR/check_overfit_metrics.py" \
    --log-path "$TRAIN_LOG" \
    --checkpoint-root "$CONSUMER_OUTPUT_DIR" \
    --expected-step "$MAX_STEPS" \
    --max-loss "$MAX_LOSS" \
    --min-accuracy "$MIN_ACCURACY"

CHECKPOINT_PATH=$CONSUMER_OUTPUT_DIR/$RUN_ID-step$MAX_STEPS
if ! gate_dry_run; then
    gate_require_file "$CHECKPOINT_PATH/training_state.pt" final_checkpoint
fi

# Release the capture stack before loading the target and draft for serving.
gate_stop_services

if gate_is_true "${RUN_SERVING_GATE:-true}" RUN_SERVING_GATE; then
    gate_run env \
        "GATE_DRY_RUN=${GATE_DRY_RUN:-0}" \
        "CHECKPOINT_PATH=$CHECKPOINT_PATH" \
        "TARGET_MODEL_PATH=$TARGET_MODEL_PATH" \
        "DRAFT_CONFIG_PATH=$DRAFT_CONFIG_PATH" \
        "PROMPT_ARTIFACT_PATH=$PROMPT_ARTIFACT_PATH" \
        "EMBEDDING_KEY=$EMBEDDING_KEY" \
        "BLOCK_SIZE=$BLOCK_SIZE" \
        "SERVING_GPUS=${SERVING_GPUS:-$CAPTURE_GPUS}" \
        "SERVING_TP=${SERVING_TP:-$CAPTURE_TP}" \
        "SERVING_PORT=${SERVING_PORT:-30001}" \
        "SPECFORGE_BIN=${SPECFORGE_BIN:-specforge}" \
        "PYTHON=$PYTHON" \
        "SGLANG_PYTHON=$SGLANG_PYTHON" \
        "WORK_DIR=$WORK_DIR/serving" \
        bash "$SERVING_GATE"
fi

if gate_dry_run; then
    printf 'DISAGG-OVERFIT-E2E-PLAN: %s\n' "$WORK_DIR"
else
    printf 'DISAGG-OVERFIT-E2E-PASSED: %s\n' "$WORK_DIR"
fi
