#!/usr/bin/env bash
# Two physical nodes, one canonical SpecForge training entry:
#   rank 0: Mooncake + patched SGLang + CPU producer
#   rank 1: GPU consumer/trainer
#
# Launch the same command on both nodes (for example with `rcli exec --per-node`).
# The nodes must share DISAGG_RUN_ROOT; tensors travel through Mooncake while the
# shared directory carries only control state and logs.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# NODE_RANK is also consumed by the unified training launch plan. Capture the
# physical orchestration rank, then remove it from child environments because
# this wrapper intentionally starts a single-node trainer on physical rank 1.
ORCHESTRATION_NODE_RANK="${NODE_RANK:-${RCLI_NODE_RANK:-}}"
unset NODE_RANK
NUM_NODES="${NUM_NODES:-${RCLI_NUM_NODES:-}}"
HEAD_IP="${HEAD_IP:-${RCLI_HEAD_IP:-}}"
RUN_ID="${DISAGG_STORE_ID:-}"
RUN_ROOT="${DISAGG_RUN_ROOT:-}"
CONSUMER_STATE_DIR="${DISAGG_CONSUMER_STATE_DIR:-${LOCAL_SCRATCH:-/tmp}/specforge/$RUN_ID/consumer-state}"
CONFIG="${CONFIG:-$ROOT_DIR/examples/configs/qwen3-8b-dflash-disaggregated.yaml}"

SERVER_GPUS="${SERVER_GPUS:-0}"
SERVER_TP="${SERVER_TP:-1}"
SERVER_PORT="${SERVER_PORT:-30000}"
SERVER_MEM_FRACTION="${SERVER_MEM_FRACTION:-0.85}"
SERVER_DISABLE_CUDA_GRAPH="${SERVER_DISABLE_CUDA_GRAPH:-0}"
SERVER_DISABLE_OVERLAP_SCHEDULE="${SERVER_DISABLE_OVERLAP_SCHEDULE:-0}"
SERVER_SKIP_WARMUP="${SERVER_SKIP_WARMUP:-0}"
SERVER_CUDA_GRAPH_MAX_BS_DECODE="${SERVER_CUDA_GRAPH_MAX_BS_DECODE:-}"
SERVER_MAX_TOTAL_TOKENS="${SERVER_MAX_TOTAL_TOKENS:-}"
SERVER_MAX_PREFILL_TOKENS="${SERVER_MAX_PREFILL_TOKENS:-}"
SERVER_CHUNKED_PREFILL_SIZE="${SERVER_CHUNKED_PREFILL_SIZE:--1}"
CAPTURE_LAYER_IDS="${CAPTURE_LAYER_IDS:-1 9 17 25 33}"
TRAINER_GPUS="${TRAINER_GPUS:-0,1,2,3}"
TRAINER_NPROC="${TRAINER_NPROC:-4}"
TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-Qwen/Qwen3-8B}"

MOONCAKE_RPC_PORT="${MOONCAKE_RPC_PORT:-35551}"
MOONCAKE_HTTP_PORT="${MOONCAKE_HTTP_PORT:-35880}"
MOONCAKE_METRICS_PORT="${MOONCAKE_METRICS_PORT:-35903}"
MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-tcp}"
MOONCAKE_DEFAULT_KV_LEASE_TTL="${MOONCAKE_DEFAULT_KV_LEASE_TTL:-600000}"
START_TIMEOUT_S="${START_TIMEOUT_S:-1800}"
PEER_TIMEOUT_S="${PEER_TIMEOUT_S:-1800}"

# EXIT traps run after function-local variables leave scope. Keep child PIDs and
# statuses in script scope so ``set -u`` cannot mask their real results.
INFERENCE_MASTER_PID=""
INFERENCE_SERVER_PID=""
INFERENCE_PRODUCER_PID=""
INFERENCE_RESULT=1
TRAINING_CONSUMER_PID=""
TRAINING_RESULT=1

log() {
    printf '[qwen3-8b-dflash-2node][rank=%s] %s\n' \
        "${ORCHESTRATION_NODE_RANK:-?}" "$*"
}

fail() {
    log "ERROR: $*" >&2
    exit 1
}

write_status() {
    local destination="$1"
    local value="$2"
    local temporary="${destination}.tmp.$$"
    printf '%s\n' "$value" > "$temporary"
    mv -f "$temporary" "$destination"
}

read_status() {
    tr -d '[:space:]' < "$1"
}

wait_for_file() {
    local wanted="$1"
    local description="$2"
    local peer_status="${3:-}"
    local started
    started="$(date +%s)"
    while [[ ! -e "$wanted" ]]; do
        if [[ -n "$peer_status" && -e "$peer_status" ]]; then
            fail "$description aborted with peer status $(read_status "$peer_status")"
        fi
        if (( $(date +%s) - started >= PEER_TIMEOUT_S )); then
            fail "timed out waiting for $description: $wanted"
        fi
        sleep 2
    done
}

count_devices() {
    local devices="$1"
    awk -F, '{print NF}' <<< "$devices"
}

kill_group() {
    local pid="${1:-}"
    [[ -n "$pid" ]] || return 0
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 0.5
    done
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
}

local_ip() {
    local value
    value="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
    if [[ -z "$value" ]]; then
        value="$(hostname -i 2>/dev/null | awk '{print $1}' || true)"
    fi
    [[ -n "$value" ]] || fail "could not resolve a routable local IP"
    printf '%s\n' "$value"
}

print_command() {
    printf 'DRY-RUN:'
    printf ' %q' "$@"
    printf '\n'
}

validate_identity() {
    [[ "$NUM_NODES" == "2" ]] || fail "NUM_NODES/RCLI_NUM_NODES must be 2"
    [[ "$ORCHESTRATION_NODE_RANK" == "0" || \
        "$ORCHESTRATION_NODE_RANK" == "1" ]] || \
        fail "NODE_RANK/RCLI_NODE_RANK must be 0 or 1"
    [[ -n "$HEAD_IP" ]] || fail "HEAD_IP/RCLI_HEAD_IP is required"
    [[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || \
        fail "set a unique DISAGG_STORE_ID using letters, digits, '.', '_' or '-'"
    [[ -n "$RUN_ROOT" && "$RUN_ROOT" != "/" ]] || \
        fail "set a non-root shared DISAGG_RUN_ROOT"
    [[ -n "$CONSUMER_STATE_DIR" && "$CONSUMER_STATE_DIR" != "/" ]] || \
        fail "set a non-root node-local DISAGG_CONSUMER_STATE_DIR"
    [[ -f "$CONFIG" ]] || fail "config does not exist: $CONFIG"
    [[ "$SERVER_TP" =~ ^[1-9][0-9]*$ ]] || fail "SERVER_TP must be positive"
    [[ "$SERVER_DISABLE_CUDA_GRAPH" == "0" || \
        "$SERVER_DISABLE_CUDA_GRAPH" == "1" ]] || \
        fail "SERVER_DISABLE_CUDA_GRAPH must be 0 or 1"
    [[ "$SERVER_DISABLE_OVERLAP_SCHEDULE" == "0" || \
        "$SERVER_DISABLE_OVERLAP_SCHEDULE" == "1" ]] || \
        fail "SERVER_DISABLE_OVERLAP_SCHEDULE must be 0 or 1"
    [[ "$SERVER_SKIP_WARMUP" == "0" || "$SERVER_SKIP_WARMUP" == "1" ]] || \
        fail "SERVER_SKIP_WARMUP must be 0 or 1"
    [[ -z "$SERVER_CUDA_GRAPH_MAX_BS_DECODE" || \
        "$SERVER_CUDA_GRAPH_MAX_BS_DECODE" =~ ^[1-9][0-9]*$ ]] || \
        fail "SERVER_CUDA_GRAPH_MAX_BS_DECODE must be empty or positive"
    [[ -z "$SERVER_MAX_TOTAL_TOKENS" || \
        "$SERVER_MAX_TOTAL_TOKENS" =~ ^[1-9][0-9]*$ ]] || \
        fail "SERVER_MAX_TOTAL_TOKENS must be empty or positive"
    [[ -z "$SERVER_MAX_PREFILL_TOKENS" || \
        "$SERVER_MAX_PREFILL_TOKENS" =~ ^[1-9][0-9]*$ ]] || \
        fail "SERVER_MAX_PREFILL_TOKENS must be empty or positive"
    [[ "$SERVER_CHUNKED_PREFILL_SIZE" == "-1" || \
        "$SERVER_CHUNKED_PREFILL_SIZE" =~ ^[1-9][0-9]*$ ]] || \
        fail "SERVER_CHUNKED_PREFILL_SIZE must be -1 or positive"
    [[ "$TRAINER_NPROC" =~ ^[1-9][0-9]*$ ]] || \
        fail "TRAINER_NPROC must be positive"
    [[ "$(count_devices "$SERVER_GPUS")" == "$SERVER_TP" ]] || \
        fail "SERVER_GPUS must contain exactly SERVER_TP=$SERVER_TP devices"
    [[ "$(count_devices "$TRAINER_GPUS")" == "$TRAINER_NPROC" ]] || \
        fail "TRAINER_GPUS must contain exactly TRAINER_NPROC=$TRAINER_NPROC devices"
}

export_common_environment() {
    export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
    export FLASHINFER_DISABLE_VERSION_CHECK=1
    export SGLANG_USE_MESSAGE_QUEUE_BROADCASTER="${SGLANG_USE_MESSAGE_QUEUE_BROADCASTER:-1}"
    export MOONCAKE_MASTER_SERVER_ADDR="$HEAD_IP:$MOONCAKE_RPC_PORT"
    export MOONCAKE_METADATA_SERVER="http://$HEAD_IP:$MOONCAKE_HTTP_PORT/metadata"
    export MOONCAKE_PROTOCOL
    export DISAGG_CLIENT_SEGMENT_SIZE=0
    export DISAGG_SERVER_URLS="http://$HEAD_IP:$SERVER_PORT"
}

COMMON_OVERRIDES=(
    "model.target_model_path=$TARGET_MODEL_PATH"
    "run_id=$RUN_ID"
    "output_dir=$RUN_ROOT/output"
    "deployment.trainer.nnodes=1"
    "deployment.trainer.nproc_per_node=$TRAINER_NPROC"
    "deployment.disaggregated.control_dir=$RUN_ROOT/control"
    "deployment.disaggregated.consumer_state_dir=$CONSUMER_STATE_DIR"
    "deployment.disaggregated.store_id=$RUN_ID"
    "deployment.disaggregated.server_urls=[\"http://$HEAD_IP:$SERVER_PORT\"]"
    "deployment.disaggregated.mooncake_metadata_server=http://$HEAD_IP:$MOONCAKE_HTTP_PORT/metadata"
    "deployment.disaggregated.mooncake_master_server_addr=$HEAD_IP:$MOONCAKE_RPC_PORT"
    "deployment.disaggregated.mooncake_protocol=$MOONCAKE_PROTOCOL"
    "deployment.disaggregated.idle_timeout_s=$PEER_TIMEOUT_S"
    "deployment.disaggregated.peer_wait_timeout_s=$PEER_TIMEOUT_S"
)

run_inference_node() {
    local producer_result=1
    local -a cuda_graph_args=()
    local -a overlap_schedule_args=()
    local -a warmup_args=()
    local -a decode_graph_args=()
    local -a max_total_tokens_args=()
    local -a max_prefill_tokens_args=()
    if [[ "$SERVER_DISABLE_CUDA_GRAPH" == "1" ]]; then
        cuda_graph_args=(--disable-cuda-graph)
    fi
    if [[ "$SERVER_DISABLE_OVERLAP_SCHEDULE" == "1" ]]; then
        overlap_schedule_args=(--disable-overlap-schedule)
    fi
    if [[ "$SERVER_SKIP_WARMUP" == "1" ]]; then
        warmup_args=(--skip-server-warmup)
    fi
    if [[ -n "$SERVER_CUDA_GRAPH_MAX_BS_DECODE" ]]; then
        decode_graph_args=(
            --cuda-graph-max-bs-decode "$SERVER_CUDA_GRAPH_MAX_BS_DECODE"
        )
    fi
    if [[ -n "$SERVER_MAX_TOTAL_TOKENS" ]]; then
        max_total_tokens_args=(--max-total-tokens "$SERVER_MAX_TOTAL_TOKENS")
    fi
    if [[ -n "$SERVER_MAX_PREFILL_TOKENS" ]]; then
        max_prefill_tokens_args=(--max-prefill-tokens "$SERVER_MAX_PREFILL_TOKENS")
    fi

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        print_command mooncake_master --enable_http_metadata_server=true \
            --http_metadata_server_host=0.0.0.0 \
            --rpc_port="$MOONCAKE_RPC_PORT" \
            --http_metadata_server_port="$MOONCAKE_HTTP_PORT" \
            --metrics_port="$MOONCAKE_METRICS_PORT" \
            --default_kv_lease_ttl="$MOONCAKE_DEFAULT_KV_LEASE_TTL"
        print_command env "CUDA_VISIBLE_DEVICES=$SERVER_GPUS" \
            python -m sglang.launch_server --host 0.0.0.0 \
            --model-path "$TARGET_MODEL_PATH" --tp-size "$SERVER_TP" \
            --mem-fraction-static "$SERVER_MEM_FRACTION" \
            "${cuda_graph_args[@]}" \
            "${overlap_schedule_args[@]}" \
            "${warmup_args[@]}" \
            "${decode_graph_args[@]}" \
            "${max_total_tokens_args[@]}" \
            "${max_prefill_tokens_args[@]}" \
            --chunked-prefill-size "$SERVER_CHUNKED_PREFILL_SIZE" \
            --enable-spec-capture --spec-capture-method dflash \
            --spec-capture-aux-layer-ids $CAPTURE_LAYER_IDS \
            --port "$SERVER_PORT"
        print_command env CUDA_VISIBLE_DEVICES= specforge train -c "$CONFIG" \
            --role producer "${COMMON_OVERRIDES[@]}" "$@"
        INFERENCE_RESULT=0
        return
    fi

    mkdir -p "$(dirname "$RUN_ROOT")"
    mkdir "$RUN_ROOT" 2>/dev/null || \
        fail "run root already exists; choose a fresh DISAGG_STORE_ID/RUN_ROOT"

    cleanup() {
        kill_group "$INFERENCE_PRODUCER_PID"
        kill_group "$INFERENCE_SERVER_PID"
        kill_group "$INFERENCE_MASTER_PID"
        write_status "$RUN_ROOT/inference.done" "$INFERENCE_RESULT"
    }
    trap cleanup EXIT
    trap 'INFERENCE_RESULT=129; exit 129' HUP
    trap 'INFERENCE_RESULT=130; exit 130' INT
    trap 'INFERENCE_RESULT=143; exit 143' TERM

    command -v mooncake_master >/dev/null || fail "mooncake_master is not on PATH"
    command -v curl >/dev/null || fail "curl is not on PATH"
    "$ROOT_DIR/scripts/apply_sglang_spec_capture_patch.sh"
    export MOONCAKE_LOCAL_HOSTNAME="${INFERENCE_NODE_IP:-$HEAD_IP}"
    export MOONCAKE_GLOBAL_SEGMENT_SIZE="${MOONCAKE_GLOBAL_SEGMENT_SIZE:-$((32 << 30))}"
    export MOONCAKE_LOCAL_BUFFER_SIZE="${MOONCAKE_LOCAL_BUFFER_SIZE:-$((1 << 30))}"

    setsid mooncake_master \
        --enable_http_metadata_server=true \
        --http_metadata_server_host=0.0.0.0 \
        --rpc_port="$MOONCAKE_RPC_PORT" \
        --http_metadata_server_port="$MOONCAKE_HTTP_PORT" \
        --metrics_port="$MOONCAKE_METRICS_PORT" \
        --default_kv_lease_ttl="$MOONCAKE_DEFAULT_KV_LEASE_TTL" \
        > "$RUN_ROOT/mooncake.log" 2>&1 &
    INFERENCE_MASTER_PID="$!"

    local started
    started="$(date +%s)"
    while true; do
        if curl -sS --max-time 1 -o /dev/null \
            "$MOONCAKE_METADATA_SERVER?key=specforge-health-check" && \
            python -c \
                'import socket,sys; socket.create_connection((sys.argv[1], int(sys.argv[2])), 1).close()' \
                "$HEAD_IP" "$MOONCAKE_RPC_PORT"; then
            break
        fi
        kill -0 "$INFERENCE_MASTER_PID" 2>/dev/null || \
            fail "Mooncake exited; see $RUN_ROOT/mooncake.log"
        (( $(date +%s) - started < START_TIMEOUT_S )) || \
            fail "Mooncake readiness timed out"
        sleep 1
    done

    read -r -a capture_layers <<< "$CAPTURE_LAYER_IDS"
    setsid env CUDA_VISIBLE_DEVICES="$SERVER_GPUS" \
        python -m sglang.launch_server \
        --host 0.0.0.0 \
        --model-path "$TARGET_MODEL_PATH" \
        --trust-remote-code \
        --skip-tokenizer-init \
        --tp-size "$SERVER_TP" \
        --mem-fraction-static "$SERVER_MEM_FRACTION" \
        "${cuda_graph_args[@]}" \
        "${overlap_schedule_args[@]}" \
        "${warmup_args[@]}" \
        "${decode_graph_args[@]}" \
        "${max_total_tokens_args[@]}" \
        "${max_prefill_tokens_args[@]}" \
        --chunked-prefill-size "$SERVER_CHUNKED_PREFILL_SIZE" \
        --enable-spec-capture \
        --spec-capture-method dflash \
        --spec-capture-aux-layer-ids "${capture_layers[@]}" \
        --port "$SERVER_PORT" \
        > "$RUN_ROOT/sglang-server.log" 2>&1 &
    INFERENCE_SERVER_PID="$!"

    started="$(date +%s)"
    until curl -fsS "http://$HEAD_IP:$SERVER_PORT/health" >/dev/null; do
        kill -0 "$INFERENCE_SERVER_PID" 2>/dev/null || \
            fail "SGLang exited; see $RUN_ROOT/sglang-server.log"
        (( $(date +%s) - started < START_TIMEOUT_S )) || \
            fail "SGLang readiness timed out"
        sleep 5
    done
    touch "$RUN_ROOT/inference.ready"

    setsid env CUDA_VISIBLE_DEVICES= specforge train -c "$CONFIG" \
        --role producer "${COMMON_OVERRIDES[@]}" "$@" \
        > >(tee "$RUN_ROOT/producer.log") 2>&1 &
    INFERENCE_PRODUCER_PID="$!"
    set +e
    wait "$INFERENCE_PRODUCER_PID"
    producer_result="$?"
    set -e
    INFERENCE_PRODUCER_PID=""
    [[ "$producer_result" == "0" ]] || {
        INFERENCE_RESULT="$producer_result"
        fail "producer exited with status $producer_result"
    }

    wait_for_file "$RUN_ROOT/consumer.done" "consumer completion"
    local consumer_result
    consumer_result="$(read_status "$RUN_ROOT/consumer.done")"
    [[ "$consumer_result" == "0" ]] || {
        INFERENCE_RESULT="$consumer_result"
        fail "consumer exited with status $consumer_result"
    }
    INFERENCE_RESULT=0
}

run_training_node() {
    local peer_result=""

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        print_command env "CUDA_VISIBLE_DEVICES=$TRAINER_GPUS" \
            specforge train -c "$CONFIG" --role consumer \
            "${COMMON_OVERRIDES[@]}" "$@"
        return
    fi

    wait_for_file "$RUN_ROOT/inference.ready" "inference readiness" \
        "$RUN_ROOT/inference.done"
    export MOONCAKE_LOCAL_HOSTNAME="${TRAINER_NODE_IP:-$(local_ip)}"

    finish() {
        kill_group "$TRAINING_CONSUMER_PID"
        write_status "$RUN_ROOT/consumer.done" "$TRAINING_RESULT"
    }
    trap finish EXIT
    trap 'TRAINING_RESULT=129; exit 129' HUP
    trap 'TRAINING_RESULT=130; exit 130' INT
    trap 'TRAINING_RESULT=143; exit 143' TERM

    mkdir -p "$(dirname "$CONSUMER_STATE_DIR")"
    mkdir "$CONSUMER_STATE_DIR" 2>/dev/null || \
        fail "consumer state already exists; choose a fresh DISAGG_CONSUMER_STATE_DIR"

    setsid env CUDA_VISIBLE_DEVICES="$TRAINER_GPUS" \
        specforge train -c "$CONFIG" --role consumer \
        "${COMMON_OVERRIDES[@]}" "$@" \
        > >(tee "$RUN_ROOT/consumer.log") 2>&1 &
    TRAINING_CONSUMER_PID="$!"

    while kill -0 "$TRAINING_CONSUMER_PID" 2>/dev/null; do
        if [[ -e "$RUN_ROOT/inference.done" ]] && \
            [[ "$(read_status "$RUN_ROOT/inference.done")" != "0" ]]; then
            peer_result="$(read_status "$RUN_ROOT/inference.done")"
            kill_group "$TRAINING_CONSUMER_PID"
            break
        fi
        sleep 2
    done
    set +e
    wait "$TRAINING_CONSUMER_PID"
    local process_result="$?"
    set -e
    TRAINING_CONSUMER_PID=""
    TRAINING_RESULT="${peer_result:-$process_result}"
    return "$TRAINING_RESULT"
}

main() {
    validate_identity
    export_common_environment
    cd "$ROOT_DIR"
    case "$ORCHESTRATION_NODE_RANK" in
        0) run_inference_node "$@" ;;
        1) run_training_node "$@" ;;
    esac
}

main "$@"
