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

NODE_RANK="${NODE_RANK:-${RCLI_NODE_RANK:-}}"
NUM_NODES="${NUM_NODES:-${RCLI_NUM_NODES:-}}"
HEAD_IP="${HEAD_IP:-${RCLI_HEAD_IP:-}}"
RUN_ID="${DISAGG_STORE_ID:-}"
RUN_ROOT="${DISAGG_RUN_ROOT:-}"
CONFIG="${CONFIG:-$ROOT_DIR/examples/configs/qwen3-8b-dflash-disaggregated.yaml}"

SERVER_GPUS="${SERVER_GPUS:-0}"
SERVER_TP="${SERVER_TP:-1}"
SERVER_PORT="${SERVER_PORT:-30000}"
SERVER_MEM_FRACTION="${SERVER_MEM_FRACTION:-0.85}"
CAPTURE_LAYER_IDS="${CAPTURE_LAYER_IDS:-1 9 17 25 33}"
TRAINER_GPUS="${TRAINER_GPUS:-0,1,2,3}"
TRAINER_NPROC="${TRAINER_NPROC:-4}"
TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-Qwen/Qwen3-8B}"

MOONCAKE_RPC_PORT="${MOONCAKE_RPC_PORT:-35551}"
MOONCAKE_HTTP_PORT="${MOONCAKE_HTTP_PORT:-35880}"
MOONCAKE_METRICS_PORT="${MOONCAKE_METRICS_PORT:-35903}"
MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-tcp}"
START_TIMEOUT_S="${START_TIMEOUT_S:-1800}"
PEER_TIMEOUT_S="${PEER_TIMEOUT_S:-1800}"

log() {
    printf '[qwen3-8b-dflash-2node][rank=%s] %s\n' "${NODE_RANK:-?}" "$*"
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
    [[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] || \
        fail "NODE_RANK/RCLI_NODE_RANK must be 0 or 1"
    [[ -n "$HEAD_IP" ]] || fail "HEAD_IP/RCLI_HEAD_IP is required"
    [[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || \
        fail "set a unique DISAGG_STORE_ID using letters, digits, '.', '_' or '-'"
    [[ -n "$RUN_ROOT" && "$RUN_ROOT" != "/" ]] || \
        fail "set a non-root shared DISAGG_RUN_ROOT"
    [[ -f "$CONFIG" ]] || fail "config does not exist: $CONFIG"
    [[ "$SERVER_TP" =~ ^[1-9][0-9]*$ ]] || fail "SERVER_TP must be positive"
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
    "deployment.disaggregated.store_id=$RUN_ID"
    "deployment.disaggregated.server_urls=[\"http://$HEAD_IP:$SERVER_PORT\"]"
    "deployment.disaggregated.mooncake_metadata_server=http://$HEAD_IP:$MOONCAKE_HTTP_PORT/metadata"
    "deployment.disaggregated.mooncake_master_server_addr=$HEAD_IP:$MOONCAKE_RPC_PORT"
    "deployment.disaggregated.mooncake_protocol=$MOONCAKE_PROTOCOL"
    "deployment.disaggregated.idle_timeout_s=$PEER_TIMEOUT_S"
    "deployment.disaggregated.peer_wait_timeout_s=$PEER_TIMEOUT_S"
)

run_inference_node() {
    local master_pid=""
    local server_pid=""
    local producer_pid=""
    local result=1
    local producer_result=1

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        print_command mooncake_master --enable_http_metadata_server=true \
            --http_metadata_server_host=0.0.0.0 \
            --rpc_port="$MOONCAKE_RPC_PORT" \
            --http_metadata_server_port="$MOONCAKE_HTTP_PORT" \
            --metrics_port="$MOONCAKE_METRICS_PORT"
        print_command env "CUDA_VISIBLE_DEVICES=$SERVER_GPUS" \
            python -m sglang.launch_server --host 0.0.0.0 \
            --model-path "$TARGET_MODEL_PATH" --tp-size "$SERVER_TP" \
            --enable-spec-capture --spec-capture-method dflash \
            --spec-capture-aux-layer-ids $CAPTURE_LAYER_IDS \
            --port "$SERVER_PORT"
        print_command env CUDA_VISIBLE_DEVICES= specforge train -c "$CONFIG" \
            --role producer "${COMMON_OVERRIDES[@]}" "$@"
        result=0
        return
    fi

    mkdir -p "$(dirname "$RUN_ROOT")"
    mkdir "$RUN_ROOT" 2>/dev/null || \
        fail "run root already exists; choose a fresh DISAGG_STORE_ID/RUN_ROOT"

    cleanup() {
        kill_group "$producer_pid"
        kill_group "$server_pid"
        kill_group "$master_pid"
        write_status "$RUN_ROOT/inference.done" "$result"
    }
    trap cleanup EXIT
    trap 'result=129; exit 129' HUP
    trap 'result=130; exit 130' INT
    trap 'result=143; exit 143' TERM

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
        > "$RUN_ROOT/mooncake.log" 2>&1 &
    master_pid="$!"

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
        kill -0 "$master_pid" 2>/dev/null || \
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
        --chunked-prefill-size -1 \
        --disable-radix-cache \
        --enable-spec-capture \
        --spec-capture-method dflash \
        --spec-capture-aux-layer-ids "${capture_layers[@]}" \
        --port "$SERVER_PORT" \
        > "$RUN_ROOT/sglang-server.log" 2>&1 &
    server_pid="$!"

    started="$(date +%s)"
    until curl -fsS "http://$HEAD_IP:$SERVER_PORT/health" >/dev/null; do
        kill -0 "$server_pid" 2>/dev/null || \
            fail "SGLang exited; see $RUN_ROOT/sglang-server.log"
        (( $(date +%s) - started < START_TIMEOUT_S )) || \
            fail "SGLang readiness timed out"
        sleep 5
    done
    touch "$RUN_ROOT/inference.ready"

    setsid env CUDA_VISIBLE_DEVICES= specforge train -c "$CONFIG" \
        --role producer "${COMMON_OVERRIDES[@]}" "$@" \
        > >(tee "$RUN_ROOT/producer.log") 2>&1 &
    producer_pid="$!"
    set +e
    wait "$producer_pid"
    producer_result="$?"
    set -e
    producer_pid=""
    [[ "$producer_result" == "0" ]] || {
        result="$producer_result"
        fail "producer exited with status $producer_result"
    }

    wait_for_file "$RUN_ROOT/consumer.done" "consumer completion"
    local consumer_result
    consumer_result="$(read_status "$RUN_ROOT/consumer.done")"
    [[ "$consumer_result" == "0" ]] || {
        result="$consumer_result"
        fail "consumer exited with status $consumer_result"
    }
    result=0
}

run_training_node() {
    local consumer_pid=""
    local result=1
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
        kill_group "$consumer_pid"
        write_status "$RUN_ROOT/consumer.done" "$result"
    }
    trap finish EXIT
    trap 'result=129; exit 129' HUP
    trap 'result=130; exit 130' INT
    trap 'result=143; exit 143' TERM

    setsid env CUDA_VISIBLE_DEVICES="$TRAINER_GPUS" \
        specforge train -c "$CONFIG" --role consumer \
        "${COMMON_OVERRIDES[@]}" "$@" \
        > >(tee "$RUN_ROOT/consumer.log") 2>&1 &
    consumer_pid="$!"

    while kill -0 "$consumer_pid" 2>/dev/null; do
        if [[ -e "$RUN_ROOT/inference.done" ]] && \
            [[ "$(read_status "$RUN_ROOT/inference.done")" != "0" ]]; then
            peer_result="$(read_status "$RUN_ROOT/inference.done")"
            kill_group "$consumer_pid"
            break
        fi
        sleep 2
    done
    set +e
    wait "$consumer_pid"
    local process_result="$?"
    set -e
    consumer_pid=""
    result="${peer_result:-$process_result}"
    return "$result"
}

main() {
    validate_identity
    export_common_environment
    cd "$ROOT_DIR"
    case "$NODE_RANK" in
        0) run_inference_node "$@" ;;
        1) run_training_node "$@" ;;
    esac
}

main "$@"
