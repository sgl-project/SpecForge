#!/usr/bin/env bash
# Qwen3-8B DFlash, ONLINE disaggregated across two physical nodes:
#   node rank 0: Mooncake master + patched SGLang server + CPU producer
#   node rank 1: DP consumer/trainer
#
# Both nodes must see DISAGG_RUN_ROOT through a shared filesystem. Feature
# tensors travel through Mooncake; the shared path only carries refs, logs, and
# lifecycle markers. RCLI_NODE_RANK/RCLI_NUM_NODES/RCLI_HEAD_IP are used when
# available; set NODE_RANK/NUM_NODES/HEAD_IP explicitly on other launchers.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

readonly NODE_RANK="${NODE_RANK:-${RCLI_NODE_RANK:-}}"
readonly NUM_NODES="${NUM_NODES:-${RCLI_NUM_NODES:-}}"
readonly HEAD_IP="${HEAD_IP:-${RCLI_HEAD_IP:-}}"
readonly RUN_ID="${DISAGG_STORE_ID:?Set DISAGG_STORE_ID to one unique value shared by both nodes}"
readonly RUN_ROOT="${DISAGG_RUN_ROOT:-${ROOT_DIR}/outputs/${RUN_ID}}"
readonly REF_CHANNEL="${DISAGG_REF_CHANNEL:-${RUN_ROOT}/refs.jsonl}"

readonly TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-Qwen/Qwen3-8B}"
readonly DRAFT_CONFIG_PATH="${DRAFT_CONFIG_PATH:-${ROOT_DIR}/configs/qwen3-8b-dflash.json}"
readonly TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${ROOT_DIR}/cache/dataset/perfectblend_train.jsonl}"
readonly CHAT_TEMPLATE="${CHAT_TEMPLATE:-qwen}"

readonly TRAIN_DP="${TRAIN_DP:-4}"
readonly BATCH_SIZE="${BATCH_SIZE:-2}"
readonly ACCUM="${ACCUM:-1}"
readonly NUM_EPOCHS="${NUM_EPOCHS:-10}"
readonly SAVE_INTERVAL="${SAVE_INTERVAL:-800}"
readonly NUM_ANCHORS="${NUM_ANCHORS:-512}"
readonly REPORT_TO="${REPORT_TO:-none}"
readonly WANDB_PROJECT="${WANDB_PROJECT:-qwen3-8b-dflash-disagg}"

readonly SERVER_TP="${SERVER_TP:-1}"
readonly SERVER_PORT="${SERVER_PORT:-30000}"
readonly SERVER_MEM_FRACTION="${SERVER_MEM_FRACTION:-0.85}"
readonly AUX_LAYER_IDS="${AUX_LAYER_IDS:-1 9 17 25 33}"
readonly MOONCAKE_RPC_PORT="${MOONCAKE_RPC_PORT:-35551}"
readonly MOONCAKE_HTTP_PORT="${MOONCAKE_HTTP_PORT:-35880}"
readonly MOONCAKE_METRICS_PORT="${MOONCAKE_METRICS_PORT:-35903}"
readonly SERVICE_START_TIMEOUT="${SERVICE_START_TIMEOUT:-1800}"
readonly CONSUMER_START_TIMEOUT="${CONSUMER_START_TIMEOUT:-1800}"

TRAINING_CUDA_DEVICES=""
SERVER_CUDA_DEVICES=""
AUX_LAYER_ID_ARGS=()

log() {
    printf '[dflash-2node][rank=%s][%s] %s\n' \
        "${NODE_RANK:-unknown}" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

fail() {
    log "ERROR: $*"
    exit 1
}

write_status() {
    local path="$1"
    local status="$2"
    local temporary_path="${path}.tmp.$$"

    printf '%s\n' "$status" > "$temporary_path"
    mv -f "$temporary_path" "$path"
}

read_status() {
    local path="$1"
    tr -d '[:space:]' < "$path"
}

kill_process_group() {
    local pid="${1:-}"
    [[ -n "$pid" ]] || return 0

    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    local attempt
    for attempt in $(seq 1 10); do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 1
    done
    kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
}

resolve_local_ip() {
    local local_ip
    local_ip="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
    if [[ -z "$local_ip" ]]; then
        local_ip="$(hostname -i 2>/dev/null | awk '{print $1}' || true)"
    fi
    [[ -n "$local_ip" ]] || fail "could not resolve this node's routable IP"
    printf '%s\n' "$local_ip"
}

resolve_gpu_devices() {
    local runtime_allocation="${NVIDIA_VISIBLE_DEVICES:-}"
    local runtime_has_no_gpus=0
    if [[ "$runtime_allocation" == "none" || "$runtime_allocation" == "void" ]]; then
        runtime_has_no_gpus=1
    fi

    if [[ -n "${CONSUMER_GPUS:-}" ]]; then
        TRAINING_CUDA_DEVICES="$CONSUMER_GPUS"
    elif [[ -n "${TRAIN_GPU_IDS:-}" ]]; then
        TRAINING_CUDA_DEVICES="$TRAIN_GPU_IDS"
    elif [[ "$runtime_has_no_gpus" == "1" ]]; then
        TRAINING_CUDA_DEVICES=""
    elif [[ -n "$runtime_allocation" && "$runtime_allocation" != "all" ]]; then
        # Kubernetes commonly provides GPU UUIDs here. Using them directly
        # avoids selecting unallocated host-visible devices in privileged pods.
        TRAINING_CUDA_DEVICES="$runtime_allocation"
    else
        TRAINING_CUDA_DEVICES="$(seq -s, 0 $((TRAIN_DP - 1)))"
    fi

    if [[ -n "${SERVER_GPUS:-}" ]]; then
        SERVER_CUDA_DEVICES="$SERVER_GPUS"
    elif [[ -n "${SERVER_GPU:-}" ]]; then
        SERVER_CUDA_DEVICES="$SERVER_GPU"
    elif [[ "$runtime_has_no_gpus" == "1" ]]; then
        SERVER_CUDA_DEVICES=""
    elif [[ -n "$runtime_allocation" && "$runtime_allocation" != "all" ]]; then
        SERVER_CUDA_DEVICES="${runtime_allocation%%,*}"
    else
        SERVER_CUDA_DEVICES="0"
    fi

    read -r -a AUX_LAYER_ID_ARGS <<< "$AUX_LAYER_IDS"
}

validate_launch_identity() {
    [[ "$NUM_NODES" == "2" ]] || fail "expected NUM_NODES=2, got '${NUM_NODES}'"
    [[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] || \
        fail "expected NODE_RANK=0 or 1, got '${NODE_RANK}'"
    [[ -n "$HEAD_IP" ]] || fail "HEAD_IP/RCLI_HEAD_IP is required"
    [[ "$RUN_ROOT" != "/" ]] || fail "DISAGG_RUN_ROOT must not be /"
    [[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || \
        fail "DISAGG_STORE_ID may contain only letters, digits, '.', '_', and '-': ${RUN_ID}"
    [[ "$TRAIN_DP" =~ ^[1-9][0-9]*$ ]] || fail "TRAIN_DP must be positive: ${TRAIN_DP}"
    [[ "$SERVER_TP" =~ ^[1-9][0-9]*$ ]] || fail "SERVER_TP must be positive: ${SERVER_TP}"
    [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || fail "BATCH_SIZE must be positive: ${BATCH_SIZE}"
    [[ "$ACCUM" =~ ^[1-9][0-9]*$ ]] || fail "ACCUM must be positive: ${ACCUM}"
}

count_gpu_devices() {
    local devices="$1"
    if [[ -z "$devices" ]]; then
        printf '0\n'
    else
        awk -F, '{print NF}' <<< "$devices"
    fi
}

validate_workload_inputs() {
    [[ -s "$TRAIN_DATA_PATH" ]] || fail \
        "training data is missing: ${TRAIN_DATA_PATH}; run scripts/prepare_data.py or set TRAIN_DATA_PATH"
    [[ -f "$DRAFT_CONFIG_PATH" ]] || fail "draft config is missing: ${DRAFT_CONFIG_PATH}"

    local training_gpu_count
    local server_gpu_count
    training_gpu_count="$(count_gpu_devices "$TRAINING_CUDA_DEVICES")"
    server_gpu_count="$(count_gpu_devices "$SERVER_CUDA_DEVICES")"
    if [[ "$NODE_RANK" == "1" && "$training_gpu_count" != "$TRAIN_DP" ]]; then
        fail "training needs ${TRAIN_DP} GPUs, got ${training_gpu_count}: ${TRAINING_CUDA_DEVICES}"
    fi
    if [[ "$NODE_RANK" == "0" && "$server_gpu_count" != "$SERVER_TP" ]]; then
        fail "server TP=${SERVER_TP} needs ${SERVER_TP} GPUs, got ${server_gpu_count}: ${SERVER_CUDA_DEVICES}"
    fi
}

validate_common_runtime() {
    python - <<'PY'
import sglang

if not sglang.__version__.startswith("0.5.14"):
    raise SystemExit(f"expected sglang 0.5.14, got {sglang.__version__}")
PY
    python - <<'PY'
from mooncake.store import MooncakeDistributedStore, ReplicateConfig

print("Mooncake preflight ok", flush=True)
PY

    if [[ "$REPORT_TO" == "wandb" ]]; then
        python - <<'PY'
import wandb

required = ("login", "init", "log", "finish")
if not all(callable(getattr(wandb, name, None)) for name in required):
    raise SystemExit("REPORT_TO=wandb requires a complete W&B client: pip install wandb")
PY
    fi
}

prepare_inference_runtime() {
    validate_common_runtime
    "${ROOT_DIR}/scripts/apply_sglang_spec_capture_patch.sh"
    python - <<'PY'
import pathlib
import shutil

import sglang

sink = pathlib.Path(sglang.__file__).parent / "srt" / "spec_capture_sink.py"
if not sink.is_file():
    raise SystemExit(f"spec-capture patch is missing: {sink}")
if shutil.which("mooncake_master") is None:
    raise SystemExit("mooncake_master is not on PATH")
print(f"inference preflight ok: sglang={sglang.__version__} patch={sink}", flush=True)
PY
}

configure_common_environment() {
    export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/scripts:${PYTHONPATH:-}"
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${ROOT_DIR}/cache/compiled_kernels}"
    export FLASHINFER_DISABLE_VERSION_CHECK=1

    export MOONCAKE_MASTER_SERVER_ADDR="${MOONCAKE_MASTER_SERVER_ADDR:-${HEAD_IP}:${MOONCAKE_RPC_PORT}}"
    export MOONCAKE_METADATA_SERVER="${MOONCAKE_METADATA_SERVER:-http://${HEAD_IP}:${MOONCAKE_HTTP_PORT}/metadata}"
    export MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-tcp}"
    export MOONCAKE_GLOBAL_SEGMENT_SIZE="${MOONCAKE_GLOBAL_SEGMENT_SIZE:-$((32 << 30))}"
    export DISAGG_CLIENT_SEGMENT_SIZE="${DISAGG_CLIENT_SEGMENT_SIZE:-0}"
    export DISAGG_CLIENT_BUFFER_SIZE="${DISAGG_CLIENT_BUFFER_SIZE:-$((256 << 20))}"
    [[ "$DISAGG_CLIENT_SEGMENT_SIZE" == "0" ]] || \
        fail "DISAGG_CLIENT_SEGMENT_SIZE must be 0 for server-owned captures"

    export DISAGG_STORE_ID="$RUN_ID"
    export DISAGG_SERVER_URLS="${DISAGG_SERVER_URLS:-http://${HEAD_IP}:${SERVER_PORT}}"
    export DISAGG_REF_CHANNEL="$REF_CHANNEL"
    export DISAGG_MAX_PROMPTS="${DISAGG_MAX_PROMPTS:-0}"
    export DISAGG_MAX_STEPS="${DISAGG_MAX_STEPS:-0}"
    export DISAGG_LOG_INTERVAL="${DISAGG_LOG_INTERVAL:-10}"
    export DISAGG_PROMPT_LOG_EVERY="${DISAGG_PROMPT_LOG_EVERY:-10000}"
    # Bounds a no-marker rank-0 failure (for example, pod loss/SIGKILL).
    export DISAGG_IDLE_TIMEOUT="${DISAGG_IDLE_TIMEOUT:-600}"

    local dataset_rows
    dataset_rows="$(wc -l < "$TRAIN_DATA_PATH")"
    export DISAGG_TOTAL_STEPS="${DISAGG_TOTAL_STEPS:-$(python -c \
        'import math, sys; print(math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / (int(sys.argv[3]) * int(sys.argv[4]) * int(sys.argv[5]))))' \
        "$NUM_EPOCHS" "$dataset_rows" "$TRAIN_DP" "$BATCH_SIZE" "$ACCUM")}"

    mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
    log "configuration: run_id=${RUN_ID} rows=${dataset_rows} dp=${TRAIN_DP} batch=${BATCH_SIZE} accum=${ACCUM} epochs=${NUM_EPOCHS} steps=${DISAGG_TOTAL_STEPS}"
}

build_common_args() {
    COMMON_ARGS=(
        --target-model-path "$TARGET_MODEL_PATH"
        --target-model-backend hf
        --trust-remote-code
        --draft-config-path "$DRAFT_CONFIG_PATH"
        --mask-token-id 151669
        --train-data-path "$TRAIN_DATA_PATH"
        --chat-template "$CHAT_TEMPLATE"
        --max-length 3072
        --batch-size "$BATCH_SIZE"
        --accumulation-steps "$ACCUM"
        --learning-rate 6e-4
        --warmup-ratio 0.04
        --max-grad-norm 1.0
        --attention-backend flex_attention
        --block-size 16
        --num-anchors "$NUM_ANCHORS"
        --loss-decay-gamma 7.0
        --num-epochs "$NUM_EPOCHS"
        --seed 42
        --save-interval "$SAVE_INTERVAL"
    )
}

wait_for_file() {
    local path="$1"
    local description="$2"
    local timeout_seconds="$3"
    local abort_status_path="${4:-}"
    local started_at

    started_at="$(date +%s)"
    while true; do
        if [[ -n "$abort_status_path" && -f "$abort_status_path" ]]; then
            fail "${description} aborted with peer status $(read_status "$abort_status_path")"
        fi
        if [[ -e "$path" ]]; then
            if [[ -n "$abort_status_path" && -f "$abort_status_path" ]]; then
                fail "${description} aborted with peer status $(read_status "$abort_status_path")"
            fi
            return 0
        fi
        if (( $(date +%s) - started_at >= timeout_seconds )); then
            fail "timed out waiting for ${description}: ${path}"
        fi
        sleep 2
    done
}

run_inference_node() {
    local master_pid=""
    local server_pid=""
    local producer_pid=""
    local inference_rc=1

    mkdir -p "$(dirname "$RUN_ROOT")"
    mkdir "$RUN_ROOT" 2>/dev/null || \
        fail "run root already exists; choose a new DISAGG_STORE_ID: ${RUN_ROOT}"

    cleanup_inference() {
        log "stopping inference services"
        kill_process_group "$producer_pid"
        kill_process_group "$server_pid"
        kill_process_group "$master_pid"
    }
    record_inference_exit() {
        write_status "${RUN_ROOT}/inference.done" "$inference_rc"
    }
    trap 'cleanup_inference; record_inference_exit' EXIT
    trap 'inference_rc=129; exit 129' HUP
    trap 'inference_rc=130; exit 130' INT
    trap 'inference_rc=143; exit 143' TERM

    validate_workload_inputs
    configure_common_environment
    for channel_path in \
        "$DISAGG_REF_CHANNEL" \
        "${DISAGG_REF_CHANNEL}.closed" \
        "${DISAGG_REF_CHANNEL}.consumed_count"; do
        [[ ! -e "$channel_path" ]] || fail \
            "ref channel state already exists; choose a new path: ${channel_path}"
    done
    mkdir -p "$(dirname "$DISAGG_REF_CHANNEL")"
    : > "$DISAGG_REF_CHANNEL"
    : > "${RUN_ROOT}/consumer.log"

    prepare_inference_runtime
    touch "${RUN_ROOT}/initialized"

    export MOONCAKE_LOCAL_HOSTNAME="${INFERENCE_NODE_IP:-${HEAD_IP}}"
    log "starting Mooncake master on ${HEAD_IP}"
    setsid mooncake_master \
        --enable_http_metadata_server=true \
        --http_metadata_server_host=0.0.0.0 \
        --rpc_port="$MOONCAKE_RPC_PORT" \
        --http_metadata_server_port="$MOONCAKE_HTTP_PORT" \
        --metrics_port="$MOONCAKE_METRICS_PORT" \
        > "${RUN_ROOT}/mooncake.log" 2>&1 &
    master_pid="$!"

    local master_ready=0
    local started_at
    started_at="$(date +%s)"
    while (( $(date +%s) - started_at < SERVICE_START_TIMEOUT )); do
        kill -0 "$master_pid" 2>/dev/null || \
            fail "Mooncake master exited; see ${RUN_ROOT}/mooncake.log"
        if curl -sS --max-time 1 -o /dev/null \
            "${MOONCAKE_METADATA_SERVER}?key=specforge-health-check" && \
            timeout 1 bash -c "</dev/tcp/${HEAD_IP}/${MOONCAKE_RPC_PORT}" 2>/dev/null; then
            master_ready=1
            break
        fi
        sleep 1
    done
    [[ "$master_ready" == "1" ]] || fail "Mooncake master did not become ready"

    log "starting patched SGLang server on ${SERVER_CUDA_DEVICES}"
    setsid env \
        CUDA_VISIBLE_DEVICES="$SERVER_CUDA_DEVICES" \
        MOONCAKE_LOCAL_HOSTNAME="$MOONCAKE_LOCAL_HOSTNAME" \
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
        --spec-capture-aux-layer-ids "${AUX_LAYER_ID_ARGS[@]}" \
        --port "$SERVER_PORT" \
        > "${RUN_ROOT}/sglang-server.log" 2>&1 &
    server_pid="$!"

    local server_ready=0
    started_at="$(date +%s)"
    while (( $(date +%s) - started_at < SERVICE_START_TIMEOUT )); do
        kill -0 "$master_pid" 2>/dev/null || fail "Mooncake master exited during server startup"
        kill -0 "$server_pid" 2>/dev/null || \
            fail "SGLang server exited; see ${RUN_ROOT}/sglang-server.log"
        if curl -fsS "http://${HEAD_IP}:${SERVER_PORT}/health" > /dev/null; then
            server_ready=1
            break
        fi
        sleep 5
    done
    [[ "$server_ready" == "1" ]] || fail "SGLang server did not become ready"
    touch "${RUN_ROOT}/inference.ready"
    log "inference services ready; waiting for the consumer model"

    wait_for_file \
        "${RUN_ROOT}/consumer.started" \
        "consumer process startup" \
        "$CONSUMER_START_TIMEOUT" \
        "${RUN_ROOT}/consumer.done"
    started_at="$(date +%s)"
    while ! grep -Fq "[consumer] training from mooncake://${RUN_ID}" \
        "${RUN_ROOT}/consumer.log" 2>/dev/null; do
        kill -0 "$master_pid" 2>/dev/null || fail "Mooncake master exited before consumer readiness"
        kill -0 "$server_pid" 2>/dev/null || fail "SGLang server exited before consumer readiness"
        if [[ -f "${RUN_ROOT}/consumer.done" ]]; then
            fail "consumer exited before readiness with code $(read_status "${RUN_ROOT}/consumer.done")"
        fi
        if (( $(date +%s) - started_at >= CONSUMER_START_TIMEOUT )); then
            fail "timed out waiting for consumer model readiness"
        fi
        sleep 2
    done

    log "consumer is ready; starting the CPU producer"
    setsid env DISAGG_ROLE=producer CUDA_VISIBLE_DEVICES="" \
        python "${SCRIPT_DIR}/run_disagg_dflash.py" \
        "${COMMON_ARGS[@]}" \
        --output-dir "${RUN_ROOT}/producer" \
        > >(tee "${RUN_ROOT}/producer.log") 2>&1 &
    producer_pid="$!"

    local consumer_finished_early=0
    while kill -0 "$producer_pid" 2>/dev/null; do
        kill -0 "$master_pid" 2>/dev/null || fail "Mooncake master exited during production"
        kill -0 "$server_pid" 2>/dev/null || fail "SGLang server exited during production"
        if [[ -f "${RUN_ROOT}/consumer.done" ]]; then
            local early_consumer_rc
            early_consumer_rc="$(read_status "${RUN_ROOT}/consumer.done")"
            [[ "$early_consumer_rc" == "0" ]] || fail "consumer exited with code ${early_consumer_rc}"
            consumer_finished_early=1
            break
        fi
        sleep 5
    done

    if [[ "$consumer_finished_early" == "1" ]]; then
        if [[ "$DISAGG_MAX_STEPS" == "0" ]]; then
            log "uncapped consumer completed early; allowing producer five seconds to close"
            sleep 5
        fi
        if [[ "$DISAGG_MAX_STEPS" == "0" ]] && kill -0 "$producer_pid" 2>/dev/null; then
            kill_process_group "$producer_pid"
            set +e
            wait "$producer_pid"
            local early_producer_rc="$?"
            set -e
            producer_pid=""
            write_status "${RUN_ROOT}/producer.done" "$early_producer_rc"
            fail "uncapped consumer completed before producer (producer code ${early_producer_rc})"
        fi

        local stopped_producer_rc
        if [[ "$DISAGG_MAX_STEPS" == "0" ]]; then
            log "producer closed after the uncapped consumer drained its refs"
            set +e
            wait "$producer_pid"
            stopped_producer_rc="$?"
            set -e
            producer_pid=""
            write_status "${RUN_ROOT}/producer.done" "$stopped_producer_rc"
            [[ "$stopped_producer_rc" == "0" ]] || \
                fail "producer exited with code ${stopped_producer_rc}"
        else
            log "capped consumer completed; stopping remaining production"
            kill_process_group "$producer_pid"
            set +e
            wait "$producer_pid"
            stopped_producer_rc="$?"
            set -e
            producer_pid=""
            write_status "${RUN_ROOT}/producer.done" "$stopped_producer_rc"
        fi
    else
        local producer_rc
        set +e
        wait "$producer_pid"
        producer_rc="$?"
        set -e
        producer_pid=""
        write_status "${RUN_ROOT}/producer.done" "$producer_rc"
        [[ "$producer_rc" == "0" ]] || fail "producer exited with code ${producer_rc}"

        log "producer completed; retaining Mooncake objects until training exits"
        while [[ ! -f "${RUN_ROOT}/consumer.done" ]]; do
            kill -0 "$master_pid" 2>/dev/null || fail "Mooncake master exited during training"
            kill -0 "$server_pid" 2>/dev/null || fail "SGLang server exited during training"
            sleep 10
        done
        local consumer_rc
        consumer_rc="$(read_status "${RUN_ROOT}/consumer.done")"
        [[ "$consumer_rc" == "0" ]] || fail "consumer exited with code ${consumer_rc}"
    fi

    inference_rc=0
    log "training completed successfully"
    cleanup_inference
    record_inference_exit
    trap - EXIT HUP INT TERM
}

run_training_node() {
    wait_for_file \
        "${RUN_ROOT}/initialized" \
        "rank-0 run initialization" \
        600 \
        "${RUN_ROOT}/inference.done"

    local trainer_pid=""
    local consumer_rc=1
    record_consumer_exit() {
        write_status "${RUN_ROOT}/consumer.done" "$consumer_rc"
    }
    cleanup_training() {
        kill_process_group "$trainer_pid"
    }
    trap 'cleanup_training; record_consumer_exit' EXIT
    trap 'consumer_rc=129; exit 129' HUP
    trap 'consumer_rc=130; exit 130' INT
    trap 'consumer_rc=143; exit 143' TERM

    validate_workload_inputs
    configure_common_environment
    validate_common_runtime
    wait_for_file \
        "${RUN_ROOT}/inference.ready" \
        "inference readiness" \
        "$SERVICE_START_TIMEOUT" \
        "${RUN_ROOT}/inference.done"

    export MOONCAKE_LOCAL_HOSTNAME="${TRAIN_NODE_IP:-$(resolve_local_ip)}"
    local local_run_root="${DISAGG_LOCAL_RUN_ROOT:-/tmp/${RUN_ID}}"
    [[ "$local_run_root" != "/" ]] || fail "DISAGG_LOCAL_RUN_ROOT must not be /"
    mkdir -p "$(dirname "$local_run_root")"
    mkdir "$local_run_root" 2>/dev/null || \
        fail "local run root already exists; choose a new DISAGG_STORE_ID: ${local_run_root}"
    mkdir "${local_run_root}/inboxes"

    : > "${RUN_ROOT}/consumer.log"
    touch "${RUN_ROOT}/consumer.started"
    log "starting DP=${TRAIN_DP} DFlash consumer on ${TRAINING_CUDA_DEVICES}"
    setsid env \
        CUDA_VISIBLE_DEVICES="$TRAINING_CUDA_DEVICES" \
        DISAGG_ROLE=consumer \
        DISAGG_DB="${local_run_root}/run.db" \
        DISAGG_INBOX_DIR="${local_run_root}/inboxes" \
        torchrun \
        --standalone \
        --nnodes 1 \
        --nproc_per_node "$TRAIN_DP" \
        "${SCRIPT_DIR}/run_disagg_dflash.py" \
        "${COMMON_ARGS[@]}" \
        --output-dir "${RUN_ROOT}/consumer" \
        --report-to "$REPORT_TO" \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-name "${WANDB_NAME:-${RUN_ID}}" \
        > >(tee "${RUN_ROOT}/consumer.log") 2>&1 &
    trainer_pid="$!"

    local inference_failed=0
    while kill -0 "$trainer_pid" 2>/dev/null; do
        if [[ -f "${RUN_ROOT}/inference.done" ]]; then
            local inference_rc
            inference_rc="$(read_status "${RUN_ROOT}/inference.done")"
            if [[ "$inference_rc" != "0" ]]; then
                log "inference node exited with code ${inference_rc}; stopping trainer"
                inference_failed=1
                consumer_rc="$inference_rc"
                kill_process_group "$trainer_pid"
                break
            fi
        fi
        sleep 5
    done

    set +e
    wait "$trainer_pid"
    local trainer_rc="$?"
    set -e
    trainer_pid=""
    if [[ "$inference_failed" == "0" ]]; then
        consumer_rc="$trainer_rc"
    fi

    if [[ "$consumer_rc" == "0" ]]; then
        log "consumer completed successfully"
    else
        log "consumer exited with code ${consumer_rc}"
    fi
    record_consumer_exit
    trap - EXIT HUP INT TERM
    exit "$consumer_rc"
}

main() {
    validate_launch_identity
    resolve_gpu_devices
    build_common_args
    cd "$ROOT_DIR"

    case "$NODE_RANK" in
        0) run_inference_node ;;
        1) run_training_node ;;
    esac
}

main "$@"
