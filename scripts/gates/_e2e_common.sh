#!/usr/bin/env bash
# Shared process lifecycle and dry-run helpers for executable E2E gates.

GATE_BACKGROUND_PIDS=()
GATE_BACKGROUND_MODES=()
GATE_BACKGROUND_LABELS=()
GATE_LAST_PID=""

gate_fail() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

gate_is_true() {
    case "${1:-}" in
        1 | true | TRUE | yes | YES | on | ON) return 0 ;;
        0 | false | FALSE | no | NO | off | OFF | "") return 1 ;;
        *) gate_fail "${2:-boolean value} must be true or false, got: $1" ;;
    esac
}

gate_require_value() {
    [[ -n "${1:-}" ]] || gate_fail "set $2"
}

gate_require_file() {
    [[ -f "$1" ]] || gate_fail "$2 does not exist or is not a file: $1"
}

gate_require_directory() {
    [[ -d "$1" ]] || gate_fail "$2 does not exist or is not a directory: $1"
}

gate_require_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]] || {
        gate_fail "$2 must be a positive integer, got: $1"
    }
}

gate_require_nonnegative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]] || {
        gate_fail "$2 must be a non-negative integer, got: $1"
    }
}

gate_require_command() {
    local candidate=$1
    local label=$2
    if [[ "$candidate" == */* ]]; then
        [[ -f "$candidate" && -x "$candidate" ]] || {
            gate_fail "$label is not an executable file: $candidate"
        }
        return 0
    fi
    command -v "$candidate" >/dev/null 2>&1 || {
        gate_fail "$label is not on PATH: $candidate"
    }
}

gate_require_tcp_port_free() {
    local python=$1
    local host=$2
    local port=$3
    local label=$4

    if "$python" - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
for family, socktype, proto, _, address in socket.getaddrinfo(
    host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
):
    sock = socket.socket(family, socktype, proto)
    try:
        sock.bind(address)
    except OSError:
        sock.close()
        raise SystemExit(1)
    sock.close()
PY
    then
        return 0
    fi
    gate_fail "$label is already in use at $host:$port"
}

gate_dry_run() {
    gate_is_true "${GATE_DRY_RUN:-0}" GATE_DRY_RUN
}

gate_print_command() {
    local argument
    printf '+'
    for argument in "$@"; do
        printf ' %q' "$argument"
    done
}

gate_run() {
    gate_print_command "$@"
    printf '\n'
    if gate_dry_run; then
        return 0
    fi
    "$@"
}

gate_run_with_tee() {
    local log_path=$1
    local command_status
    shift

    gate_print_command "$@"
    printf ' 2>&1 | tee %q\n' "$log_path"
    if gate_dry_run; then
        return 0
    fi

    set +e
    "$@" 2>&1 | tee "$log_path"
    command_status=${PIPESTATUS[0]}
    set -e
    return "$command_status"
}

gate_start_service() {
    local label=$1
    local log_path=$2
    local mode
    local pid
    shift 2

    gate_print_command "$@"
    printf ' > %q 2>&1 &\n' "$log_path"
    if gate_dry_run; then
        GATE_LAST_PID=""
        return 0
    fi

    if command -v setsid >/dev/null 2>&1; then
        setsid "$@" >"$log_path" 2>&1 &
        mode=group
    else
        "$@" >"$log_path" 2>&1 &
        mode=process
    fi
    pid=$!
    GATE_BACKGROUND_PIDS+=("$pid")
    GATE_BACKGROUND_MODES+=("$mode")
    GATE_BACKGROUND_LABELS+=("$label")
    GATE_LAST_PID=$pid
}

_gate_process_alive() {
    local pid=$1
    local mode=$2
    if [[ "$mode" == group ]]; then
        kill -0 -- "-$pid" 2>/dev/null
    else
        kill -0 "$pid" 2>/dev/null
    fi
}

_gate_signal_process() {
    local signal=$1
    local pid=$2
    local mode=$3
    if [[ "$mode" == group ]]; then
        kill "-$signal" -- "-$pid" 2>/dev/null || true
    else
        kill "-$signal" "$pid" 2>/dev/null || true
    fi
}

gate_forget_pid() {
    local target=$1
    local index
    for ((index = 0; index < ${#GATE_BACKGROUND_PIDS[@]}; index++)); do
        if [[ "${GATE_BACKGROUND_PIDS[$index]}" == "$target" ]]; then
            GATE_BACKGROUND_PIDS[$index]=""
            return 0
        fi
    done
}

gate_wait_pid() {
    local pid=$1
    local label=$2
    local log_path=$3
    local status

    if gate_dry_run; then
        return 0
    fi
    if wait "$pid"; then
        status=0
    else
        status=$?
    fi
    gate_forget_pid "$pid"
    if ((status != 0)); then
        printf '%s exited with status %s; see %s\n' \
            "$label" "$status" "$log_path" >&2
        return "$status"
    fi
}

gate_wait_http() {
    local pid=$1
    local label=$2
    local url=$3
    local log_path=$4
    local accept_any_status=${5:-false}
    local attempts=${GATE_HEALTH_ATTEMPTS:-180}
    local interval=${GATE_HEALTH_INTERVAL_SECONDS:-2}
    local attempt

    if gate_dry_run; then
        return 0
    fi
    gate_require_positive_integer "$attempts" GATE_HEALTH_ATTEMPTS

    for ((attempt = 1; attempt <= attempts; attempt++)); do
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
            gate_fail "$label exited during startup; see $log_path"
        fi
        if gate_is_true "$accept_any_status" accept_any_status; then
            if curl -sS --max-time 2 -o /dev/null "$url" 2>/dev/null; then
                return 0
            fi
        elif curl -fsS --max-time 2 -o /dev/null "$url" 2>/dev/null; then
            return 0
        fi

        sleep "$interval"
    done
    gate_fail "$label did not become ready at $url; see $log_path"
}

gate_stop_services() {
    local index
    local attempt
    local any_alive
    local pid
    local mode

    if gate_dry_run; then
        GATE_BACKGROUND_PIDS=()
        GATE_BACKGROUND_MODES=()
        GATE_BACKGROUND_LABELS=()
        return 0
    fi

    for ((index = ${#GATE_BACKGROUND_PIDS[@]} - 1; index >= 0; index--)); do
        pid=${GATE_BACKGROUND_PIDS[$index]}
        [[ -n "$pid" ]] || continue
        mode=${GATE_BACKGROUND_MODES[$index]}
        if _gate_process_alive "$pid" "$mode"; then
            _gate_signal_process TERM "$pid" "$mode"
        fi
    done

    for ((attempt = 0; attempt < 50; attempt++)); do
        any_alive=false
        for ((index = 0; index < ${#GATE_BACKGROUND_PIDS[@]}; index++)); do
            pid=${GATE_BACKGROUND_PIDS[$index]}
            [[ -n "$pid" ]] || continue
            mode=${GATE_BACKGROUND_MODES[$index]}
            if _gate_process_alive "$pid" "$mode"; then
                any_alive=true
                break
            fi
        done
        if [[ "$any_alive" == false ]]; then
            break
        fi
        sleep 0.1
    done

    for ((index = ${#GATE_BACKGROUND_PIDS[@]} - 1; index >= 0; index--)); do
        pid=${GATE_BACKGROUND_PIDS[$index]}
        [[ -n "$pid" ]] || continue
        mode=${GATE_BACKGROUND_MODES[$index]}
        if _gate_process_alive "$pid" "$mode"; then
            _gate_signal_process KILL "$pid" "$mode"
        fi
        wait "$pid" 2>/dev/null || true
    done

    GATE_BACKGROUND_PIDS=()
    GATE_BACKGROUND_MODES=()
    GATE_BACKGROUND_LABELS=()
}

gate_cleanup() {
    local status=$?
    trap - EXIT INT TERM
    gate_stop_services || true
    exit "$status"
}

gate_install_cleanup_traps() {
    trap gate_cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
}
