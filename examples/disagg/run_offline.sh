#!/usr/bin/env bash
# Launch one offline-disaggregated EAGLE3 role through the canonical CLI.
set -euo pipefail

usage() {
    printf '%s\n' \
        "Usage: CONFIG=path/to/run.yaml $0 <producer|consumer> [CONFIG_OVERRIDE ...]" \
        "" \
        "Required for both roles:" \
        "  CONFIG DISAGG_MANIFEST" \
        "For DISAGG_BACKEND=shared_dir (default): DISAGG_STORE_ROOT" \
        "For DISAGG_BACKEND=mooncake: MOONCAKE_METADATA_SERVER," \
        "  MOONCAKE_MASTER_SERVER_ADDR, and MOONCAKE_LOCAL_HOSTNAME"
}

fail() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

require_value() {
    [[ -n "$1" ]] || fail "set $2"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
[[ $# -ge 1 ]] || {
    usage >&2
    exit 2
}

role=$1
shift
case "$role" in
    producer | consumer) ;;
    *) fail "role must be producer or consumer, got: $role" ;;
esac

config=${CONFIG:-}
manifest=${DISAGG_MANIFEST:-}
backend=${DISAGG_BACKEND:-shared_dir}
export DISAGG_BACKEND=$backend

require_value "$config" CONFIG
[[ -f "$config" ]] || fail "CONFIG does not exist: $config"
require_value "$manifest" DISAGG_MANIFEST

case "$backend" in
    shared_dir)
        require_value "${DISAGG_STORE_ROOT:-}" DISAGG_STORE_ROOT
        ;;
    mooncake)
        require_value "${MOONCAKE_METADATA_SERVER:-}" MOONCAKE_METADATA_SERVER
        require_value "${MOONCAKE_MASTER_SERVER_ADDR:-}" \
            MOONCAKE_MASTER_SERVER_ADDR
        require_value "${MOONCAKE_LOCAL_HOSTNAME:-}" MOONCAKE_LOCAL_HOSTNAME
        ;;
    *)
        fail "DISAGG_BACKEND must be shared_dir or mooncake, got: $backend"
        ;;
esac

specforge_bin=$(command -v specforge) || fail "specforge is not on PATH"

case "$role" in
    producer)
        exec "$specforge_bin" train --config "$config" \
            "$@" training.role=producer
        ;;
    consumer)
        nproc=${NPROC_PER_NODE:-1}
        [[ "$nproc" == "1" ]] || {
            fail "offline consumer is single-rank; got NPROC_PER_NODE=$nproc"
        }
        exec "$specforge_bin" train --config "$config" \
            "$@" training.role=consumer
        ;;
esac
