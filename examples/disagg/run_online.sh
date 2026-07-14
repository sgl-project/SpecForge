#!/usr/bin/env bash
# Launch one online-disaggregated role through the canonical SpecForge CLI.
set -euo pipefail

usage() {
    printf '%s\n' \
        "Usage: CONFIG=path/to/run.yaml $0 <producer|consumer> [CONFIG_OVERRIDE ...]" \
        "" \
        "Required for both roles:" \
        "  CONFIG DISAGG_REF_CHANNEL MOONCAKE_METADATA_SERVER" \
        "  MOONCAKE_MASTER_SERVER_ADDR MOONCAKE_LOCAL_HOSTNAME" \
        "Producer server URLs may come from the config or DISAGG_SERVER_URL(S)." \
        "Consumer NPROC_PER_NODE defaults to 1." \
        "  Every consumer requires a fresh DISAGG_DB."
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
channel=${DISAGG_REF_CHANNEL:-}
metadata_server=${MOONCAKE_METADATA_SERVER:-}
master_server=${MOONCAKE_MASTER_SERVER_ADDR:-}
local_hostname=${MOONCAKE_LOCAL_HOSTNAME:-}

require_value "$config" CONFIG
[[ -f "$config" ]] || fail "CONFIG does not exist: $config"
require_value "$channel" DISAGG_REF_CHANNEL
require_value "$metadata_server" MOONCAKE_METADATA_SERVER
require_value "$master_server" MOONCAKE_MASTER_SERVER_ADDR
require_value "$local_hostname" MOONCAKE_LOCAL_HOSTNAME

specforge_bin=$(command -v specforge) || fail "specforge is not on PATH"

case "$role" in
    producer)
        exec "$specforge_bin" train --config "$config" \
            "$@" training.role=producer
        ;;
    consumer)
        database=${DISAGG_DB:-}
        nproc=${NPROC_PER_NODE:-1}
        [[ "$nproc" =~ ^[1-9][0-9]*$ ]] || {
            fail "NPROC_PER_NODE must be a positive integer, got: $nproc"
        }
        require_value "$database" DISAGG_DB
        if [[ -e "$database" || -e "${database}-wal" || -e "${database}-shm" ]]; then
            fail "DISAGG_DB must be a fresh attempt path: $database"
        fi
        torchrun_bin=$(command -v torchrun) || fail "torchrun is not on PATH"
        exec "$torchrun_bin" --standalone --nproc_per_node "$nproc" \
            "$specforge_bin" train --config "$config" \
            "$@" "training.metadata_db_path=$database" \
            training.role=consumer
        ;;
esac
