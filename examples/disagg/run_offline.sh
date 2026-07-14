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
        "  MOONCAKE_MASTER_SERVER_ADDR, and MOONCAKE_LOCAL_HOSTNAME" \
        "Consumer NPROC_PER_NODE and NNODES default to 1." \
        "  For NNODES>1, also set NODE_RANK, MASTER_ADDR, and MASTER_PORT."
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
        nnodes=${NNODES:-1}
        [[ "$nproc" =~ ^[1-9][0-9]*$ ]] || {
            fail "NPROC_PER_NODE must be a positive integer, got: $nproc"
        }
        [[ "$nnodes" =~ ^[1-9][0-9]*$ ]] || {
            fail "NNODES must be a positive integer, got: $nnodes"
        }
        if ((nnodes == 1)); then
            node_rank=${NODE_RANK:-0}
            [[ "$node_rank" == "0" ]] || {
                fail "NODE_RANK must be 0 when NNODES=1, got: $node_rank"
            }
            torchrun_args=(--standalone --nproc_per_node "$nproc")
        else
            node_rank=${NODE_RANK:-}
            master_addr=${MASTER_ADDR:-}
            master_port=${MASTER_PORT:-}
            [[ "$node_rank" =~ ^[0-9]+$ ]] || {
                fail "NODE_RANK must be a non-negative integer for NNODES>1"
            }
            ((node_rank < nnodes)) || {
                fail "NODE_RANK must be smaller than NNODES, got: $node_rank"
            }
            require_value "$master_addr" MASTER_ADDR
            [[ "$master_port" =~ ^[1-9][0-9]*$ ]] && \
                ((master_port <= 65535)) || {
                fail "MASTER_PORT must be an integer from 1 to 65535"
            }
            torchrun_args=(
                --nnodes "$nnodes"
                --node_rank "$node_rank"
                --master_addr "$master_addr"
                --master_port "$master_port"
                --nproc_per_node "$nproc"
            )
        fi
        torchrun_bin=$(command -v torchrun) || fail "torchrun is not on PATH"
        exec "$torchrun_bin" "${torchrun_args[@]}" \
            "$specforge_bin" train --config "$config" \
            "$@" training.role=consumer
        ;;
esac
