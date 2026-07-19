#!/usr/bin/env bash
# Rank-dispatch one offline disaggregated config through the public CLI.
set -euo pipefail

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    printf '%s\n' \
        "Usage: CONFIG=path/to/offline-disagg.yaml $0 [SPECFORGE_TRAIN_ARG ...]" \
        "Run the same command on two nodes; RCLI_NODE_RANK 0 selects producer" \
        "and RCLI_NODE_RANK 1 selects consumer (NODE_RANK is also accepted)." \
        "The YAML owns shared store/control paths and the consumer topology."
    exit 0
fi

config=${CONFIG:-}
[[ -n "$config" ]] || {
    printf 'error: set CONFIG\n' >&2
    exit 2
}
[[ -f "$config" ]] || {
    printf 'error: CONFIG does not exist: %s\n' "$config" >&2
    exit 2
}

node_rank=${RCLI_NODE_RANK:-${NODE_RANK:-}}
case "$node_rank" in
    0) role=producer ;;
    1) role=consumer ;;
    *)
        printf 'error: RCLI_NODE_RANK (or NODE_RANK) must be 0 or 1\n' >&2
        exit 2
        ;;
esac

num_nodes=${RCLI_NUM_NODES:-${NUM_NODES:-}}
if [[ -n "$num_nodes" && "$num_nodes" != 2 ]]; then
    printf 'error: RCLI_NUM_NODES (or NUM_NODES) must be 2\n' >&2
    exit 2
fi

specforge_bin=$(command -v specforge) || {
    printf 'error: specforge is not on PATH\n' >&2
    exit 2
}
exec "$specforge_bin" train --config "$config" --role "$role" "$@"
