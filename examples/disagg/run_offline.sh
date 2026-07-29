#!/usr/bin/env bash
# Thin offline-disaggregated example over the one public training entry.
set -euo pipefail

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    printf '%s\n' \
        "Usage: CONFIG=path/to/offline-disagg.yaml $0 [SPECFORGE_TRAIN_ARG ...]" \
        "Example: CONFIG=... $0 --role consumer" \
        "With no --role, a single-node config supervises producer and consumer."
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

specforge_bin=$(command -v specforge) || {
    printf 'error: specforge is not on PATH\n' >&2
    exit 2
}
exec "$specforge_bin" train --config "$config" "$@"
