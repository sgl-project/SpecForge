#!/usr/bin/env bash
# Launch one role of a disaggregated run through the same public training CLI.
set -euo pipefail

: "${CONFIG:?set CONFIG to a disaggregated run YAML}"
ROLE=${ROLE:-producer}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

case "$ROLE" in
    producer)
        exec specforge train --config "$CONFIG" training.role=producer
        ;;
    consumer)
        exec torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" \
            "$(command -v specforge)" train --config "$CONFIG" \
            training.role=consumer
        ;;
    *)
        echo "ROLE must be producer or consumer, got: $ROLE" >&2
        exit 2
        ;;
esac
