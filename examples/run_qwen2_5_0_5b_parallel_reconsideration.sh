#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    printf 'Usage: %s FEATURE_DIR OUTPUT_DIR [--plan] [section.field=value ...]\n' "$0"
    exit 0
fi
if [[ $# -lt 2 ]]; then
    printf 'Specify FEATURE_DIR and OUTPUT_DIR. Run %s --help for usage.\n' "$0" >&2
    exit 2
fi

features=$1
output=$2
shift 2
exec python -m specforge.cli train \
    -c examples/configs/offline/colocated/qwen2.5-0.5b-parallel-reconsideration.yaml \
    "data.hidden_states_path=$features" "output_dir=$output" "$@"
