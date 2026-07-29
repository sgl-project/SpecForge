#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PROFILE=qwen3-8b exec bash "${SCRIPT_DIR}/run_qwen_sharegpt_regeneration.sh" "$@"
