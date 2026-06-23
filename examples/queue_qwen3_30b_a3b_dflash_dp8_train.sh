#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/workspace/SpecForge}
cd "$ROOT_DIR"

while pgrep -f "scripts/train_dflash_offline.py" >/dev/null; do
  sleep 15
done

if tmux has-session -t dflash_dp8_train 2>/dev/null; then
  echo "dflash_dp8_train already running"
  exit 0
fi

tmux new-session -d -s dflash_dp8_train "bash examples/run_qwen3_30b_a3b_dflash_offline_dp8_train.sh"
echo "started dflash_dp8_train"
