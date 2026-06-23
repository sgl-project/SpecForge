#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/workspace/SpecForge}
cd "$ROOT_DIR"

while tmux has-session -t dflash_long_sweep 2>/dev/null; do
  sleep 30
done

while pgrep -f "scripts/train_dflash_offline.py" >/dev/null; do
  sleep 30
done

if tmux has-session -t dflash_focused_sweep 2>/dev/null; then
  echo "dflash_focused_sweep already running"
  exit 0
fi

tmux new-session -d -s dflash_focused_sweep "bash examples/run_qwen3_30b_a3b_dflash_offline_focused_sweep.sh"
echo "started dflash_focused_sweep"
