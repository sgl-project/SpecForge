#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/workspace/SpecForge}
cd "$ROOT_DIR"

source /venv/main/bin/activate
source /workspace/.env

python scripts/smoke_train_dflash_offline.py \
  --target-model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --hidden-states-path cache/dflash_hidden_states/qwen3_30b_fullcap_10k_12layers \
  --hidden-states-path cache/dflash_hidden_states/qwen3_30b_fullcap_10k_s001_12layers \
  --output-dir cache/training_smokes/qwen3_30b_dflash_offline_smoke \
  --max-samples 16 \
  --max-steps 2 \
  --max-length 512 \
  --batch-size 1 \
  --num-draft-layers 1 \
  --block-size 16 \
  --num-anchors 8 \
  --attention-backend sdpa \
  --learning-rate 1e-4 \
  --trust-remote-code
