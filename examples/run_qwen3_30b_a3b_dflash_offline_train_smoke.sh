#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/workspace/SpecForge}
cd "$ROOT_DIR"

source /venv/main/bin/activate
source /workspace/.env

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python scripts/train_dflash_offline.py \
  --target-model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --train-hidden-states-path cache/dflash_hidden_states/qwen3_30b_fullcap_10k_12layers \
  --train-hidden-states-path cache/dflash_hidden_states/qwen3_30b_fullcap_10k_s001_12layers \
  --eval-hidden-states-path cache/dflash_hidden_states/qwen3_30b_fullcap_10k_s005_12layers \
  --output-dir cache/training_smokes/qwen3_30b_dflash_offline_train_smoke \
  --max-train-samples 64 \
  --max-eval-samples 16 \
  --max-steps 10 \
  --max-length 512 \
  --batch-size 1 \
  --num-draft-layers 1 \
  --block-size 16 \
  --num-anchors 32 \
  --attention-backend sdpa \
  --learning-rate 1e-4 \
  --eval-interval 5 \
  --save-interval 5 \
  --log-interval 1 \
  --wandb-project specforge-qwen3-30b-dflash \
  --wandb-entity artin_kim-etched- \
  --wandb-name qwen3-30b-a3b-dflash-offline-train-smoke \
  --hf-repo-id qwen3-30b-dflash/qwen3-30b-a3b-dflash-smokes \
  --push-to-hub \
  --trust-remote-code \
  --shuffle-files
