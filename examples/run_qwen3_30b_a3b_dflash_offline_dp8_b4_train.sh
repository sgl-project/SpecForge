#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/workspace/SpecForge}
cd "$ROOT_DIR"

source /venv/main/bin/activate
source /workspace/.env

PROJECT=${WANDB_PROJECT:-specforge-qwen3-30b-dflash}
ENTITY=${WANDB_ENTITY:-artin_kim-etched-}
HF_REPO=${HF_REPO:-qwen3-30b-dflash/qwen3-30b-a3b-dflash-smokes}
BASE_OUT=${BASE_OUT:-cache/training_sweeps/qwen3_30b_dflash_offline_dp8}
NAME=${WANDB_NAME:-q30b-dflash-dp8-b4-a512-l2048-s10000}
LOGFILE=${LOGFILE:-logs/${NAME}.log}

mkdir -p "$BASE_OUT" logs

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}

{
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    scripts/train_dflash_offline.py \
      --target-model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
      --train-hidden-states-path cache/dflash_hidden_states/qwen3_30b_fullcap_10k_12layers \
      --train-hidden-states-path cache/dflash_hidden_states/qwen3_30b_fullcap_10k_s001_12layers \
      --train-hidden-states-path cache/dflash_hidden_states/qwen3_30b_fullcap_10k_s002_12layers \
      --train-hidden-states-path cache/dflash_hidden_states/qwen3_30b_fullcap_10k_s003_12layers \
      --train-hidden-states-path cache/dflash_hidden_states/qwen3_30b_fullcap_10k_s004_12layers \
      --eval-hidden-states-path cache/dflash_hidden_states/qwen3_30b_fullcap_10k_s005_12layers \
      --output-dir "$BASE_OUT/$NAME" \
      --max-eval-samples 512 \
      --max-steps 10000 \
      --max-length 2048 \
      --batch-size 4 \
      --num-draft-layers 1 \
      --block-size 16 \
      --num-anchors 512 \
      --attention-backend sdpa \
      --learning-rate 1e-4 \
      --eval-interval 250 \
      --save-interval 1250 \
      --log-interval 10 \
      --loss-type dflash \
      --seed 70 \
      --num-workers 1 \
      --prefetch-factor 2 \
      --pin-memory \
      --wandb-project "$PROJECT" \
      --wandb-entity "$ENTITY" \
      --wandb-name "$NAME" \
      --hf-repo-id "$HF_REPO" \
      --push-to-hub \
      --trust-remote-code \
      --shuffle-files
} >"$LOGFILE" 2>&1
