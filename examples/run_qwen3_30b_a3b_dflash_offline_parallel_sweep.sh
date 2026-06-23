#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/workspace/SpecForge}
cd "$ROOT_DIR"

source /venv/main/bin/activate
source /workspace/.env

PROJECT=${WANDB_PROJECT:-specforge-qwen3-30b-dflash}
ENTITY=${WANDB_ENTITY:-artin_kim-etched-}
HF_REPO=${HF_REPO:-qwen3-30b-dflash/qwen3-30b-a3b-dflash-smokes}
BASE_OUT=${BASE_OUT:-cache/training_sweeps/qwen3_30b_dflash_offline_parallel_sweep}
mkdir -p "$BASE_OUT" logs

run_one() {
  local gpu="$1"
  local name="$2"
  local anchors="$3"
  local max_len="$4"
  local steps="$5"
  local loss_type="$6"
  local lr="${7:-1e-4}"

  CUDA_VISIBLE_DEVICES="$gpu" python scripts/train_dflash_offline.py \
    --target-model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --train-hidden-states-path cache/dflash_hidden_states/qwen3_30b_fullcap_10k_12layers \
    --train-hidden-states-path cache/dflash_hidden_states/qwen3_30b_fullcap_10k_s001_12layers \
    --train-hidden-states-path cache/dflash_hidden_states/qwen3_30b_fullcap_10k_s002_12layers \
    --eval-hidden-states-path cache/dflash_hidden_states/qwen3_30b_fullcap_10k_s005_12layers \
    --output-dir "$BASE_OUT/$name" \
    --max-train-samples 512 \
    --max-eval-samples 64 \
    --max-steps "$steps" \
    --max-length "$max_len" \
    --batch-size 1 \
    --num-draft-layers 1 \
    --block-size 16 \
    --num-anchors "$anchors" \
    --attention-backend sdpa \
    --learning-rate "$lr" \
    --eval-interval 25 \
    --save-interval "$steps" \
    --log-interval 1 \
    --loss-type "$loss_type" \
    --wandb-project "$PROJECT" \
    --wandb-entity "$ENTITY" \
    --wandb-name "$name" \
    --hf-repo-id "$HF_REPO" \
    --push-to-hub \
    --trust-remote-code \
    --shuffle-files
}

run_one 0 q30b-dflash-b16-a128-l1024-s100 128 1024 100 dflash &
run_one 1 q30b-dflash-b16-a256-l1024-s100 256 1024 100 dflash &
run_one 2 q30b-dflash-b16-a512-l1024-s100 512 1024 100 dflash &
run_one 3 q30b-dflash-b16-a128-l2048-s100 128 2048 100 dflash &
run_one 4 q30b-dflash-b16-a256-l2048-s100 256 2048 100 dflash &
run_one 5 q30b-dflash-b16-a512-l2048-s50 512 2048 50 dflash &
run_one 6 q30b-dflash-b16-a512-l512-s100 512 512 100 dflash &
run_one 7 q30b-dpace-b16-a256-l1024-s100 256 1024 100 dpace &

wait
