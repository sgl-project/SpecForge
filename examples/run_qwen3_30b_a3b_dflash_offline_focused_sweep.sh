#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/workspace/SpecForge}
cd "$ROOT_DIR"

source /venv/main/bin/activate
source /workspace/.env

PROJECT=${WANDB_PROJECT:-specforge-qwen3-30b-dflash}
ENTITY=${WANDB_ENTITY:-artin_kim-etched-}
HF_REPO=${HF_REPO:-qwen3-30b-dflash/qwen3-30b-a3b-dflash-smokes}
BASE_OUT=${BASE_OUT:-cache/training_sweeps/qwen3_30b_dflash_offline_focused_sweep}
mkdir -p "$BASE_OUT" logs

run_one() {
  local gpu="$1"
  local name="$2"
  local anchors="$3"
  local max_len="$4"
  local steps="$5"
  local lr="$6"
  local seed="$7"
  local draft_layers="$8"
  local max_train_samples="$9"
  local max_eval_samples="${10}"
  local logfile="logs/${name}.log"

  echo "starting gpu=${gpu} name=${name} anchors=${anchors} max_len=${max_len} steps=${steps} layers=${draft_layers} log=${logfile}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python -u scripts/train_dflash_offline.py \
      --target-model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
      --train-hidden-states-path cache/dflash_hidden_states/qwen3_30b_fullcap_10k_12layers \
      --train-hidden-states-path cache/dflash_hidden_states/qwen3_30b_fullcap_10k_s001_12layers \
      --train-hidden-states-path cache/dflash_hidden_states/qwen3_30b_fullcap_10k_s002_12layers \
      --train-hidden-states-path cache/dflash_hidden_states/qwen3_30b_fullcap_10k_s003_12layers \
      --train-hidden-states-path cache/dflash_hidden_states/qwen3_30b_fullcap_10k_s004_12layers \
      --eval-hidden-states-path cache/dflash_hidden_states/qwen3_30b_fullcap_10k_s005_12layers \
      --output-dir "$BASE_OUT/$name" \
      --max-train-samples "$max_train_samples" \
      --max-eval-samples "$max_eval_samples" \
      --max-steps "$steps" \
      --max-length "$max_len" \
      --batch-size 1 \
      --num-draft-layers "$draft_layers" \
      --block-size 16 \
      --num-anchors "$anchors" \
      --attention-backend sdpa \
      --learning-rate "$lr" \
      --eval-interval 250 \
      --save-interval 2000 \
      --log-interval 10 \
      --loss-type dflash \
      --seed "$seed" \
      --num-workers 2 \
      --prefetch-factor 2 \
      --pin-memory \
      --wandb-project "$PROJECT" \
      --wandb-entity "$ENTITY" \
      --wandb-name "$name" \
      --hf-repo-id "$HF_REPO" \
      --push-to-hub \
      --trust-remote-code \
      --shuffle-files
  ) >"$logfile" 2>&1 &
}

run_one 0 q30b-dflash-b16-a512-l2048-s10000-seed50 512 2048 10000 1e-4 50 1 16000 256
run_one 1 q30b-dflash-b16-a512-l2048-s10000-seed51 512 2048 10000 1e-4 51 1 16000 256
run_one 2 q30b-dflash-b16-a512-l2048-s10000-seed52 512 2048 10000 1e-4 52 1 16000 256
run_one 3 q30b-dflash-b16-a512-l2048-s10000-seed53 512 2048 10000 1e-4 53 1 16000 256
run_one 4 q30b-dflash-b16-a512-l4096-s6000-seed54 512 4096 6000 1e-4 54 1 8192 128
run_one 5 q30b-dflash-b16-a256-l4096-s6000-seed55 256 4096 6000 1e-4 55 1 8192 128
run_one 6 q30b-dflash-b16-a1024-l1024-s8000-seed56 1024 1024 8000 7e-5 56 1 8192 128
run_one 7 q30b-dflash2l-b16-a512-l1024-s8000-seed57 512 1024 8000 7e-5 57 2 8192 128

wait
