#!/bin/bash
# Faithful PR #593 recipe on the PR #645 base: Qwen3.6-27B DFlash, Nemotron-v2,
# 6 epochs, wandb tracking. COLOCATED 8-GPU data-parallel (train_dflash.py's
# native DistributedSampler + tracker) — the right engine for multi-epoch
# training (the DataFlow online-disagg path is consume-once + single-DP-rank).
#
# Secrets come from the environment, never hard-coded:
#   export WANDB_API_KEY=<your key>          # required for --report-to wandb
#   export HF_TOKEN=<token>                  # only if (re)downloading gated repos
# The Qwen3.6-27B weights + Nemotron-v2 data are assumed already cached/prepared
# (see setup36_1n.sh; prepare_nemotron_post_training_v2.py needs HF_TOKEN once).
set -euxo pipefail
E=${EXP_ROOT:-/root/exp36}

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export FLASHINFER_DISABLE_VERSION_CHECK=1
export TORCHINDUCTOR_CACHE_DIR=$E/cache/compiled_kernels
export WANDB_PROJECT=${WANDB_PROJECT:-qwen36-dflash-pr645}
export WANDB_NAME=${WANDB_NAME:-qwen36-27b-dflash-nemotron-6ep}
: "${WANDB_API_KEY:?set WANDB_API_KEY in the environment}"

NUM_GPUS=${NUM_GPUS:-8}

cd $E/sf
torchrun --standalone --nproc_per_node "$NUM_GPUS" \
    scripts/train_dflash.py \
    --target-model-path Qwen/Qwen3.6-27B \
    --target-model-backend hf --trust-remote-code \
    --draft-config-path configs/qwen3.6-27b-dflash.json \
    --embedding-key model.language_model.embed_tokens.weight \
    --lm-head-key lm_head.weight --mask-token-id 248070 \
    --train-data-path $E/data/nemotron_v2_train.jsonl \
    --eval-data-path $E/data/nemotron_v2_eval.jsonl \
    --output-dir $E/out/qwen36-6ep \
    --num-epochs 6 --batch-size 1 \
    --learning-rate 6e-4 --warmup-ratio 0.04 --max-grad-norm 1.0 \
    --max-length 4096 --chat-template qwen3.5 \
    --attention-backend flex_attention \
    --block-size 16 --num-anchors 512 --loss-decay-gamma 7.0 \
    --log-interval 10 --save-interval 2000 --eval-interval 2000 \
    --report-to wandb --seed 42
