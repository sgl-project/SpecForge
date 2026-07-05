#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# Train a DFlash draft for Qwen3.6-27B, 6 epochs, colocated 8-GPU data-parallel
# (the target runs inference in-process; features stream to the draft trainer).
#
# This is the exact recipe behind the training curve in
#   examples/assets/qwen36-27b-dflash-nemotron-6ep.png
# (W&B project `qwen36-dflash-pr645`, run `qwen36-27b-dflash-nemotron-6ep`):
# train loss 9.03 -> 3.46, token accuracy 0.03 -> 0.25 over ~2900 steps.
#
# Prerequisites:
#   - Qwen/Qwen3.6-27B weights (cached locally or downloadable from HF).
#   - A chat dataset at $ROOT_DIR/cache/dataset/. The plotted run used a
#     4000-sample Nemotron-v2 post-training split (seed 42); prepare your own
#     data with scripts/prepare_data.py (see examples/README.md).
#   - For W&B logging, export your key first (never hard-code it):
#         export WANDB_API_KEY=<your key>
#     or pass --report-to none to disable tracking.

NUM_GPUS=${1:-8}
ATTENTION_BACKEND=${2:-flex_attention}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path Qwen/Qwen3.6-27B \
    --target-model-backend hf \
    --trust-remote-code \
    --draft-config-path $ROOT_DIR/configs/qwen3.6-27b-dflash.json \
    --embedding-key model.language_model.embed_tokens.weight \
    --lm-head-key lm_head.weight \
    --mask-token-id 248070 \
    --train-data-path $ROOT_DIR/cache/dataset/nemotron_v2_train.jsonl \
    --eval-data-path $ROOT_DIR/cache/dataset/nemotron_v2_eval.jsonl \
    --output-dir $ROOT_DIR/outputs/qwen3.6-27b-dflash-nemotron-6ep \
    --num-epochs 6 \
    --batch-size 1 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 4096 \
    --chat-template qwen3.5 \
    --attention-backend $ATTENTION_BACKEND \
    --block-size 16 \
    --num-anchors 512 \
    --loss-decay-gamma 7.0 \
    --log-interval 10 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --seed 42 \
    --report-to wandb \
    --wandb-project qwen36-dflash-pr645 \
    --wandb-name qwen36-27b-dflash-nemotron-6ep
