#!/bin/bash
# DSpark online training for Qwen3-8B.
#
# DSpark = DFlash block-diffusion drafter + EAGLE-style Markov & confidence heads,
# trained with cross-entropy + L1 distribution distillation + confidence BCE.
# The L1 / confidence losses need the target model's FINAL hidden state, so the
# target backend must surface it. The 'hf' backend (default below) always does;
# the 'sglang' backend does when its runner returns both the captured aux stream
# and the final hidden state. To train CE-only (no target final hidden state),
# pass: --l1-loss-alpha 0 --no-confidence-head --ce-loss-alpha 1.0

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=32
NUM_GPUS=${1:-8}

ATTENTION_BACKEND=${2:-flex_attention}
TARGET_BACKEND=${3:-hf}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dspark.py \
    --target-model-path Qwen/Qwen3-8B \
    --draft-config-path $ROOT_DIR/configs/qwen3-8b-dspark.json \
    --train-data-path $ROOT_DIR/cache/dataset/perfectblend_qwen3-8b_regen.jsonl \
    --output-dir $ROOT_DIR/outputs/qwen3-8b-dspark \
    --num-epochs 6 \
    --batch-size 4 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 3072 \
    --chat-template qwen \
    --attention-backend $ATTENTION_BACKEND \
    --loss-decay-gamma 4.0 \
    --log-interval 50 \
    --save-interval 1000 \
    --report-to wandb \
    --wandb-project specforge-qwen3-8b-dspark \
    --target-model-backend $TARGET_BACKEND \
    --block-size 16 \
    --num-anchors 512 \
    --markov-rank 256 \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --ce-loss-alpha 0.1 \
    --l1-loss-alpha 0.9 \
    --confidence-head-alpha 1.0 \
    --wandb-name qwen3-8b-dspark-perfectblend
