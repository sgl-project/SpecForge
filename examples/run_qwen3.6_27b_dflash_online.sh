#!/bin/bash
# Train the DFlash draft model for Qwen/Qwen3.6-27B (z-lab/Qwen3.6-27B-DFlash)
# on the NVIDIA Nemotron-Post-Training-Dataset-v2, online.
#
# Qwen3.6-27B is a hybrid linear-attention multimodal model
# (Qwen3_5ForConditionalGeneration). The DFlash draft is a 5-layer Qwen3
# (4 sliding + 1 full attention) that consumes the concatenated hidden
# states captured at target layers [1, 16, 31, 46, 61].
#
# The "hf" target backend is used: it loads the target via
# AutoModelForCausalLM (-> Qwen3_5ForCausalLM) and reads output_hidden_states.
# This avoids the still-evolving SGLang DFlash capture path for this brand-new
# architecture. Switch to --target-model-backend sglang once SGLang's qwen3_5
# DFlash capture API stabilizes.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=${SPECFORGE_DATA_NUM_PROC:-32}
# The Qwen3.6-27B weights are already cached locally; pin to the cached
# snapshot so transformers does not re-resolve against a possibly-drifted hub
# HEAD (which 404s on the shard names of the cached revision). Unset these if
# you need to (re)download the model on a fresh machine.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

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
    --train-data-path $ROOT_DIR/cache/dataset/nemotron-post-training-v2/nemotron_v2_train.jsonl \
    --eval-data-path $ROOT_DIR/cache/dataset/nemotron-post-training-v2/nemotron_v2_eval.jsonl \
    --output-dir $ROOT_DIR/outputs/qwen3.6-27b-dflash-nemotron \
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
    --log-interval 50 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --report-to tensorboard \
    --resume
