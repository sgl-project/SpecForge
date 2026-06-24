#!/bin/bash
# Train the DFlash draft model for Qwen/Qwen3.6-27B (z-lab/Qwen3.6-27B-DFlash)
# on the full NVIDIA Nemotron-Post-Training-Dataset-v2 (stem+chat+math+code),
# online, reporting to Weights & Biases.
#
# Recipe is identical to run_qwen3.6_27b_dflash_online.sh; the only differences:
#   - logs to wandb instead of tensorboard (W&B key read from ~/.netrc)
#   - SPECFORGE_DATA_NUM_PROC defaults to 64 so the launch reuses the
#     pre-tokenized cache built with 64 shards (no 8-rank re-tokenize race).
#
# The "hf" target backend loads Qwen3_5ForCausalLM via AutoModelForCausalLM and
# reads output_hidden_states at target layers [1, 16, 31, 46, 61].

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=${SPECFORGE_DATA_NUM_PROC:-64}
# Pin to the locally-cached model snapshot (avoids re-resolving a drifted hub HEAD).
export HF_HOME=${HF_HOME:-/cluster-storage/models}
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
    --dataloader-num-workers 0 \
    --draft-config-path $ROOT_DIR/configs/qwen3.6-27b-dflash.json \
    --embedding-key model.language_model.embed_tokens.weight \
    --lm-head-key lm_head.weight \
    --mask-token-id 248070 \
    --train-data-path $ROOT_DIR/cache/dataset/nemotron-post-training-v2/nemotron_v2_train.jsonl \
    --eval-data-path $ROOT_DIR/cache/dataset/nemotron-post-training-v2/nemotron_v2_eval_2k.jsonl \
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
    --save-interval 10000 \
    --eval-interval 10000 \
    --report-to wandb \
    --wandb-project specforge-dflash \
    --wandb-name qwen3.6-27b-dflash-nemotron-v2 \
    --resume
