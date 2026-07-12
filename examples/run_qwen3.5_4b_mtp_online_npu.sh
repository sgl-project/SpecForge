#!/bin/bash
#
# Qwen3.5-4B MTP online training on NPU (HF backend).
#
# NOTE: the text decoder in the Qwen3.5-4B VLM checkpoint is nested under
# model.language_model, so we explicitly pass --embedding-key and --lm-head-key.
# If your checkpoint uses a different key layout, adjust the two keys below.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

export HF_DATASETS_CACHE=$ROOT_DIR/cache/hf_datasets
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=64

ATTENTION_BACKEND=${2:-sdpa}
NUM_GPUS=${1:-8}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_mtp.py \
    --target-model-path PATH/TO/Qwen3.5-4B \
    --draft-config-path $ROOT_DIR/configs/qwen3.5-4b-mtp.json \
    --train-data-path $ROOT_DIR/cache/dataset/train_regen.jsonl \
    --output-dir $ROOT_DIR/outputs/qwen3.5-4b-mtp \
    --num-epochs 10 \
    --batch-size 2 \
    --accumulation-steps 4 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 3072 \
    --chat-template qwen3.5 \
    --attention-backend $ATTENTION_BACKEND \
    --ploss-decay 1.0 \
    --log-interval 50 \
    --save-interval 10000 \
    --report-to tensorboard \
    --target-model-backend hf \
    --trust-remote-code \
    --embedding-key model.language_model.embed_tokens.weight \
    --lm-head-key model.language_model.embed_tokens.weight \
    "${@:3}"
