#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

export HF_DATASETS_CACHE=$ROOT_DIR/cache/hf_datasets
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=64

ATTENTION_BACKEND=${2:-flex_attention}
NUM_GPUS=${1:-8}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path PATH/TO/Qwen3.5-4B \
    --draft-config-path $ROOT_DIR/configs/qwen3.5-4b-dflash.json \
    --train-data-path $ROOT_DIR/cache/dataset/train_regen.jsonl \
    --output-dir $ROOT_DIR/outputs/qwen3.5-4b-dflash \
    --num-epochs 10 \
    --batch-size 2 \
    --accumulation-steps 4 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 3072 \
    --chat-template qwen3.5 \
    --attention-backend $ATTENTION_BACKEND \
    --num-anchors 512 \
    --loss-decay-gamma 7.0 \
    --log-interval 50 \
    --save-interval 10000 \
    --report-to tensorboard \
    --target-model-backend hf \
    --block-size 16 \
    --trust-remote-code
