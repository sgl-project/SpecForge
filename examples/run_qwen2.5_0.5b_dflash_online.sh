#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# Train a DFlash draft for Qwen2.5-0.5B-Instruct with online data collection.
#
# Small-model parity check for the DataFlow runtime: DataFlow colocated and
# disaggregated are bit-exact with each other and with main on this recipe. The
# comparison curve (100 steps) is in examples/assets/qwen2.5-0.5b-dflash-vs-main.png.
# The disaggregated variant is examples/disagg/run_disagg_dflash.py.

NUM_GPUS=${1:-1}
ATTENTION_BACKEND=${2:-flex_attention}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path Qwen/Qwen2.5-0.5B-Instruct \
    --target-model-backend hf \
    --draft-config-path $ROOT_DIR/configs/qwen2.5-0.5b-dflash.json \
    --mask-token-id 151669 \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --output-dir $ROOT_DIR/outputs/qwen2.5-0.5b-dflash-sharegpt \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 512 \
    --chat-template qwen \
    --attention-backend $ATTENTION_BACKEND \
    --block-size 16 \
    --num-anchors 512 \
    --loss-decay-gamma 7.0 \
    --log-interval 10
