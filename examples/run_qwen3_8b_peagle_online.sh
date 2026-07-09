#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
# peagle.py is not in the installed specforge package yet; prefer the repo source
export PYTHONPATH=$ROOT_DIR:${PYTHONPATH:-}
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# support tp8 train P-EAGLE for Qwen3-4B/8B/32B up to tp_size = 8
NUM_GPUS=${1:-1}
TP_SIZE=${2:-1}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-32}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_peagle.py \
    --target-model-path Qwen/Qwen3-8B \
    --draft-model-config $ROOT_DIR/configs/qwen3-8b-peagle.json \
    --train-data-path $ROOT_DIR/cache/dataset/perfectblend-qwen3-8b-regen.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/peagle_qwen3_8b \
    --num-epochs 20 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --warmup-ratio 0.0025 \
    --max-grad-norm 1 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --tp-size $TP_SIZE \
    --num-depths 5 \
    --down-sample-ratio 0.8 \
    --down-sample-ratio-min 0.2 \
    --num-draft-layers 4 \
    --no-norm-before-residual \
    --target-model-backend sglang \
    --save-interval 50000 \
    --eval-interval 50000 \
    --log-interval 50 \
    --report-to wandb \
    --dist-timeout 120
