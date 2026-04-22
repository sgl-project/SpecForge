#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# support tp8 train eagle3 for Qwen3-4B/8B/32B up to tp_size = 8
export CUDA_VISIBLE_DEVICES=1,2,3,4
NUM_GPUS=${1:-1}
MODEL=/mnt/tidalfs-hssh01/dataset/xiaowen/model/plan_toolv6x0825/tool/v0-20250824-173634/checkpoint-302
DATASET=/mnt/tidalfs-hssh01/dataset/xiaowen/data/planning/tool.v6_qwen3_openai_infer.processed_eval_100_convert.jsonl
OUTPUT=/mnt/nj-larc/dataset/xiaowen/model/Qwen3-8B-eagle3-test

python3 \
    $ROOT_DIR/scripts/regenerate_train_data.py \
    --model $MODEL \
    --input-file-path $DATASET \
    --output-file-path $ROOT_DIR/cache/regenerate_train_data.jsonl \
    --batch-size 128 \
    --tp-size $NUM_GPUS \
    --port 30000 \
    --temperature 0 \
    --max-tokens 2048 \
    --mem-fraction-static 0.85 \
    --auto-launch-server

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path $MODEL \
    --draft-model-config $ROOT_DIR/configs/qwen3-8b-eagle3.json \
    --train-data-path $ROOT_DIR/cache/regenerate_train_data.jsonl \
    --output-dir $OUTPUT \
    --num-epochs 1 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template qwen \
    --cache-dir /mnt/tidalfs-hssh01/dataset/xiaowen/data/planning/cache_test \
    --embedding-key model.embed_tokens.weight \
    --tp-size $NUM_GPUS \
    --ttt-length 7 \
    --draft-init-ckpt-path /mnt/tidalfs-hssh01/dataset/xiaowen/model/Qwen3-8B-speculator.eagle3
