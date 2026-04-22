#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# support tp8 train eagle3 for Qwen3-4B/8B/32B up to tp_size = 8
# export CUDA_VISIBLE_DEVICES=1,2,3,4
NUM_GPUS=${1:-1}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online.py \
    --target-model-path /mnt/nj-larc/dataset/xiaowen/model/Qwen3-8b-planning-tool-v6/checkpoint-302 \
    --draft-model-config $ROOT_DIR/configs/qwen3-8b-eagle3.json \
    --train-data-path /mnt/nj-larc/dataset/xiaowen/data/eagle/tool.v6_qwen3_openai_infer.processed_eval_100_convert.jsonl \
    --output-dir /mnt/nj-larc/dataset/xiaowen/model/Qwen3-8B-eagle3-test \
    --num-epochs 1 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template qwen \
    --cache-dir /mnt/nj-larc/dataset/xiaowen/data/eagle/planning/cache_test \
    --embedding-key model.embed_tokens.weight \
    --tp-size $NUM_GPUS \
    --ttt-length 7 \
    --draft-init-ckpt-path /mnt/nj-larc/dataset/xiaowen/model/Qwen3-8B-eagle3-test/epoch_9/
