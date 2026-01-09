#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# support tp4/tp8 train eagle3 for Qwen3-30B-A3B
NUM_GPUS=8
TP_SIZE=8
# BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}
BUILD_DATASET_NUM_PROC=24
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path /home/qspace/Qwen3_235B_SFT_0729_gcore_2_grpo_hf_ckpt \
    --draft-model-config $ROOT_DIR/configs/qwen3-235B-A22B-eagle3.json \
    --train-data-path /home/qspace/SpecForge/cache/dataset/xiaodian_eagle3_1224_formatted_conversation.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir /home/qspace/outputs/qwen3-235B-A22B-eagle3-1224.json \
    --num-epochs 2 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template qwen \
    --cache-dir /home/qspace/outputs/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $TP_SIZE \
    --sglang-mem-fraction-static 0.75 \
    --ttt-length 4 
    #--target-model-backend sglang
