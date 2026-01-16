#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# support tp4/tp8 train eagle3 for Qwen3-30B-A3B

# export TOKENIZERS_PARALLELISM=false

NUM_GPUS=8
TP_SIZE=2

TARGET_MODEL_PATH=/disk3/wjp/pretrained_models/Qwen3-Coder-30B-A3B-Instruct
TRAIN_DATA_PATH=/disk3/wjp/datasets/repowiki/data_for_SpecForge_test.jsonl



# # Prepare hidden states
# export TORCH_NCCL_TIMEOUT_SEC=1800
# torchrun \
#     --standalone \
#     --nproc_per_node $NUM_GPUS \
#     scripts/prepare_hidden_states.py \
#     --target-model-path $TARGET_MODEL_PATH \
#     --enable-aux-hidden-states \
#     --data-path $TRAIN_DATA_PATH \
#     --chat-template repo-wiki \
#     --tp-size $TP_SIZE \
#     --batch-size 4 \
#     --max-length 65536 \
#     --output-path $ROOT_DIR/outputs/repo-wiki/train_hidden_states \
#     --sglang-mem-fraction-static 0.8



# offline training
BUILD_DATASET_NUM_PROC=1

LOR_INTERNAL=200
SAVE_INTERNAL=10

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path $TARGET_MODEL_PATH \
    --train-hidden-states-path $ROOT_DIR/outputs/repo-wiki/train_hidden_states \
    --draft-model-config $ROOT_DIR/configs/qwen3-30B-A3B-eagle3.json \
    --train-data-path $TRAIN_DATA_PATH \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/repo-wiki \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 65536 \
    --chat-template repo-wiki \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size 1 \
    --report-to tensorboard \
    --save-interval $LOR_INTERNAL \
    --log-interval $SAVE_INTERNAL \
    --sp-ring-size 2 \
    --sp-ulysses-size 4 \
    --attention-backend usp


# online training
# torchrun \
#     --standalone \
#     --nproc_per_node $NUM_GPUS \
#     $ROOT_DIR/scripts/train_eagle3.py \
#     --target-model-path $TARGET_MODEL_PATH \
#     --draft-model-config $ROOT_DIR/configs/qwen3-30B-A3B-eagle3.json \
#     --train-data-path $TRAIN_DATA_PATH \
#     --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
#     --output-dir $ROOT_DIR/outputs/repo-wiki \
#     --num-epochs 10 \
#     --batch-size 1 \
#     --learning-rate 1e-4 \
#     --max-length 32768 \
#     --chat-template repo-wiki \
#     --cache-dir $ROOT_DIR/cache \
#     --embedding-key model.embed_tokens.weight \
#     --tp-size 1 \
#     --report-to tensorboard \
#     --save-interval $LOR_INTERNAL \
#     --log-interval $SAVE_INTERNAL \
#     --sp-ring-size 2 \
#     --sp-ulysses-size 4 \
#     --attention-backend usp