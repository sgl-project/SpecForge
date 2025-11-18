#!/bin/bash
export PERSIST_DIR=/tmp # Please Change this to your own directory
export MODEL_PATH=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
export MODEL_NAME=Qwen3-235B-A22B-Instruct-2507-FP8

export RAW_DATASET_PATH=$PERSIST_DIR/dataset/
export GENERATED_DATASET_PATH=$PERSIST_DIR/$MODEL_NAME/generated_dataset
export CACHE_DIR=$PERSIST_DIR/$MODEL_NAME/cache
export OUTPUT_DIR=$PERSIST_DIR/$MODEL_NAME/outputs/
export CHAT_TEMPLATE=qwen
export MAX_LENGTH=4096

hf download $MODEL_PATH
hf download Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1 --repo-type dataset
hf download mlabonne/open-perfectblend --repo-type dataset

python scripts/prepare_data.py --dataset perfectblend --output-path $RAW_DATASET_PATH
python scripts/prepare_data.py --dataset magpie-qwen2.5-pro-1m-v0.1 --output-path $RAW_DATASET_PATH

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server \
    --model $MODEL_PATH \
    --cuda-graph-bs 1 2 4 8 16 32 64 128 256 \
    --mem-frac=0.8 --port 30001 --tp 4

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m sglang.launch_server \
    --model $MODEL_PATH \
    --cuda-graph-bs 1 2 4 8 16 32 64 128 256 \
    --mem-frac=0.8 --port 30002 --tp 4

python scripts/generate_data_by_target.py \
    --model-name $MODEL_PATH \
    --raw-data-file $RAW_DATASET_PATH/perfectblend_train.jsonl \
    --output-dir $GENERATED_DATASET_PATH/perfectblend \
    --max-concurrency 256 \
    --num-per-shard 50000 \
    --max-tokens $MAX_LENGTH \
    --server-address-port 127.0.0.1:30001 127.0.0.1:30002

python scripts/generate_data_by_target.py \
    --model-name $MODEL_PATH \
    --raw-data-file $RAW_DATASET_PATH/magpie-qwen2.5-pro-1m-v0.1_train.jsonl \
    --output-dir $GENERATED_DATASET_PATH/magpie-qwen2.5-pro-1m-v0.1 \
    --max-concurrency 256 \
    --num-per-shard 50000 \
    --max-tokens $MAX_LENGTH \
    --server-address-port 127.0.0.1:30001 127.0.0.1:30002

pkill -9 sglang

hf repo create zhuyksir/perfectblend-Qwen3-235B-A22B-Instruct-2507-FP8-generated --repo-type dataset
hf upload zhuyksir/perfectblend-Qwen3-235B-A22B-Instruct-2507-FP8-generated \
    $GENERATED_DATASET_PATH/perfectblend \
    --commit-message "generated dataset by $MODEL_PATH" \
    --repo-type dataset

hf repo create zhuyksir/magpie-qwen2.5-pro-1m-v0.1-Qwen3-235B-A22B-Instruct-2507-FP8-generated --repo-type dataset
hf upload zhuyksir/magpie-qwen2.5-pro-1m-v0.1-Qwen3-235B-A22B-Instruct-2507-FP8-generated \
    $GENERATED_DATASET_PATH/magpie-qwen2.5-pro-1m-v0.1 \
    --commit-message "generated dataset by $MODEL_PATH" \
    --repo-type dataset

# download the generated dataset and save them to train.jsonl and test.jsonl

export DRAFT_MODEL_CONFIG=./configs/qwen3-235B-A22B-eagle3.json
python scripts/build_eagle3_dataset_cache.py \
    --target-model-path $MODEL_PATH \
    --draft-model-config $DRAFT_MODEL_CONFIG \
    --train-data-path $DATASET_PATH/train.jsonl \
    --eval-data-path $DATASET_PATH/test.jsonl \
    --cache-dir $CACHE_DIR \
    --chat-template $CHAT_TEMPLATE \
    --max-length $MAX_LENGTH \
    --view-train-data 1 --debug

python scripts/build_eagle3_dataset_cache.py \
    --target-model-path $MODEL_PATH \
    --draft-model-config $DRAFT_MODEL_CONFIG \
    --train-data-path $DATASET_PATH/train.jsonl \
    --eval-data-path $DATASET_PATH/test.jsonl \
    --cache-dir $CACHE_DIR \
    --chat-template $CHAT_TEMPLATE \
    --max-length $MAX_LENGTH \
    --view-train-data 1

export NUM_GPUS=8
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    scripts/train_eagle3_sgl_online.py \
    --target-model-path $MODEL_PATH \
    --model-path $MODEL_PATH \
    --draft-model-config $DRAFT_MODEL_CONFIG \
    --train-data-path $DATASET_PATH/train.jsonl \
    --eval-data-path $DATASET_PATH/test.jsonl \
    --tp-size $NUM_GPUS \
    --output-dir $OUTPUT_DIR \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 5e-5 \
    --draft-attention-backend flex_attention \
    --draft-global-batch-size 16 \
    --max-length $MAX_LENGTH \
    --chat-template $CHAT_TEMPLATE \
    --cache-dir $CACHE_DIR \
    --mem-frac=0.4 \
    --total-steps=800000 \
    --warmup-ratio=0.015 \
    --dist-timeout=10 \
    --resume \
    --wandb-project qwen3-235b-a22b-eagle3 \
    --wandb-name sgl-online-continue \
    --report-to wandb
