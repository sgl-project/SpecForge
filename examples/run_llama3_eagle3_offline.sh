#!/bin/bash
export MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
export DATASET_PATH=~/.cache/huggingface/Llama-3.1-8B-Instruct/dataset/
export CACHE_DIR=~/.cache/huggingface/Llama-3.1-8B-Instruct/cache/
export OUTPUT_DIR=~/.cache/huggingface/Llama-3.1-8B-Instruct/outputs/
export HIDDEN_STATES_DIR=~/.cache/huggingface/Llama-3.1-8B-Instruct/hidden_states/
export MAX_LENGTH=2048
export CHAT_TEMPLATE=llama3

hf download $MODEL_PATH
hf download Aeala/ShareGPT_Vicuna_unfiltered --repo-type dataset

python scripts/prepare_data.py --dataset sharegpt --output_path $DATASET_PATH --test-size 0.01
python scripts/build_eagle3_dataset_cache.py \
    --target-model-path $MODEL_PATH \
    --draft-model-config ./configs/llama3-8B-eagle3.json \
    --train-data-path $DATASET_PATH/sharegpt_train.jsonl \
    --eval-data-path $DATASET_PATH/sharegpt_test.jsonl \
    --cache-dir $CACHE_DIR \
    --chat-template $CHAT_TEMPLATE \
    --max-length $MAX_LENGTH \
    --view-train-data 1

CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 \
    scripts/prepare_hidden_states.py \
    --data-path $DATASET_PATH/sharegpt_test.jsonl \
    --model-path $MODEL_PATH \
    --cache-dir $CACHE_DIR \
    --output-dir $HIDDEN_STATES_DIR/sharegpt_test \
    --chat-template $CHAT_TEMPLATE \
    --max-length $MAX_LENGTH \
    --enable-aux-hidden-states \
    --tp-size 4 \
    --batch-size 4 \
    --mem-frac=0.75

CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 \
    scripts/prepare_hidden_states.py \
    --data-path $DATASET_PATH/sharegpt_train.jsonl \
    --model-path $MODEL_PATH \
    --cache-dir $CACHE_DIR \
    --output-dir $HIDDEN_STATES_DIR/sharegpt_train \
    --chat-template $CHAT_TEMPLATE \
    --max-length $MAX_LENGTH \
    --enable-aux-hidden-states \
    --tp-size 4 \
    --batch-size 4 \
    --mem-frac=0.75

# python scripts/view_data.py --data-path $HIDDEN_STATES_DIR/all_test/rows_0-5000/data_100.ckpt --tokenizer $MODEL_PATH
# python scripts/view_data.py --data-path $HIDDEN_STATES_DIR/all_train/rows_0-5000/data_100.ckpt --tokenizer $MODEL_PATH

export NUM_GPUS=4
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    scripts/train_eagle3_offline.py \
    --target-model-path $MODEL_PATH \
    --draft-model-config ./configs/llama3-8B-eagle3.json \
    --train-data-path $DATASET_PATH/sharegpt_train.jsonl \
    --train-hidden-states-path $HIDDEN_STATES_DIR/sharegpt_train/ \
    --eval-data-path $DATASET_PATH/sharegpt_test.jsonl \
    --eval-hidden-states-path $HIDDEN_STATES_DIR/sharegpt_test/ \
    --output-dir $OUTPUT_DIR \
    --num-epochs 10 \
    --draft-global-batch-size 16 \
    --draft-micro-batch-size 1 \
    --learning-rate 5e-5 \
    --draft-attention-backend flex_attention \
    --max-length $MAX_LENGTH \
    --chat-template $CHAT_TEMPLATE \
    --cache-dir $CACHE_DIR \
    --dist-timeout=10 \
    --log-steps 1 \
    --report-to wandb \
    --wandb-project llama3-8b-eagle3 \
    --wandb-name offline-100k-4gpus
