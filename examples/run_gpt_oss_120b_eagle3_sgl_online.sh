#!/bin/bash
export PERSIST_DIR=/root/.cache/user_artifacts
export MODEL_PATH="openai/gpt-oss-120b"
export DATASET_PATH=$PERSIST_DIR/dataset/
export CACHE_DIR=$PERSIST_DIR/gpt-oss-120b/cache/
export OUTPUT_DIR=$PERSIST_DIR/gpt-oss-120b/outputs/
export MAX_LENGTH=4096
export CHAT_TEMPLATE=gpt-oss-naive

hf download $MODEL_PATH
hf download Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1 --repo-type dataset
hf download mlabonne/open-perfectblend --repo-type dataset

python scripts/prepare_data.py --dataset perfectblend --output-path $DATASET_PATH
python scripts/prepare_data.py --dataset magpie-qwen2.5-pro-1m-v0.1 --output-path $DATASET_PATH

python3 -m sglang.launch_server \
    --model openai/gpt-oss-120b \
    --cuda-graph-bs 1 2 4 8 16 32 64 128 256 512 \
    --dtype bfloat16 --mem-frac=0.8 --port 30001 &

python scripts/generate_data_by_target.py \
    --model-name openai/gpt-oss-120b \
    --raw-data-file $DATASET_PATH/perfectblend_train.jsonl \
    --output-dir $PERSIST_DIR/gpt-oss-120b-generated/perfectblend \
    --max-concurrency 512 \
    --num-per-shard 50000 \
    --server-address-port 127.0.0.1:30001

python scripts/generate_data_by_target.py \
    --model-name openai/gpt-oss-120b \
    --raw-data-file $DATASET_PATH/magpie-qwen2.5-pro-1m-v0.1_train.jsonl \
    --output-dir $PERSIST_DIR/gpt-oss-120b-generated/magpie-qwen2.5-pro-1m-v0.1 \
    --max-concurrency 512 \
    --num-per-shard 50000 \
    --server-address-port 127.0.0.1:30001

pkill -9 sglang

export DATASET_PATH=$PERSIST_DIR/gpt-oss-120b/dataset/
hf download baseten-admin/gpt-oss120b-generated-perfectblend --repo-type dataset
hf download baseten-admin/gpt-oss120b-generated-magpie-1m-v0.1 --repo-type dataset
python scripts/prepare_data.py --dataset baseten-admin/gpt-oss120b-generated-perfectblend --output-path $DATASET_PATH --split-eval
python scripts/prepare_data.py --dataset baseten-admin/gpt-oss120b-generated-magpie-1m-v0.1 --output-path $DATASET_PATH --split-eval
cat $DATASET_PATH/perfectblend_train.jsonl $DATASET_PATH/magpie-1m-v0.1_train.jsonl > $DATASET_PATH/all_train.jsonl
cat  $DATASET_PATH/perfectblend_test.jsonl $DATASET_PATH/magpie-1m-v0.1_test.jsonl > $DATASET_PATH/all_test.jsonl

python scripts/build_eagle3_dataset_cache.py \
    --target-model-path $MODEL_PATH \
    --draft-model-config ./configs/gpt-oss-120B-eagle3.json \
    --train-data-path $DATASET_PATH/all_train.jsonl \
    --eval-data-path $DATASET_PATH/all_test.jsonl \
    --cache-dir $CACHE_DIR \
    --chat-template $CHAT_TEMPLATE \
    --max-length $MAX_LENGTH \
    --view-train-data 1 --debug

python scripts/build_eagle3_dataset_cache.py \
    --target-model-path $MODEL_PATH \
    --draft-model-config ./configs/gpt-oss-120B-eagle3.json \
    --train-data-path $DATASET_PATH/all_train.jsonl \
    --eval-data-path $DATASET_PATH/all_test.jsonl \
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
    --draft-model-config ./configs/gpt-oss-120B-eagle3.json \
    --train-data-path $DATASET_PATH/all_train.jsonl \
    --eval-data-path $DATASET_PATH/all_test.jsonl \
    --tp-size $NUM_GPUS \
    --output-dir $OUTPUT_DIR \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 7e-5 \
    --draft-attention-backend flex_attention \
    --draft-global-batch-size 32 \
    --max-length $MAX_LENGTH \
    --chat-template $CHAT_TEMPLATE \
    --cache-dir $CACHE_DIR \
    --mem-frac=0.4 \
    --total-steps=800000 \
    --warmup-ratio=0.015 \
    --dist-timeout=10 \
    --save-interval 40000 \
    --eval-interval 40000 \
    --resume \
    --wandb-project gpt-oss-120b-eagle3 \
    --wandb-name sgl-online-continue \
    --report-to wandb

config_list=(
    "8,3,1,4"
    "8,5,4,8"
)
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 benchmarks/bench_model_speedup.py \
    --model-path openai/gpt-oss-120b \
    --speculative-draft-model-path zhuyksir/EAGLE3-gpt-oss-120b-bf16 \
    --port 20001 \
    --trust-remote-code \
    --mem-fraction-static 0.9 \
    --config-list "${config_list[@]}" \
    --attention-backend fa3 \
    --benchmark-list mtbench:80 \
    --output gpt-oss-120b_Eagle3-300k_result.jsonl --split-category
