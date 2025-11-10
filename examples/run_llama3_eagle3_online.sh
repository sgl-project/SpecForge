#!/bin/bash
export PERSIST_DIR=/tmp # Please Change this to your own directory
export MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
export MODEL_NAME=Llama-3.1-8B-Instruct

export RAW_DATASET_PATH=$PERSIST_DIR/dataset/
export GENERATED_DATASET_PATH=$PERSIST_DIR/$MODEL_NAME/generated_dataset
export CACHE_DIR=$PERSIST_DIR/$MODEL_NAME/cache
export OUTPUT_DIR=$PERSIST_DIR/$MODEL_NAME/outputs/
export CHAT_TEMPLATE=llama3
export MAX_LENGTH=2048

hf download $MODEL_PATH
hf download Aeala/ShareGPT_Vicuna_unfiltered --repo-type dataset
hf download HuggingFaceH4/ultrachat_200k --repo-type dataset

# python scripts/prepare_data.py --dataset sharegpt --output-path $DATASET_PATH
# python scripts/prepare_data.py --dataset ultrachat --output-path $DATASET_PATH

# for i in {1..4}; do
#     CUDA_VISIBLE_DEVICES=$i python3 -m sglang.launch_server \
#         --model meta-llama/Llama-3.1-8B-Instruct \
#         --cuda-graph-bs 1 2 4 8 16 32 64 128 256 \
#         --context-length 8192 \
#         --dtype bfloat16 --mem-frac=0.8 --port $((30000 + i)) &
# done

# python scripts/generate_data_by_target.py \
#     --model-name meta-llama/Llama-3.1-8B-Instruct \
#     --raw-data-file $DATASET_PATH/sharegpt.jsonl \
#     --output-dir $DATASET_PATH/sharegpt-llama-3.1-8b-instruct \
#     --max-concurrency 256 \
#     --num-per-shard 10000 \
#     --server-address-port 127.0.0.1:30001 127.0.0.1:30002 127.0.0.1:30003 127.0.0.1:30004

hf download zhuyksir/Ultrachat-Sharegpt-Llama3.1-8B --repo-type dataset --local-dir $GENERATED_DATASET_PATH
shuf $GENERATED_DATASET_PATH/sharegpt_ultrachat.jsonl -o $GENERATED_DATASET_PATH/shuffled_sharegpt_ultrachat.jsonl
total=$(wc -l < $GENERATED_DATASET_PATH/shuffled_sharegpt_ultrachat.jsonl)
train_count=$((total * 95 / 100))
head -n $train_count $GENERATED_DATASET_PATH/shuffled_sharegpt_ultrachat.jsonl > $GENERATED_DATASET_PATH/train_data.jsonl
tail -n +$((train_count + 1)) $GENERATED_DATASET_PATH/shuffled_sharegpt_ultrachat.jsonl > $GENERATED_DATASET_PATH/eval_data.jsonl
python scripts/build_eagle3_dataset_cache.py \
    --target-model-path $MODEL_PATH \
    --draft-model-config ./configs/llama3-8B-eagle3.json \
    --train-data-path $GENERATED_DATASET_PATH/train_data.jsonl \
    --eval-data-path $GENERATED_DATASET_PATH/eval_data.jsonl \
    --cache-dir $CACHE_DIR \
    --chat-template $CHAT_TEMPLATE \
    --max-length $MAX_LENGTH \
    --view-train-data 1 --debug

python scripts/build_eagle3_dataset_cache.py \
    --target-model-path $MODEL_PATH \
    --draft-model-config ./configs/llama3-8B-eagle3.json \
    --train-data-path $GENERATED_DATASET_PATH/train_data.jsonl \
    --eval-data-path $GENERATED_DATASET_PATH/eval_data.jsonl \
    --cache-dir $CACHE_DIR \
    --chat-template $CHAT_TEMPLATE \
    --max-length $MAX_LENGTH

export NUM_GPUS=4
export OUTPUT_DIR=$PERSIST_DIR/$MODEL_NAME/draft_tp1_target_tp4_outputs/
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    scripts/train_eagle3_online.py \
    --model-path $MODEL_PATH \
    --draft-model-config ./configs/llama3-8B-eagle3.json \
    --train-data-path $GENERATED_DATASET_PATH/train_data.jsonl \
    --eval-data-path $GENERATED_DATASET_PATH/eval_data.jsonl \
    --target-tp-size 1 \
    --draft-tp-size 1 \
    --batch-size 16 \
    --draft-micro-batch-size 1 \
    --target-model-backend sglang \
    --output-dir $OUTPUT_DIR \
    --num-epochs 10 \
    --learning-rate 5e-5 \
    --draft-attention-backend flex_attention \
    --max-length $MAX_LENGTH \
    --chat-template $CHAT_TEMPLATE \
    --cache-dir $CACHE_DIR \
    --total-steps=800000 \
    --warmup-ratio=0.015 \
    --enable-zero2 \
    --resume \
    --wandb-project llama3-8b-eagle3 \
    --wandb-name dev_draft_tp1_target_tp4 \
    --report-to wandb

config_list=(
    "8,7,10,60"
    "8,3,1,4"
)
CUDA_VISIBLE_DEVICES=1,2 python3 benchmarks/bench_model_speedup.py \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --speculative-draft-model-path $OUTPUT_DIR/epoch_2/ \
    --port 20001 \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --config-list "${config_list[@]}" \
    --benchmark-list mtbench:80 \
    --output dev_result.jsonl --enable-multi-turn-conversation
