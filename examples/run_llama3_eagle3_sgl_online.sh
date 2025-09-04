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
hf download HuggingFaceH4/ultrachat_200k --repo-type dataset

python scripts/prepare_data.py --dataset sharegpt --output_path $DATASET_PATH --test-size 0.0
python scripts/prepare_data.py --dataset ultrachat --output_path $DATASET_PATH --test-size 0.0

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

hf download zhuyksir/Ultrachat-Sharegpt-Llama3.1-8B --repo-type dataset --local-dir $DATASET_PATH
python scripts/build_eagle3_dataset_cache.py \
    --target-model-path $MODEL_PATH \
    --draft-model-config ./configs/llama3-8B-eagle3.json \
    --train-data-path $DATASET_PATH/sharegpt_ultrachat.jsonl \
    --cache-dir $CACHE_DIR \
    --chat-template $CHAT_TEMPLATE \
    --max-length $MAX_LENGTH \
    --view-train-data 1

export NUM_GPUS=4
export OUTPUT_DIR=~/.cache/huggingface/Llama-3.1-8B-Instruct/dev_outputs/
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    scripts/train_eagle3_sgl_online.py \
    --target-model-path $MODEL_PATH \
    --model-path $MODEL_PATH \
    --draft-model-config ./configs/llama3-8B-eagle3.json \
    --train-data-path $DATASET_PATH/sharegpt_ultrachat.jsonl \
    --tp-size $NUM_GPUS \
    --output-dir $OUTPUT_DIR \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 5e-5 \
    --draft-attention-backend flex_attention \
    --max-length $MAX_LENGTH \
    --chat-template $CHAT_TEMPLATE \
    --cache-dir $CACHE_DIR \
    --mem-frac=0.4 \
    --total-steps=800000 \
    --warmup-ratio=0.015 \
    --dist-timeout=10 \
    --wandb-project llama3-8b-eagle3-offline \
    --wandb-name sgl-online \
    --report-to wandb

config_list=(
    "8,7,10,60"
    "8,3,1,4"
)
CUDA_VISIBLE_DEVICES=1,2 python3 benchmarks/bench_model_speedup.py \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --speculative-draft-model-path ~/.cache/huggingface/Llama-3.1-8B-Instruct/dev_outputs/step_20533/ \
    --port 20001 \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --config-list "${config_list[@]}" \
    --benchmark-list mtbench:80 \
    --output Llama-3.1-8B-Instruct_Eagle3-300k_result.jsonl --enable-multi-turn-conversation
