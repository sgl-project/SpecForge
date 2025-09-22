#!/bin/bash
# export PERSIST_DIR=/data/yikai
export PERSIST_DIR=/root/.cache/user_artifacts
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

python3 -m sglang.launch_server \
    --model $MODEL_PATH \
    --cuda-graph-bs 1 2 4 8 16 32 64 128 \
    --mem-frac=0.8 --port 30001 --tp 8 --enable-ep-moe

python scripts/generate_data_by_target.py \
    --model-name $MODEL_PATH \
    --raw-data-file $RAW_DATASET_PATH/perfectblend_train.jsonl \
    --output-dir $GENERATED_DATASET_PATH/perfectblend \
    --max-concurrency 128 \
    --num-per-shard 50000 \
    --max-tokens $MAX_LENGTH \
    --server-address-port 127.0.0.1:30001

python scripts/generate_data_by_target.py \
    --model-name $MODEL_PATH \
    --raw-data-file $RAW_DATASET_PATH/magpie-qwen2.5-pro-1m-v0.1_train.jsonl \
    --output-dir $GENERATED_DATASET_PATH/magpie-qwen2.5-pro-1m-v0.1 \
    --max-concurrency 128 \
    --num-per-shard 50000 \
    --max-tokens $MAX_LENGTH \
    --server-address-port 127.0.0.1:30001

pkill -9 sglang

hf repo create zhuyksir/Ultrachat-Sharegpt-Qwen3-Next-80B-A3B-Instruct --repo-type dataset
hf upload zhuyksir/Ultrachat-Sharegpt-Qwen3-Next-80B-A3B-Instruct \
    $PERSIST_DIR/qwen3-next-80b-a3b-instruct-generated/ultrachat ultrachat \
    --commit-message "generated dataset by Qwen3-Next-80B-A3B-Instruct" \
    --repo-type dataset
hf upload zhuyksir/Ultrachat-Sharegpt-Qwen3-Next-80B-A3B-Instruct \
    $PERSIST_DIR/qwen3-next-80b-a3b-instruct-generated/sharegpt sharegpt \
    --commit-message "generated dataset by Qwen3-Next-80B-A3B-Instruct" \
    --repo-type dataset

export DATASET_PATH=$PERSIST_DIR/qwen3-next-80b-a3b-instruct/dataset/
# from datasets import load_dataset
# ds = load_dataset("zhuyksir/Ultrachat-Sharegpt-Qwen3-Next-80B-A3B-Instruct", split="train")
# dir_prefix = "/root/.cache/user_artifacts/qwen3-next-80b-a3b-instruct/dataset/"
# ds = ds.train_test_split(test_size=0.05)
# train_ds = ds["train"]
# test_ds = ds["test"]
# train_ds.to_json(dir_prefix + "train.jsonl", orient="records", lines=True)
# test_ds.to_json(dir_prefix + "test.jsonl", orient="records", lines=True)

python scripts/build_eagle3_dataset_cache.py \
    --target-model-path $MODEL_PATH \
    --draft-model-config ./configs/qwen3-next-80b-a3b-eagle3.json \
    --train-data-path $DATASET_PATH/train.jsonl \
    --eval-data-path $DATASET_PATH/test.jsonl \
    --cache-dir $CACHE_DIR \
    --chat-template $CHAT_TEMPLATE \
    --max-length $MAX_LENGTH \
    --view-train-data 1 --debug

python scripts/build_eagle3_dataset_cache.py \
    --target-model-path $MODEL_PATH \
    --draft-model-config ./configs/qwen3-next-80b-a3b-eagle3.json \
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
    --draft-model-config ./configs/qwen3-next-80b-a3b-eagle3.json \
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
    --wandb-project qwen3-next-80b-a3b-eagle3 \
    --wandb-name sgl-online-continue \
    --report-to wandb


python -m sglang.launch_server \
    --model-path Qwen/Qwen3-Next-80B-A3B-Instruct \
    --port 30000 \
    --tp-size 8 \
    --context-length 262144 \
    --mem-fraction-static 0.8 \
    --speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4

python3 benchmarks/bench_model_speedup.py \
    --model-path Qwen/Qwen3-Next-80B-A3B-Instruct \
    --port 30000 \
    --skip-launch-server \
    --benchmark-list mtbench:80 gsm8k:200 humaneval:200 math500:200 \
    --output qwen3-next-80b-a3b-instruct_Eagle3_result.jsonl

config_list=(
    "8,3,1,4"
    "8,3,4,4"
)
python3 benchmarks/bench_model_speedup.py \
    --model-path Qwen/Qwen3-Next-80B-A3B-Instruct \
    --speculative-draft-model-path $PERSIST_DIR/qwen3-next-80b-a3b-instruct/outputs/step_77468 \
    --port 20001 \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --config-list "${config_list[@]}" \
    --tp-size 8 \
    --benchmark-list mtbench:80 gsm8k:200 humaneval:200 math500:200 \
    --output qwen3-next-80b-a3b-instruct_Eagle3-300k_result.jsonl
