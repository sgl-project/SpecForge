#!/bin/bash
# Colocated mode: inference and training share the same 4 GPUs (4,5,6,7)
# Target model: Qwen3-8B  |  Draft model: Eagle3
# All 4 GPUs run both rollout (SGLang target model) and training (FSDP draft model).
# Parallelism: DP=4, TP=1, SP=1  (pure data-parallel)

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export CUDA_VISIBLE_DEVICES=4,5,6,7

BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

python "$ROOT_DIR/scripts/train_eagle3_ray.py" \
    --target-model-path Qwen/Qwen3-8B \
    --draft-model-config "$ROOT_DIR/configs/qwen3-8b-eagle3.json" \
    --train-data-path "$ROOT_DIR/cache/dataset/sharegpt_train.jsonl" \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir "$ROOT_DIR/outputs/qwen3-8b-eagle3-colocated" \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template qwen \
    --cache-dir "$ROOT_DIR/cache" \
    --embedding-key model.embed_tokens.weight \
    --attention-backend flex_attention \
    --target-model-backend sglang \
    --sglang-mem-fraction-static 0.4 \
    --sglang-enable-torch-compile \
    --log-interval 50 \
    --eval-interval 5000 \
    --save-interval 5000 \
    --train-num-gpus 4 \
    --rollout-tp-size 1 \
    --train-tp-size 1 \
    --train-sp-ulysses-size 1 \
    --train-sp-ring-size 1 \
    --seed 0
