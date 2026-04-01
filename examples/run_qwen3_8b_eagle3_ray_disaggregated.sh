#!/bin/bash
# Disaggregated mode: 2 GPUs for training, 2 GPUs for inference (TP=1)
# Target model: Qwen3-8B  |  Draft model: Eagle3
# GPUs 4,5 → RolloutWorkers (target model, rollout TP=1, 2 independent rollout groups)
# GPUs 6,7 → TrainWorkers  (draft model, DP=2 TP=1 SP=1)

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
# All 4 GPUs visible; Ray placement groups isolate rollout (0,1) from train (2,3)
# within the CUDA_VISIBLE_DEVICES namespace.
export CUDA_VISIBLE_DEVICES=0,5,6,7
export HF_DATASETS_CACHE=$ROOT_DIR/cache/hf_datasets

BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

python "$ROOT_DIR/scripts/train_eagle3_ray.py" \
    --target-model-path Qwen/Qwen3-8B \
    --draft-model-config $ROOT_DIR/configs/qwen3-8b-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/ultrachat_train.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/qwen3-8b-eagle3-disaggregated \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --attention-backend flex_attention \
    --target-model-backend sglang \
    --sglang-mem-fraction-static 0.5 \
    --sglang-enable-torch-compile \
    --log-interval 50 \
    --save-interval 5000 \
    --disaggregate \
    --rollout-num-gpus 2 \
    --train-num-gpus 2 \
    --rollout-tp-size 1 \
    --train-tp-size 1 \
    --train-sp-ulysses-size 1 \
    --train-sp-ring-size 1 \
    --seed 0
