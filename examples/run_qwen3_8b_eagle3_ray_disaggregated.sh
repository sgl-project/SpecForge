#!/bin/bash
# Disaggregated mode: 2 GPUs for rollout (target model), 2 GPUs for training (draft model)
# Target model: Qwen3-8B  |  Draft model: Eagle3
# GPUs 0,1 → RolloutWorkers (target model, rollout TP=1, 2 independent rollout groups)
# GPUs 2,3 → TrainWorkers  (draft model, DP=2 TP=1 SP=1)
#
# Ray manages GPU isolation via placement groups.
# All 4 GPUs must be visible so Ray can allocate them to separate actors.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export CUDA_VISIBLE_DEVICES=0,2,4
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
    --sglang-mem-fraction-static 0.6 \
    --sglang-enable-torch-compile \
    --log-interval 50 \
    --eval-interval 5000 \
    --save-interval 5000 \
    --disaggregate \
    --rollout-num-gpus 2 \
    --train-num-gpus 1 \
    --rollout-tp-size 1 \
    --train-tp-size 1 \
    --train-sp-ulysses-size 1 \
    --train-sp-ring-size 1 \
    --rollout-batch-size 4 \
    --transfer-backend nccl \
    --seed 0 \
    --resume
