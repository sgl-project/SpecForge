#!/bin/bash
# Disaggregated mode: DFlash training
# Target model: Qwen3-8B  |  Draft model: DFlash (5 layers, block_size=16)
# GPUs 0,2 → RolloutWorkers (target model, rollout TP=1, 2 independent rollout groups)
# GPU 3    → TrainWorker   (draft model, DP=1)
#
# Ray manages GPU isolation via placement groups.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export CUDA_VISIBLE_DEVICES=0,2,3
export HF_DATASETS_CACHE=$ROOT_DIR/cache/hf_datasets

BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

python "$ROOT_DIR/scripts/train_eagle3_ray.py" \
    --method dflash \
    --target-model-path Qwen/Qwen3-8B \
    --draft-model-config $ROOT_DIR/configs/qwen3-8b-dflash.json \
    --train-data-path $ROOT_DIR/cache/dataset/ultrachat_train.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/qwen3-8b-dflash-disaggregated \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --lm-head-key lm_head.weight \
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
    --block-size 16 \
    --num-anchors 512 \
    --seed 0
