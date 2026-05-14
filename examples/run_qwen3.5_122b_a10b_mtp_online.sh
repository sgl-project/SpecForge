#!/usr/bin/env bash
# Train the native MTP draft head for Qwen3.5-122B-A10B with online data collection.


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export https_proxy=10.140.24.177:3128

TP_SIZE=8
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

export HF_DATASETS_CACHE=$ROOT_DIR/cache/hf_datasets

NUM_GPUS=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path /mnt/tidal-alsh01/dataset/xiaowen/model/qwen35-122b \
    --draft-model-config $ROOT_DIR/configs/qwen3.5-122b-a10b-mtp.json \
    --load-mtp-weights \
    --mtp-layer-idx 0 \
    --draft-accumulation-steps 8 \
    --train-data-path /mnt/tidal-alsh01/dataset/xiaowen/data/w1w/online/train_openai_chat.jsonl  \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/qwen3.5-122b-a10b-mtp-online-onv8-lr-1e-6 \
    --num-epochs 5 \
    --batch-size 1 \
    --tp-size $TP_SIZE \
    --learning-rate 1e-6 \
    --max-length 20480 \
    --ttt-length 4 \
    --chat-template qwen3.5 \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key "model.language_model.embed_tokens.weight" \
    --sglang-mem-fraction-static 0.4 \
    --save-interval 500 \
    --report-to wandb \
    --wandb-project "your_project" \
    --wandb-name "qwen3.5-122b-mtp-online-onv8-lr-1e-6" \
    --target-micro-batch-size 1 \
    --logits-chunk-size 2048 \
    --warmup-ratio 0.03

    # Resume example: append --resume --ckpt-dir <path>; build_draft_model() will
    # detect lm_head_frozen from epoch_*/training_state.pt and re-copy lm_head only
    # via load_mtp_weights(only_lm_head=True), preserving trained MoE/attn weights.
