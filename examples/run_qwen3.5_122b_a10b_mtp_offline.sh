#!/usr/bin/env bash
# Train the native MTP draft head for Qwen3.5-122B-A10B with online data collection.
#
# Differences vs. run_qwen3.5_122b_a10b_eagle3_online.sh:
#   * draft model uses configs/qwen3.5-122b-a10b-mtp.json (Qwen3MoeForCausalLMMTP-style)
#   * --load-mtp-weights / --mtp-layer-idx 0  ->  init draft weights from target's
#     native MTP block (incl. lm_head + shared embed_tokens + MoE + attn norms)
#   * --ttt-length 1                          ->  match target MTP's single-step regime
#   * target path points to the latest tidal-alsh01 snapshot
#   * separate --output-dir / --wandb-name to avoid clashing with the eagle3 run

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export https_proxy=10.140.24.177:3128

# TP_SIZE=8
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 scripts/prepare_hidden_states.py \
#     --target-model-path /mnt/tidal-alsh01/dataset/xiaowen/model/qwen35-122b \
#     --enable-aux-hidden-states \
#     --aux-hidden-states-layers 47 \
#     --data-path /mnt/tidal-alsh01/dataset/xiaowen/data/w1w/online/train_openai_chat.jsonl \
#     --output-path /mnt/tidal-alsh-share2/dataset/xiaowen/data/w1w/online/qwen3.5-122b-a10b-w1w-onv8 \
#     --max-length 20480 \
#     --chat-template qwen3.5 \
#     --batch-size 2 \
#     --tp-size 8 \
#     --build-dataset-num-proc 32 \
#     --sglang-mem-fraction-static 0.6 \
#     --dist-timeout 10000

# export HF_DATASETS_CACHE=$ROOT_DIR/cache/hf_datasets

NUM_GPUS=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path /mnt/tidal-alsh01/dataset/xiaowen/model/qwen35-122b \
    --draft-model-config $ROOT_DIR/configs/qwen3.5-122b-a10b-mtp.json \
    --load-mtp-weights \
    --mtp-layer-idx 0 \
    --train-data-path /mnt/tidal-alsh01/dataset/xiaowen/data/w1w/online/train_openai_chat.jsonl  \
    --train-hidden-states-path /mnt/tidal-alsh-share2/dataset/xiaowen/data/w1w/online/qwen3.5-122b-a10b-w1w-onv8 \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/qwen3.5-122b-a10b-mtp-offline-lr-2e-6-ttt-4-onv8 \
    --num-epochs 5 \
    --batch-size 1 \
    --draft-accumulation-steps 8 \
    --tp-size 1 \
    --learning-rate 2e-6 \
    --max-length 20480 \
    --ttt-length 4 \
    --chat-template qwen3.5 \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key "model.language_model.embed_tokens.weight" \
    --sglang-mem-fraction-static 0.4 \
    --save-interval 500 \
    --report-to wandb \
    --wandb-project "your_project" \
    --wandb-name "qwen3.5-122b-mtp-offline-lr-2e-6-ttt-4-onv8" \
    --target-micro-batch-size 1 \
    --warmup-ratio 0.06 \
    --logits-chunk-size 2048


    # Resume example: append --resume --ckpt-dir <path>; build_draft_model() will
    # detect lm_head_frozen from epoch_*/training_state.pt and re-copy lm_head only
    # via load_mtp_weights(only_lm_head=True), preserving trained MoE/attn weights.