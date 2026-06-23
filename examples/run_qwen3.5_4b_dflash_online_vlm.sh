#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$ROOT_DIR/cache/hf_datasets}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/cache/compiled_kernels}
export SPECFORGE_DATA_NUM_PROC=${SPECFORGE_DATA_NUM_PROC:-64}

NUM_GPUS=${1:-8}
ATTENTION_BACKEND=${2:-flex_attention}

TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-PATH/TO/Qwen3.5-4B-VL}
DRAFT_CONFIG_PATH=${DRAFT_CONFIG_PATH:-$ROOT_DIR/configs/qwen3.5-4b-dflash.json}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-$ROOT_DIR/cache/dataset/vlm_train.jsonl}
IMAGE_ROOT=${IMAGE_ROOT:-$ROOT_DIR/cache/dataset/images}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/outputs/qwen3.5-4b-dflash-vlm}

DEVICE_TYPE=${DEVICE_TYPE:-cuda}   # set DEVICE_TYPE=npu for Ascend
DIST_BACKEND=${DIST_BACKEND:-nccl} # set DIST_BACKEND=hccl for Ascend
DTYPE=${DTYPE:-bfloat16}

EMBEDDING_KEY=${EMBEDDING_KEY:-model.language_model.embed_tokens.weight}
LM_HEAD_KEY=${LM_HEAD_KEY:-lm_head.weight}

torchrun \
    --standalone \
    --nproc_per_node "$NUM_GPUS" \
    "$ROOT_DIR/scripts/train_dflash_vlm.py" \
    --target-model-path "$TARGET_MODEL_PATH" \
    --draft-config-path "$DRAFT_CONFIG_PATH" \
    --train-data-path "$TRAIN_DATA_PATH" \
    --image-root "$IMAGE_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --num-epochs 10 \
    --batch-size 1 \
    --accumulation-steps 4 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 3072 \
    --chat-template qwen2-vl \
    --attention-backend "$ATTENTION_BACKEND" \
    --num-anchors 512 \
    --loss-decay-gamma 7.0 \
    --log-interval 50 \
    --save-interval 10000 \
    --cache-dir "$ROOT_DIR/cache" \
    --target-model-backend hf \
    --block-size 16 \
    --dtype "$DTYPE" \
    --device-type "$DEVICE_TYPE" \
    --dist-backend "$DIST_BACKEND" \
    --embedding-key "$EMBEDDING_KEY" \
    --lm-head-key "$LM_HEAD_KEY" \
    --min-pixels 50176 \
    --max-pixels 802816 \
    --report-to tensorboard \
    --trust-remote-code
