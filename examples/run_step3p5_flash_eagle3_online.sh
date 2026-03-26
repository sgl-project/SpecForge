
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# train eagle3 for step-3.5-flash
NUM_GPUS=${1:-8}
TP_SIZE=${2:-4}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

# train eagle3 online
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path stepfun-ai/Step-3.5-Flash \
    --draft-model-config configs/step-3.5-flash-eagle3.json \
    --train-data-path cache/dataset/ultrachat_train_regen.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/step-3.5-flash-eagle3-ultrachat-regen-online \
    --tp-size $TP_SIZE \
    --sglang-ep-size $TP_SIZE \
    --target-model-backend sglang \
    --trust-remote-code \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 5e-5 \
    --max-length 4096 \
    --sglang-attention-backend fa3 \
    --chat-template step3.5 \
    --cache-dir $ROOT_DIR/cache \
    --dist-timeout 60 \
    --sglang-mem-fraction-static 0.75 \
    --report-to wandb \
    --wandb-project specforge-step3p5-flash \
    --wandb-name specforge-step3p5-flash-ultrachat-regen
