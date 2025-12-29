SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# train eagle3 for GPT-OSS-120B
NUM_GPUS=${1:-8}
TP_SIZE=${2:-8}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path openai/gpt-oss-120b \
    --draft-model-config $ROOT_DIR/configs/gpt-oss-20B-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/perfect-blend-gptoss-20B.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/gpt-oss-20b-eagle3 \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template gpt-oss \
    --cache-dir $ROOT_DIR/cache \
    --dist-timeout 60
