SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# train eagle3 for deepseek-v2-lite
NUM_GPUS=${1:-8}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online.py \
    --target-model-path DeepSeek-V2-Lite \
    --draft-model-config $ROOT_DIR/configs/deepseek-v2-lite-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt.jsonl \
    --output-dir $ROOT_DIR/outputs/deepseek-v2-lite-eagle3 \
    --num-epochs 10 \
    --batch-size 1 \
    --tp-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template deepseek \
    --cache-dir $ROOT_DIR/cache \
