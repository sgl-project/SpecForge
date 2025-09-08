SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)


# regenerate eagle3 train data
NUM_GPUS=${1:-8}

python3 \
    $ROOT_DIR/scripts/regenerate_data.py \
    --model Qwen/QwQ-32B \
    --input-file-path $ROOT_DIR/cache/dataset/sharegpt.jsonl \
    --output-file-path $ROOT_DIR/cache/dataset/sharegpt_regenerate.jsonl \
    --batch-size 128 \
    --tp-size $NUM_GPUS \
    --num-samples 1000 \
    --port 30000 \
    --temperature 0 \
    --mem-fraction-static 0.85 \
    --auto-launch-server
