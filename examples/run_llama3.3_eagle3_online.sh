SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# train eagle3 for llama3.1-8b
NUM_GPUS=${1:-8}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online.py \
    --target-model-path meta-llama/Llama-3.3-70B-Instruct \
    --draft-model-config $ROOT_DIR/configs/llama3-8B-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt.jsonl \
    --output-dir $ROOT_DIR/outputs/llama3.3-70b-eagle3 \
    --num-epochs 10 \
    --batch-size 1 \
    --tp-size 8 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template llama3 \
    --cache-dir $ROOT_DIR/cache
