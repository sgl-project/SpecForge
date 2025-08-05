SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# train eagle3 for llama3.1-8b
NUM_GPUS=${1:-8}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_offline.py \
    --target-model-path /root/models/Kimi-K2-Instruct \
    --draft-model-config $ROOT_DIR/configs/kimi-k2-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/test.jsonl \
    --train-hidden-states-path $ROOT_DIR/cache/hidden_states/rows_0-5000 \
    --output-dir $ROOT_DIR/outputs/Kimi-K2-eagle3 \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template kimi_k2 \
    --cache-dir $ROOT_DIR/cache