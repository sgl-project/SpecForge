SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# train eagle3 for DeepSeek-R1 offline
NUM_GPUS=${1:-8}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_offline.py \
    --target-model-path deepseek-ai/DeepSeek-R1-0528 \
    --draft-model-config $ROOT_DIR/configs/deepseek-r1-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt.jsonl \
    --train-hidden-states-path $ROOT_DIR/cache/hidden_states/ \
    --output-dir $ROOT_DIR/outputs/Deepseek-r1-eagle3 \
    --num-epochs 10 \
    --batch-size 2 \
    --learning-rate 1e-5 \
    --max-length 2048 \
    --chat-template deepseek_r1 \
    --cache-dir $ROOT_DIR/cache\
