export PYTHONPATH=/Users/qibaoyuan/PycharmProjects/SpecForge-own:${PYTHONPATH}
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# train eagle3 for llama3.1-8b
NUM_GPUS=${1:-8}
NUM_GPUS=1

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online.py \
    --target-model-path /Users/qibaoyuan/Documents/llm/Meta-Llama-3.1-8B-Instruct  \
    --draft-model-config $ROOT_DIR/configs/llama3-8B-eagle3.json \
    --train-data-path /Users/qibaoyuan/PycharmProjects/SpecForge-own/examples/sharegpt-top100.jsonl \
    --output-dir $ROOT_DIR/outputs/llama3-8b-eagle3 \
    --num-epochs 5 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template llama3 \
    --cache-dir $ROOT_DIR/cache
