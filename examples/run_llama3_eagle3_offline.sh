SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# train eagle3 for llama3.1-8b
NUM_GPUS=${1:-8}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config $ROOT_DIR/configs/llama3-8B-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt.jsonl \
    --train-hidden-states-path $ROOT_DIR/cache/hidden_states/sharegpt \
    --output-dir $ROOT_DIR/outputs/llama3-8b-eagle3 \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template llama3 \
    --cache-dir $ROOT_DIR/cache


NUM_GPUS=8
ROOT_DIR=$(pwd)
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_offline.py \
    --target-model-path /root/huggingface_cache/Llama-3.1-8B-Instruct \
    --draft-model-config $ROOT_DIR/configs/llama3-8B-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt.jsonl \
    --train-hidden-states-path /root/huggingface_cache/Llama-3.1-8B-Instruct-sharegpt-eagle3 \
    --output-dir $ROOT_DIR/outputs/llama3-8b-eagle3 \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template llama3 \
    --cache-dir $ROOT_DIR/cache \
    --wandb \
    --wandb-project eagle3-llama31-8b \
    --wandb-name sgl-spec-llama3-fsdp
