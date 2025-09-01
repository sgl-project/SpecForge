SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# train eagle3 for GPT-OSS-20B
NUM_GPUS=${1:-8}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online.py \
    --target-model-path openai/gpt-oss-20b \
    --draft-model-config $ROOT_DIR/configs/gpt-oss-20B-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/perfect-blend-gptoss-20B.jsonl \
    --output-dir $ROOT_DIR/outputs/perfect-blend-gptoss-20b-eagle3 \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template gpt-oss \
    --cache-dir $ROOT_DIR/cache \
    --dist-timeout 60


# --train-data-path $ROOT_DIR/cache/dataset/perfect-blend-gptoss-20B.jsonl \
