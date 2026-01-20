SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# train eagle3 for gemma3-1b
NUM_GPUS=${1:-1}
TP_SIZE=${2:-1}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path google/gemma-3-1b-it \
    --draft-model-config $ROOT_DIR/configs/gemma3-1b-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --output-dir $ROOT_DIR/outputs/gemma3-1b-eagle3-sharegpt \
    --num-epochs 10 \
    --batch-size 1 \
    --tp-size $TP_SIZE \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template gemma \
    --cache-dir $ROOT_DIR/cache \
    --attention-backend sdpa \
    --target-model-backend hf \
    --log-interval 10
