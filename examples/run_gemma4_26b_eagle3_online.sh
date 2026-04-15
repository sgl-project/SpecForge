SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# train eagle3 for gemma3-1b
NUM_GPUS=${1:-8}
TP_SIZE=${2:-2}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path google/gemma-4-26b-a4b-it \
    --draft-model-config $ROOT_DIR/configs/gemma4-26b-a4b-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/ultrachat_train.jsonl \
    --output-dir $ROOT_DIR/outputs/gemma4-26b-a4b-eagle3-ultrachat \
    --num-epochs 10 \
    --batch-size 4 \
    --tp-size $TP_SIZE \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template gemma-4 \
    --cache-dir $ROOT_DIR/cache \
    --attention-backend sdpa \
    --target-model-backend hf \
    --log-interval 500 \
    --eval-interval 2500 \
    --save-interval 10000 \
    --report-to tensorboard \
    --embedding-key=model.language_model.embed_tokens.weight \
    --eval-holdout-ratio 0.05
