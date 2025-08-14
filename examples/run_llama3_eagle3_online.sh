SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# train eagle3 for llama3.1-8b
NUM_GPUS=${1:-8}

TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels \
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online.py \
    --target-model-path /shared/public/models/meta-llama/Meta-Llama-3.1-8B-Instruct \
    --draft-model-config $ROOT_DIR/configs/llama3-8B-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt.jsonl \
    --output-dir $ROOT_DIR/outputs/llama3-8b-eagle3 \
    --num-epochs 2 \
    --batch-size 2 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template llama3 \
    --cache-dir $ROOT_DIR/cache \
    # --logger-backend mlflow \
    # --mlflow-experiment $USER/llama3-8b-eagle3-specforge \
    # --mlflow-run-name $USER/llama3-8b-eagle3-specforge-run-$(date +%Y%m%d-%H%M%S) \
    # --mlflow-tracking-uri http://mlflow.grid1.ard.grid.linkedin.com:31812 \
    # --eval-data-split 0.01 \
    --attention-backend flex_attention
    --cache-dir $ROOT_DIR/cache \
    --wandb \
    --wandb-key "f3b46a484034ca1fe99fc5ae4d19402c94da12c1" \
    --wandb-project "specforge-training" \
    --wandb-name "llama3-8b-online-fixed-run-1"
