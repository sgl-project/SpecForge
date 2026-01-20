SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# train eagle3 for qwen3-coder
NUM_GPUS=${1:-8}
TP_SIZE=${2:-8}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
    --draft-model-config $ROOT_DIR/configs/qwen3-coder-480B-A35B-instruct-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/opc_regenerated.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/Qwen3-Coder-480B-A35B-Instruct-FP8 \
    --tp-size $TP_SIZE \
    --sglang-ep-size 2 \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-5 \
    --ttt-length 13 \
    --sglang-mem-fraction-static 0.6 \
    --max-length 2048 \
    --chat-template qwen \
    --target-model-backend sglang \
    --save-interval 20000 \
    --eval-interval 20000 \
    --report-to wandb \
    --wandb-project specforge-qwen3-480-coder-fp8 \
    --wandb-name qwen3-coder-480b-a35b-eagle3-tp8-ep2-opc-regen
