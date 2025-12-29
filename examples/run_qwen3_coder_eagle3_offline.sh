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
    --target-model-path Qwen/Qwen3-Coder-480B-A35B-Instruct \
    --draft-model-config $ROOT_DIR/configs/qwen3-coder-480B-A35B-instruct-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/opc.jsonl \
    --train-hidden-states-path $ROOT_DIR/cache/hidden_states \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/Qwen3-Coder-480B-A35B-Instruct \
    --num-epochs 10 \
    --draft-micro-batch-size 1 \
    --draft-global-batch-size $TP_SIZE \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template qwen \
    --target-model-backend sglang
