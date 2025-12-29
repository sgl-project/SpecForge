SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
NUM_GPUS=${1:-1}
TP_SIZE=${2:-1}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

# generate hidden states
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    scripts/prepare_hidden_states.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-aux-hidden-states \
    --data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --output-path $ROOT_DIR/cache/hidden_states/sharegpt_train_Llama-3.1-8B-Instruct \
    --chat-template llama3 \
    --max-length 4096 \
    --tp-size $TP_SIZE \
    --batch-size 32

# train eagle3 offline
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config $ROOT_DIR/configs/llama3-8B-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --train-hidden-states-path $ROOT_DIR/cache/hidden_states/sharegpt_train_Llama-3.1-8B-Instruct \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/llama3-8b-eagle3-sharegpt-offline \
    --num-epochs 10 \
    --batch-size 1 \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template llama3 \
    --cache-dir $ROOT_DIR/cache
