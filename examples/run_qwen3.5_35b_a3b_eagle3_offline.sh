
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# train eagle3 for deepseek-v3
NUM_GPUS=4
# TP_SIZE=${2:-8}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

# generate hidden states
CUDA_VISIBLE_DEVICES=1,2,3,5 torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    scripts/prepare_hidden_states.py \
    --target-model-path /data/jiapingW/pretrained_models/Qwen3.5-35B-A3B \
    --enable-aux-hidden-states \
    --data-path $ROOT_DIR/cache/dataset/ultrachat_train.jsonl \
    --output-path $ROOT_DIR/cache/hidden_states/qwen3.5-35b-a3b-ultrachat \
    --chat-template qwen \
    --max-length 4096 \
    --tp-size 1 \
    --batch-size 4 \
    --sglang-mem-fraction-static 0.7


# NUM_GPUS=2
# CUDA_VISIBLE_DEVICES=6,7 torchrun \
#     --standalone \
#     --nproc_per_node $NUM_GPUS \
#     $ROOT_DIR/scripts/train_eagle3.py \
#     --target-model-path /data/jiapingW/pretrained_models/Qwen3.5-35B-A3B \
#     --draft-model-config $ROOT_DIR/configs/qwen3.5-35b-a3b-eagle3.json \
#     --train-data-path $ROOT_DIR/cache/dataset/ultrachat_train.jsonl  \
#     --train-hidden-states-path $ROOT_DIR/cache/hidden_states/qwen3.5-35b-a3b-ultrachat \
#     --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
#     --output-dir $ROOT_DIR/outputs/qwen3.5-35b-a3b-ultrachat \
#     --num-epochs 10 \
#     --batch-size 1 \
#     --tp-size 1 \
#     --learning-rate 5e-5 \
#     --max-length 4096 \
#     --chat-template qwen \
#     --cache-dir $ROOT_DIR/cache \
#     --embedding-key "model.language_model.embed_tokens.weight"
