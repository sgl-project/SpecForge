
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# train eagle3 for Qwen3.5-35B-A3B on ultrachat with online data collection and training
TP_SIZE=1
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

export HF_DATASETS_CACHE=$ROOT_DIR/cache/hf_datasets

NUM_GPUS=2
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path Qwen/Qwen3.5-35B-A3B \
    --draft-model-config $ROOT_DIR/configs/qwen3.5-35b-a3b-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/ultrachat_train_regen_first_turn.jsonl  \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/qwen3.5-35b-a3b-ultrachat-regen \
    --num-epochs 10 \
    --batch-size 1 \
    --tp-size $TP_SIZE \
    --learning-rate 1e-4 \
    --max-length 8192 \
    --chat-template qwen3.5 \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key "model.language_model.embed_tokens.weight" \
    --sglang-mem-fraction-static 0.6 \
    --save-interval 5000 \
    --report-to tensorboard
