
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# train eagle3 for deepseek-v3
NUM_GPUS=1
TP_SIZE=1
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

export HF_DATASETS_CACHE=$ROOT_DIR/cache/hf_datasets

NUM_GPUS=2
CUDA_VISIBLE_DEVICES=2,3 torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path /data/jiapingW/pretrained_models/Qwen3.5-35B-A3B \
    --draft-model-config $ROOT_DIR/configs/qwen3.5-35b-a3b-eagle3.json \
    --train-data-path /data/jiapingW/projects/SpecForge/cache/dataset/ultrachat_train_regen_1w_first_turn.jsonl  \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/qwen3.5-35b-a3b-ultrachat-regen-1w-first-turn/draft-vocab-32000-kvhead-2-eagle3-attn-error-template-qwen \
    --num-epochs 10 \
    --batch-size 1 \
    --tp-size 1 \
    --learning-rate 5e-5 \
    --max-length 8192 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key "model.language_model.embed_tokens.weight" \
    --sglang-mem-fraction-static 0.6 \
    --save-interval 5000 \
    --report-to tensorboard
