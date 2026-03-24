
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# train eagle3 for Qwen3.5-35B-A3B on ultrachat with online data collection and training
TP_SIZE=1
BUILD_DATASET_NUM_PROC=64

export HF_DATASETS_CACHE=$ROOT_DIR/cache/hf_datasets
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

ATTENTION_BACKEND=${2:-flex_attention}
NUM_GPUS=4

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path /data/jiapingW/pretrained_models/Qwen3.5-35B-A3B \
    --draft-config-path $ROOT_DIR/configs/qwen3.5-35b-a3b-dflash.json \
    --train-data-path $ROOT_DIR/cache/dataset/opc_train_regen_first_turn.jsonl \
    --output-dir $ROOT_DIR/outputs/qwen3.5-35a-a3b-dflash-opc \
    --num-epochs 10 \
    --batch-size 2 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 4096 \
    --chat-template qwen3.5 \
    --attention-backend $ATTENTION_BACKEND \
    --num-anchors 512 \
    --loss-decay-gamma 7.0 \
    --log-interval 50 \
    --save-interval 10000 \
    --report-to tensorboard \
    --target-model-backend sglang \
    --block-size 16 \
    --num-anchors 512 \
    --sglang-mem-fraction-static 0.5 \
    --embedding-key model.language_model.embed_tokens.weight \
    --resume
