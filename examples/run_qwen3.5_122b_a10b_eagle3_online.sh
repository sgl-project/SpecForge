
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export https_proxy=10.140.24.177:3128

# train eagle3 for Qwen3.5-122B-A10B with online data collection and training
TP_SIZE=8
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

export HF_DATASETS_CACHE=$ROOT_DIR/cache/hf_datasets

NUM_GPUS=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path /mnt/nj-larc/dataset/xiaowen/model/qwen35-122b \
    --draft-model-config $ROOT_DIR/configs/qwen3.5-122b-a10b-eagle3-moe.json \
    --train-data-path /mnt/nj-larc/dataset/xiaowen/data/w1w/train_regen.jsonl  \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/qwen3.5-122b-a10b-moe-mtp \
    --num-epochs 5 \
    --batch-size 1 \
    --tp-size $TP_SIZE \
    --learning-rate 1e-4 \
    --max-length 16384 \
    --chat-template qwen3.5 \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key "model.language_model.embed_tokens.weight" \
    --sglang-mem-fraction-static 0.4 \
    --save-interval 5000 \
    --report-to wandb \
    --target-micro-batch-size 1 \
    --logits-chunk-size 2048

    
    # --report-to tensorboard \
    #     --wandb-project "w1w_moe_mtp" \
    # --wandb-name "qwen3.5-122b-eagle3-moe-run1" \
    # --wandb-key "972064c6eee2c9aae590" \