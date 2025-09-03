SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# train eagle3 for GPT-OSS-20B
NUM_GPUS=${1:-8}

#/sgl-workspace/SpecForge/perfect-blend-gptoss-20B-shard20.jsonl
#/sgl-workspace/SpecForge/shard20_head100k.jsonl
#/root/.cache/huggingface/hub/datasets--zhuyksir--perfect-blend-gptoss-20B-1M/snapshots/3843e7defdc3ab1b418d6bee5e6a8f8099558f3c/perfect-blend-gptoss-20B-shard20.jsonl
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online.py \
    --target-model-path openai/gpt-oss-20b \
    --draft-model-config $ROOT_DIR/configs/gpt-oss-20B-eagle3.json \
    --train-data-path /sgl-workspace/SpecForge/shard20_head100k.jsonl \
    --output-dir $ROOT_DIR/outputs/perfect-blend-gptoss-20b-eagle3 \
    --num-epochs 1 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --report-to wandb \
    --max-length 2048 \
    --chat-template gpt-oss \
    --cache-dir $ROOT_DIR/cache \
    --build-dataset-num-proc 32 \
    --dist-timeout 60


# --train-data-path $ROOT_DIR/cache/dataset/perfect-blend-gptoss-20B.jsonl \
