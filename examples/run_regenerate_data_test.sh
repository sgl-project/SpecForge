SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)


# regenerate eagle3 train data
# NUM_GPUS=${1:-8}

# python3 \
#     $ROOT_DIR/scripts/regenerate_train_data.py \
#     --model /mnt/tidalfs-hssh01/dataset/xiaowen/model/plan_toolv6x0825/tool/v0-20250824-173634/checkpoint-302 \
#     --input-file-path /mnt/tidalfs-hssh01/dataset/xiaowen/data/planning/tool.v6_qwen3_openai_infer.processed_eval_100_convert.jsonl \
#     --output-file-path /mnt/tidalfs-hssh01/dataset/xiaowen/data/planning/tool.v6_qwen3_openai_infer.processed_eval_100_convert_test.jsonl \
#     --batch-size 128 \
#     --tp-size $NUM_GPUS \
#     --num-samples 1000 \
#     --port 30000 \
#     --temperature 0 \
#     --mem-fraction-static 0.85 \
#     --auto-launch-server

python3 \
    $ROOT_DIR/scripts/regenerate_train_data.py \
    --model /mnt/nj-larc/dataset/xiaowen/model/qwen35-122b \
    --input-file-path /mnt/nj-larc/dataset/xiaowen/data/w1w/train.jsonl \
    --output-file-path /mnt/nj-larc/dataset/xiaowen/data/w1w/train_regen.jsonl \
    --concurrency 32 \
    --max-tokens 4096 \
    --num-samples 100000 \
    --temperature 0 \
    --server-address 127.0.0.1:8081 \
    --resume \
    --is-reasoning-model