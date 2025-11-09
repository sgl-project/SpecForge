SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
DATASET_PATH=${DATASET_PATH:-"/tmp/sharegpt_train.jsonl"}

# regenerate eagle3 train data
# NUM_GPUS=${1:-8}
# python3 \
#     $ROOT_DIR/scripts/regenerate_data.py \
#     --model Qwen/QwQ-32B \
#     --input-file-path $ROOT_DIR/cache/dataset/sharegpt.jsonl \
#     --output-file-path $ROOT_DIR/cache/dataset/sharegpt_regenerate.jsonl \
#     --batch-size 128 \
#     --tp-size $NUM_GPUS \
#     --num-samples 1000 \
#     --port 30000 \
#     --temperature 0 \
#     --mem-fraction-static 0.85 \
#     --auto-launch-server

for i in {1..4}; do
    CUDA_VISIBLE_DEVICES=$i python3 -m sglang.launch_server \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --cuda-graph-bs 1 2 4 8 16 32 64 128 256 \
        --context-length 8192 \
        --dtype bfloat16 --mem-frac=0.8 --port $((30000 + i)) &
done

python $ROOT_DIR/scripts/generate_data_by_target.py \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --raw-data-file $DATASET_PATH/sharegpt.jsonl \
    --output-dir $DATASET_PATH/sharegpt-llama-3.1-8b-instruct \
    --max-concurrency 256 \
    --num-per-shard 10000 \
    --server-address-port 127.0.0.1:30001 127.0.0.1:30002 127.0.0.1:30003 127.0.0.1:30004
