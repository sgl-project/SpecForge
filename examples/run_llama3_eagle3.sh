#! /bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# train eagle3 for llama3.1-8b
torchrun \
    --standalone \
    --nproc_per_node 8 \
    $SCRIPT_DIR/train_online.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config $SCRIPT_DIR/configs/llama3-8b-eagle3.json \
    --train-data-path $SCRIPT_DIR/cache/dataset/sharegpt.jsonl \
    --output-dir $SCRIPT_DIR/outputs/llama3-8b-eagle3 \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --data-type llama3 \