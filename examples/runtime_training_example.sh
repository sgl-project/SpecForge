#!/bin/bash

# run server 
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

cd SpecForge

python3 -m sglang.launch_server --model /models/gpt-oss-120b --tp 4 \
    --speculative-draft-model-path /cache/from_scratch_dumped_train_fixed_output/epoch_9 \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 8 \
    --mem-fraction 0.8 \
    --speculative-algorithm EAGLE3 \
    --cuda-graph-max-bs 32 \
    --port 41555 \
    --trust-remote-code \
    --disable-radix-cache \
    --enable-dump-hidden-states \
    --hidden-states-dump-path /cache/hidden_states_default \

# data generation
python3 -m sglang.bench_serving \
    --backend sglang-oai-chat\
    --dataset-name sharegpt \
    --num-prompts 1000\
    --model /models/gpt-oss-120b \
    --dataset-path  /cache/dataset_new/cluster0_user_test.json \
    --output-file output.jsonl \
    --max-concurrency 32 \
    --port 41555

# postprocess
python postprocess_test.py \
    --data-path /cache/hidden_states_default/ \
    --model-path /models/gpt-oss-120b/ \
    --output-path /cache/dump_train

python postprocess_test.py \
    --data-path /cache/hidden_states_default/ \
    --model-path /models/gpt-oss-120b/ \
    --output-path /cache/dump_eval \
    --test-mode

# finetuning
export NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_API_KEY=your_wandb_api_key_here
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    scripts/train_eagle3_offline.py \
    --target-model-path /models/gpt-oss-120b \
    --draft-model-config ./configs/gpt-oss-120B-eagle3.json \
    --train-hidden-states-path /cache/dump_train \
    --eval-hidden-states-path /cache/dump_eval  \
    --output-dir /cache/dump_output \
    --num-epochs 10 \
    --draft-global-batch-size 16 \
    --draft-micro-batch-size 1 \
    --learning-rate 5e-5 \
    --draft-attention-backend flex_attention \
    --max-length 2048 \
    --chat-template gpt-oss \
    --cache-dir /cache/dump_cache \
    --dist-timeout 3600 \
    --log-steps 1 \
    --is-preformatted \
    --finetune \
    --baseline-dir /workspace/EAGLE3-gpt-oss-120b-bf16 \
    --report-to wandb \
    --wandb-project gpt-oss-120b-eagle3 \
    --wandb-name dump-train-10epoch-batch16-lr5e-5