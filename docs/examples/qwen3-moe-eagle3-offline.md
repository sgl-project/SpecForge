# Eagle3 for Llama3 - Offline

## Introduction

This document provides a step-by-step guide on how to train the EAGLE3 model for the Llama3.1-8B-Instruct model in an offline manner. In offline training, we generate the hidden states required by EAGLE3 draft model beforehand and store them to the disk. During training, we load them back to the GPU memory. As offline training requires a lot of disk space, we do not recommend running this on large datasets such as Perfect-Blend.

## Training on ShareGPT dataset

### **Step 1. Prepare ShareGPT dataset**

First of all, we should download the dataset.

```shell
python ./scripts/prepare_data.py --dataset sharegpt
```

### **Step 2. Prepare Hidden States**

We need to prepare the hidden states for the training.

```shell
torchrun --nproc_per_node=8 \
    scripts/prepare_hidden_states.py \
    --target-model-path /home/data/weights/Qwen3-32B \
    --enable-aux-hidden-states \
    --data-path ./cache/dataset/sharegpt_train.jsonl \
    --chat-template qwen \
    --max-length 2048 \
    --tp-size 8 \
    --batch-size 32 \
    --num-samples 20 \
    --output-path ./cache/hidden_states
```

The hidden states will be saved to the disk in the `output-path` directory.

### **Step 3. Start Training**

```shell
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path  /home/data/weights/Qwen3-30B-A3B/  \
    --draft-model-config $ROOT_DIR/configs/qwen3-30B-A3B-eagle3_moe.json \
    --train-data-path ./cache/dataset/sharegpt_train.jsonl \
    --train-hidden-states-path  ./cache/hidden_states \
    --output-dir ./outputs/qwen3-moe-8b-eagle3-sharegpt-offline \
    --num-epochs 4 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --save-interval 1984 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --sp-ulysses-size 4 \
    --attention-backend "usp" \
    --target-model-backend sglang

```
