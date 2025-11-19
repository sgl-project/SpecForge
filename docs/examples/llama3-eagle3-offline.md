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
torchrun \
    --standalone \
    --nproc_per_node 8 \
    scripts/prepare_hidden_states.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-aux-hidden-states \
    --data-path ./cache/dataset/sharegpt_train.jsonl \
    --output-path ./cache/hidden_states/sharegpt_train_Llama-3.1-8B-Instruct \
    --chat-template llama3 \
    --max-length 4096 \
    --tp-size 1 \
    --batch-size 32
```

The hidden states will be saved to the disk in the `output-path` directory.

### **Step 3. Start Training**

```shell
torchrun \
    --standalone \
    --nproc_per_node 8 \
    ./scripts/train_eagle3.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config ./configs/llama3-8B-eagle3.json \
    --train-data-path ./cache/dataset/sharegpt_train.jsonl \
    --train-hidden-states-path ./cache/hidden_states/sharegpt_train_Llama-3.1-8B-Instruct \
    --output-dir ./outputs/llama3-8b-eagle3-sharegpt-offline \
    --num-epochs 10 \
    --batch-size 1 \
    --tp-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template llama3 \
    --cache-dir ./cache
```
