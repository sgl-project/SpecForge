# Eagle3 for Llama3 - Online

## Introduction

This document provides a step-by-step guide on how to train the EAGLE3 model for the Llama3.1-8B-Instruct model in an online manner. In online training, we generate the hidden states required by EAGLE3 draft model on the fly during training. This example is using `ShareGPT` dataset for training, the performance is not optimal due to the size and limited coverage of the dataset. If you look for optimal performance, we recommend you to try more diverse datasets such as [`Perfect-Blend`](https://huggingface.co/datasets/facebook/perfect-blend). We have also included a section on training on `Perfect-Blend` dataset at the end of this document.


## Training on ShareGPT dataset

### **Step 1. Prepare ShareGPT dataset**

First of all, we should download the dataset.

```shell
python ./scripts/prepare_data.py --dataset sharegpt
```

### **Step 2. Launch Online Training**

```shell
torchrun \
    --standalone \
    --nproc_per_node 8 \
    scripts/train_eagle3.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config configs/llama3-8B-eagle3.json \
    --train-data-path ./cache/dataset/sharegpt_train.jsonl \
    --output-dir ./outputs/llama3-8b-eagle3 \
    --num-epochs 2 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template llama3 \
    --target-model-backend sglang \
```

### **Step 3. Benchmark**

For `Llama3.1-8B`, we add a system prompt to all training data, following the approach used in the official repository. Consequently, when benchmarking, we should also include this system prompt to obtain the full accept length. Please uncomment the corresponding line and add the system prompt.

The four numbers in the config represent: `batch_size, num_steps, topk, num_verify_tokens`.  You can adjust the values in the config list to experiment with different test cases.

A pre-trained EAGLE model is available at [zhuyksir/EAGLE3-Llama-3.1-8B-Instruct](https://huggingface.co/zhuyksir/EAGLE3-Llama-3.1-8B-Instruct) for reference.

```shell
cd benchmarks

config_list=(
    "4,3,1,4"
    "4,7,10,60"
)
python3 bench_eagle3.py \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --speculative-draft-model-path /YOUR/PATH/Llama-3.1-8B-Instruct/dev_outputs/epoch_0 \
    --port 30000 \
    --mem-fraction-static 0.8 \
    --tp-size 1 \
    --config-list "${config_list[@]}" \
    --benchmark-list mtbench gsm8k humaneval math500
```


## Training on Perfect-Blend dataset

### **Step 1. Prepare Perfect-Blend dataset**

First of all, we should download the dataset.

```shell
python ./scripts/prepare_data.py --dataset perfectblend
```

### **Step 2. Launch Online Training**

We just need to change the `--train-data-path` to the path of the Perfect-Blend dataset (e.g. `./cache/dataset/perfectblend_train.jsonl`), then we can launch training smoothly.
