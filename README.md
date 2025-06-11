# SGL Speculative

sgl-spec is a repository for training various speculative decoding methods and supports seamless migration to SGLang for inference workflows.


# Model Decoding Training Support Table

| Decoding Training Method / Model Name | Llama3 | Llama4 Scout 17x16E | Qwen3 |
|--------------------------------------|--------|--------|--------------|
| EAGLE3                               | 🚧     | ✅     |      🚧      |

# Training Backend

We use hacked torchtune for training. need install torchtune.

# Training Eagle3

## Installation

```bash
# Clone and install locally
git clone <https://github.com/sgl-project/sgl-spec.git>
cd sgl-spec
pip install .
```

Note: This will automatically install SGLang and TorchTune with all dependencies.

## 1. Training Data Fetch

- [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)

- [Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered](https://huggingface.co/datasets/Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered)

## 2. Filter Training Data

When start traing step, we should using base model filter training data firstly.

We first need to use sglang to start the llama4 17Bx16E service

```bash
python3 -m sglang.launch_server --model-path meta-llama/Llama-4-Scout-17B-16E-Instruct --port 30000 --tp 4 --mem-fraction-static 0.8 --context-length 8192
```

### filter ultrachat_200k

Using tools/filter_data.py for data filter.

```bash
python3 tools/filter_data.py  --dataset-name HuggingFaceH4/ultrachat_200k  --dataset-split train_sft --parallel 128
```

### filter Magpie-Llama-3.1-Pro-300K-Filtered

```bash
python3 tools/filter_data.py  --dataset-name Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered  --dataset-split train --parallel 128
```

## 3. Training spec head

### 3.1 Download model from HF

```bash
tune download meta-llama/Llama-4-Scout-17B-16E-Instruct --hf-token xxx
```

### 3.2 Training from scratch

- one node

```bash
tune run --nproc_per_node 8 spec/eagle_full_finetune_distributed.py --config spec/configs/llama4/scout_17B_16E_eagle.yaml
```

- multi node

```bash
# this is for node 1
tune run --nproc-per-node 8 --nnodes 2 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "xx.xx.xx.xx:29500" spec/eagle_full_finetune_distributed.py --config spec/configs/llama4/scout_17B_16E_eagle.yaml > log 2>&1

# use node1 master-addr
# this is for node 2
tune run --nproc-per-node 8 --nnodes 2 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "xx.xx.xx.xx:29500" spec/eagle_full_finetune_distributed.py --config spec/configs/llama4/scout_17B_16E_eagle.yaml > log 2>&1
```

## 4. Export spec

```bash
python3 tools/extract_draft_model.py --checkpoint_dir /tmp/torchtune/llama4_17Bx16E/full
```

# Pretrained Eagle3

- Qwen3

- Llama4

# Filtered Dataset
