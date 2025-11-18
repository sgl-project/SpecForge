# ↩️ Regenerate Datasets

## 📍 Overview

When training speculative decoding draft models for a specific target model, instead of using the original dataset, we can regenerate the assistant responses using the target model to better align the draft model with the target model's output distribution. This will improve the acceptance rate of the draft model and the overall performance of the speculative decoding. According to the [EAGLE1 paper](https://arxiv.org/pdf/2401.15077), the EAGLE method is not very sensitive to the dataset quality, which means the performance is still good even if you use the original dataset. However, if you are looking for optimal performance in the production environment, it is recommended to regenerate the dataset using the target model.

## 🔧 Regeneration Workflow

We can follow the following steps to regenerate the dataset. In the example below, we will use `meta-llama/Llama-3.1-8B-Instruct` as an example, you can replace it with your own target model.

1. Start the SGLang server for the target model.

```shell
python3 -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --cuda-graph-bs 1 2 4 8 16 32 64 128 \
    --dtype bfloat16 \
    --mem-frac=0.8 \
    --port 30000
```

2. Regenerate the dataset using the `regenerate_train_data.py` script.

```shell
python scripts/regenerate_train_data.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --concurrency 128 \
    --max-tokens 98304 \
    --server-address localhost:30000 \
    --temperature 0.8 \
    --input-file-path ./cache/dataset/perfectblend_train.jsonl \
    --output-file-path ./cache/dataset/perfectblend_train_regen.jsonl
```

For maximum performance, we recommend to scale the number of GPUs to regenerate the dataset in data parallel mode. To do this, you can simply add more server addresses to the `--server-address` argument, e.g. `--server-address localhost:30000 localhost:30001 localhost:30002 localhost:30003`.
