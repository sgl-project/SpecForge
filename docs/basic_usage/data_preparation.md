# 📝 Data Preparation

## 📍 Overview

Data is an important aspect of speculative decoding as the quality of the dataset directly affects the acceptance rate of the draft model. In this section, we will introduce how to prepare the dataset for both online and offline training.

## ☁️ Pre-supported Datasets

We have provided a script to prepare some sample datasets out of the box, these datasets include:
1. [ultrachat](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) (200k)
2. [sharegpt](https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered) (120k)
3. [perfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend) (1.4M)
4. and others (we continuously add support for more datasets)

You can run the script below to prepare the corresponding dataset.

```bash
# ultrachat
python scripts/prepare_data.py --dataset ultrachat

# sharegpt
python scripts/prepare_data.py --dataset sharegpt
```

You can view the full list of pre-supported datasets using `python scripts/prepare_data.py --help`. The datasets are processed and saved as `jsonl` files in the `cache/dataset/<dataset_name>` directory of the project path by default.


## ↩️ Regenerate Datasets

When training speculative decoding draft models for a specific target model, instead of using the original dataset, we can regenerate the assistant responses using the target model to better align the draft model with the target model's output distribution. This will improve the acceptance rate of the draft model and the overall performance of the speculative decoding. According to the [EAGLE1 paper](https://arxiv.org/pdf/2401.15077), the EAGLE method is not very sensitive to the dataset quality, which means the performance is still good even if you use the original dataset. However, if you are looking for optimal performance in the production environment, it is recommended to regenerate the dataset using the target model.

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
    --input-file-path ./cache/dataset/sharegpt_train.jsonl \
    --output-file-path ./cache/dataset/sharegpt_train_regen.jsonl
```

For maximum performance, we recommend to scale the number of GPUs to regenerate the dataset in data parallel mode. To do this, you can simply add more server addresses to the `--server-address` argument, e.g. `--server-address localhost:30000 localhost:30001 localhost:30002 localhost:30003`.


## 🤩 Prepare your own dataset

Besides the provided datasets, you can also prepare your own dataset. We support two formats:

#### Option 1: Conversation Format

You should prepare the dataset in jsonl format and the schema should look like this:

```json
{
    "id": "xxxx",
    "conversations": [
        {
            "role": "user | assistant",
            "content": "The message content"
        }
    ],
}
```

#### Option 2: Pre-formatted Text Format

If you already have conversations formatted with a specific chat template, you can use the pre-formatted text directly:

```json
{
    "id": "xxxx",
    "text": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there!<|im_end|>\n"
}
```

This format is useful when you have pre-formatted prompts that were used during training of the target model and have raw generations from the target model.

To use pre-formatted datasets, add the `--is-preformatted` flag to your training command. Note that the `--chat-template` parameter is still needed and should match the template used in your pre-formatted text, as it is used to identify user/assistant tokens to determine the assistant spans and generate the corresponding loss mask.

```bash
# Online training with pre-formatted data
torchrun --standalone --nproc_per_node 8 \
    scripts/train_eagle3.py \
    --is-preformatted \
    --train-data-path ./your_preformatted_dataset.jsonl \
    # ... other arguments
```

For offline training, you can also use `--is-preformatted` when generating hidden states:

```bash
# Generate hidden states from pre-formatted data
torchrun --nproc_per_node=8 \
    scripts/prepare_hidden_states.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --data-path ./your_preformatted_dataset.jsonl \
    --output-path ./cache/hidden_states \
    --chat-template llama3 \
    --is-preformatted \
    --max-length 2048
```

Once you have the `jsonl` file ready, you can proceed with online training or generate hidden states for offline training. See the Training guide for more details.


## ➕ Handling Multiple Datasets

If you have multiple datasets, you can just merge them into the one jsonl file. For example, you can do something like this

```bash
cat dataset1.jsonl dataset2.jsonl > merged_dataset.jsonl
```
