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

For reasoning models, add `--reasoning save` to store `reasoning_content` in the regenerated dataset. To use a reasoning model with thinking disabled, add `--reasoning disable`, which forwards `chat_template_kwargs.enable_thinking=false` to the SGLang server and does not save `reasoning_content`.

For maximum performance, we recommend to scale the number of GPUs to regenerate the dataset in data parallel mode. To do this, you can simply add more server addresses to the `--server-address` argument, e.g. `--server-address localhost:30000 localhost:30001 localhost:30002 localhost:30003`.

### Qwen ShareGPT recipes

The Qwen recipe wraps regeneration, complete-row accounting, and output
validation. It expects one or more SGLang servers to already be running; it does
not launch or stop the servers itself.

For Qwen3-8B non-reasoning regeneration, start a target server in one terminal:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp-size 1 \
    --dtype bfloat16 \
    --mem-fraction-static 0.8 \
    --host 0.0.0.0 \
    --port 30000
```

After `curl --fail http://127.0.0.1:30000/health` succeeds, run the recipe from
the SpecForge repository root:

```bash
MODEL_PROFILE=qwen3-8b \
INPUT_FILE=./cache/dataset/sharegpt_train.jsonl \
OUTPUT_FILE=./cache/dataset/sharegpt_train_regen_qwen3_8b_temperature0_non_reasoning.jsonl \
SERVER_ADDRESSES="localhost:30000" \
bash examples/data_regeneration/run_qwen_sharegpt_regeneration.sh
```

This profile sets temperature to zero, disables thinking through
`chat_template_kwargs.enable_thinking=false`, and rejects non-empty structured
reasoning in successful rows.

For Qwen3.6-27B structured-reasoning regeneration, start SGLang with the Qwen3
reasoning parser so the OpenAI-compatible response includes
`reasoning_content`:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3.6-27B \
    --tp-size 1 \
    --dtype bfloat16 \
    --mem-fraction-static 0.8 \
    --reasoning-parser qwen3 \
    --host 0.0.0.0 \
    --port 30000
```

After `curl --fail http://127.0.0.1:30000/health` succeeds, run:

```bash
MODEL_PROFILE=qwen3.6-27b \
INPUT_FILE=./cache/dataset/sharegpt_train.jsonl \
OUTPUT_FILE=./cache/dataset/sharegpt_train_regen_qwen3.6-27b_temperature0_reasoning.jsonl \
SERVER_ADDRESSES="localhost:30000" \
bash examples/data_regeneration/run_qwen_sharegpt_regeneration.sh
```

The default maximum completion lengths are 4096 tokens for Qwen3-8B and 32768
tokens for Qwen3.6-27B. The server context length must accommodate both the
rendered prompt and `MAX_TOKENS`; override `MAX_TOKENS` when using a shorter
server context.

`INPUT_FILE` may point to another dataset without changing the recipe as long
as it is JSONL in the conversation format documented below, with a non-empty
string `id` and alternating `user`/`assistant` messages. Always choose a fresh
`OUTPUT_FILE`: the recipe deliberately refuses to overwrite or resume an
existing run.

For an output such as `dataset_regen.jsonl`, the recipe produces:

- `dataset_regen.jsonl`: successful regenerated training rows;
- `dataset_regen_error.jsonl`: request or generation errors;
- `dataset_regen_skipped.jsonl`: schema-invalid inputs and outputs excluded by the
  selected reasoning contract.

The run passes accounting only when
`success + error + skipped == input`. It prints the success fraction without a
fixed minimum threshold, then strictly validates every successful row,
including its conversation structure, reasoning contract, and absence of raw
`<think>` markers.

To distribute regeneration across independently launched servers, provide all
addresses as a space-separated list, for example:

```bash
SERVER_ADDRESSES="localhost:30000 localhost:30010" \
CONCURRENCY=64 \
bash examples/data_regeneration/run_qwen_sharegpt_regeneration.sh
```

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
