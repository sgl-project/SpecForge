## 📝 Data Preparation

In this section, we will introduce how to prepare the dataset for both online and offline training. As mentioned in the [Overview](#-overview) section, online training only requires the raw dataset while offline training requires the hidden states generated from the target model. In the section below, we will introduce how to prepare both the raw dataset and the hidden states.

### ☁️ Prepare Online Training Dataset

We have provided a script to prepare some sample datasets including ultrachat (200k) and sharegpt (120k) for demo purpose. You can easily process the dataset by running the following command. The jsonl files will be placed in the `cache/dataset/<dataset_name>` directory of the project path by default. These datasets will be processed into `jsonl` files, which are the raw dataset ready for online training!

```bash
# ultrachat
python scripts/prepare_data.py --dataset ultrachat

# sharegpt
python scripts/prepare_data.py --dataset sharegpt
```

### 💾 Prepare Offline Training Dataset

Compared to online data, offline data requires one more step for hidden states generation. Thus, before you delve into this section, make sure you have your `jsonl` files ready as mentioned in the [Prepare Online Training Dataset](#-prepare-online-training-dataset) section. Once you have the `jsonl` files, you can start the hidden states generation.

You can run the following command to obtain the hidden states.

```bash
torchrun --nproc_per_node=8 \
    scripts/prepare_hidden_states.py \
    --model-path <target-model-path> \
    --enable-aux-hidden-states \
    --data-path <jsonl-file-path> \
    --chat-template llama3 \
    --max-length 2048 \
    --tp-size 8 \
    --batch-size 4 \
    --mem-frac=0.75 \
    --num-samples 1000
```
> ⚠️ This extract may take 2 hours and about 5T Disk

You need to specify the following arguments:
- `--model-path`: this is the huggingface repo name or path to the target model.
- `--data-path`: this is actual output path from the previous `prepare_data.py` script.
- `--chat-template`: this is the chat template to use for this model.
- `--num-samples`: this specifies how many data samples to use for hidden states generation. By default it will use all the data from `data-path`.


### 🤩 Prepare your own dataset

Besides the provided ShareGPT/Ultrachat datasets, you can also prepare your own dataset. We support two formats:

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
torchrun --standalone --nproc_per_node 8 \
    scripts/train_eagle3_online.py \
    --is-preformatted \
    --chat-template qwen \
    --train-data-path ./your_preformatted_dataset.jsonl \
    # ... other arguments
```

Once you have the `jsonl` file ready, you can go straight for online training or hidden states generation for offline training.

If you have multiple datasets, you can just merge them into the one jsonl file. For example, you can do something like this

```bash
cat dataset1.jsonl dataset2.jsonl > merged_dataset.jsonl
```
