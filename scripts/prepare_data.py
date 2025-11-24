import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

"""
This script will convert the ultrachat/sharegpt dataset to the following schema in jsonl format:
{
    "id": str,
    "conversations": [
        {
            "role": str,
            "content": str
        }
    ],
}
"""

ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "ultrachat",
            "sharegpt",
            "eaglechat",
            "perfectblend",
            "magpie-qwen2.5-pro-1m-v0.1",
            "sharegpt4v",
            "allava4v",
            "opc",
        ],
        help="The demo dataset to quickly run the training for speculative decoding",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="The path to save the processed dataset, if not specified, the dataset will be saved in the cache/dataset/dataset_name directory of the root path",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="The path to the custom dataset, if not specified, the default dataset will be loaded",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="The number of samples to process from the dataset, if not specified, all samples will be processed",
    )
    parser.add_argument(
        "--split-eval",
        action="store_true",
        help="Whether to split the dataset into train and eval sets, default is False",
    )
    parser.add_argument(
        "--opc-subset",
        type=str,
        default="largescale_diverse_instruct",
        choices=[
            "largescale_diverse_instruct",
            "filtered_infinity_instruct",
            "realuser_instruct",
            "all",
        ],
        help="The subset of OpenCoder opc-sft-stage1 dataset to use, or 'all' to use all subsets (default: largescale_diverse_instruct)",
    )
    return parser.parse_args()


def process_ultrachat_row(row: Dict) -> Tuple[Dict, int]:
    """Process a row from the ultrachat dataset.

    The function expects a row with the following schema:
    "messages": [
        {
            "role": "user" | "assistant",
            "content": str
        }
    ]
    """
    conversations = row["messages"]
    formatted_conversations = []
    for message in conversations:
        role = message["role"]
        content = message["content"]
        assert role in ["user", "assistant"]
        formatted_conversations.append({"role": role, "content": content})
    row = {"id": row["prompt_id"], "conversations": formatted_conversations}
    return row, 0


def process_sharegpt_row(row: Dict) -> Tuple[Dict, int]:
    """
    sharegpt dataset schema:
    {
        "conversations": [
            {
                "from": <system|human|gpt>,
                "value": <message>,
            },
            ...
        ]
    }
    """
    conversations = row["conversations"]
    formatted_conversations = []
    skipped_count = 0
    for message in conversations:
        if message["from"] not in ROLE_MAPPING:
            skipped_count += 1
            continue
        new_role = ROLE_MAPPING[message["from"]]
        content = message["value"]
        formatted_conversations.append({"role": new_role, "content": content})

    row = {"id": row["id"], "conversations": formatted_conversations}
    return row, skipped_count


def process_sharegpt4v_row(row) -> Dict:
    """
    sharegpt4v dataset schema:
    {
        "id": str,
        "image": str,  # path to the image
        "conversations": [
            {
                "from": <human|gpt>,
                "value": <message>,
            },
            ...
        ]
    }
    """
    conversations = row["conversations"]
    image = f'FreedomIntelligence/ALLaVA-4V/{row["image"]}'
    if not os.path.exists(image):
        print(f"Image path {image} does not exist, skipping this sample.")
        return None, None
    formatted_conversations = []
    skipped_count = 0
    for message in conversations:
        if message["from"] not in ROLE_MAPPING:
            skipped_count += 1
            continue
        new_role = ROLE_MAPPING[message["from"]]
        if new_role == "user":
            text_content = message["value"].replace("<image>\n", "")
            content = text_content
        else:
            content = message["value"]
        formatted_conversations.append({"role": new_role, "content": content})

    row = {"id": row["id"], "image": image, "conversations": formatted_conversations}
    return row, skipped_count


def load_dataset_from_path(data_path: Path):
    suffix = data_path.suffix.split(".")[1]
    ds = load_dataset(suffix, data_files=str(data_path), split="train")
    return ds


def process_and_save_ds(train_ds, test_ds, output_path, proc_fn, dataset_name):
    train_output_jsonl_path = output_path.joinpath(f"{dataset_name}_train.jsonl")
    if train_output_jsonl_path.exists():
        print(
            f"The dataset {dataset_name} has already been processed and saved in {train_output_jsonl_path}, skipping..."
        )
        return

    total_skipped_count = 0
    with open(train_output_jsonl_path, "w") as f:
        for item in tqdm(train_ds, desc=f"Processing {dataset_name} dataset"):
            row, skipped_count = proc_fn(item)
            if row is None:
                continue
            total_skipped_count += skipped_count
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if test_ds is not None:
        test_output_jsonl_path = output_path.joinpath(f"{dataset_name}_test.jsonl")
        with open(test_output_jsonl_path, "w") as f:
            for item in tqdm(test_ds, desc=f"Processing {dataset_name} test dataset"):
                row, skipped_count = proc_fn(item)
                if row is None:
                    continue
                total_skipped_count += skipped_count
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if total_skipped_count > 0:
        total_messages = len(train_ds) + (len(test_ds) if test_ds is not None else 0)
        print(
            f"Skipped {total_skipped_count}/{total_messages} messages for {dataset_name}"
        )


import hashlib


def process_opc_sft_stage1(row: Dict) -> Tuple[Dict, int]:
    row_id = hashlib.md5((row["instruction"] + row["output"]).encode()).hexdigest()
    processed_row = {
        "id": row_id,
        "conversations": [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["output"]},
        ],
    }
    return processed_row, 0


def add_index(row, idx) -> Dict:
    row["id"] = idx
    return row


def main():
    args = parse_args()
    # load dataset
    if args.dataset == "ultrachat":
        ds = load_dataset("HuggingFaceH4/ultrachat_200k")["train_sft"]
        proc_fn = process_ultrachat_row
    elif args.dataset == "sharegpt":
        if args.data_path is None:
            ds = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered")["train"]
        else:
            print("Loading dataset from custom data path: ", args.data_path)
            ds = load_dataset_from_path(Path(args.data_path))
        proc_fn = process_sharegpt_row
    elif args.dataset == "eaglechat":
        ds = load_dataset("zhaode/EagleChat")["train"]
        proc_fn = lambda row: (row, 0)
    elif args.dataset == "perfectblend":
        ds = load_dataset("mlabonne/open-perfectblend")["train"]
        ds = ds.map(add_index, with_indices=True)
        proc_fn = process_sharegpt_row
    elif args.dataset == "magpie-qwen2.5-pro-1m-v0.1":
        ds = load_dataset("Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1")["train"]
        ds = ds.rename_column("uuid", "id")
        proc_fn = process_sharegpt_row
    elif args.dataset == "sharegpt4v":
        ds = load_dataset("Lin-Chen/ShareGPT4V", "ShareGPT4V")["train"]
        proc_fn = process_sharegpt4v_row
    elif args.dataset == "allava4v":
        ds = load_dataset("FreedomIntelligence/ALLaVA-4V", name="allava_laion")[
            "instruct"
        ]
        proc_fn = process_sharegpt4v_row
    elif args.dataset == "opc":
        if args.opc_subset == "all":
            # Load all subsets and concatenate them
            subsets = [
                "largescale_diverse_instruct",
                "filtered_infinity_instruct",
                "realuser_instruct",
            ]
            datasets_list = [
                load_dataset("OpenCoder-LLM/opc-sft-stage1", subset)["train"]
                for subset in subsets
            ]
            ds = concatenate_datasets(datasets_list)
        else:
            ds = load_dataset("OpenCoder-LLM/opc-sft-stage1", args.opc_subset)["train"]
        proc_fn = process_opc_sft_stage1
    else:
        raise ValueError(
            f"This script only supports ultrachat, sharegpt, sharegpt4v, allava4v, opc, and perfect-blend-gptoss-20B datasets for demo purpose, if you wish to use other datasets, please modify this script."
        )

    # filter and split dataset
    if args.sample_size is not None and args.sample_size < len(ds):
        ds = ds.select(range(args.sample_size))
        print(f"Processing {args.sample_size} samples from the dataset {args.dataset}")
    if args.split_eval:
        ds = ds.train_test_split(test_size=0.05)
        train_ds = ds["train"]
        test_ds = ds["test"]
    else:
        train_ds = ds
        test_ds = None

    if args.output_path is None:
        root_path = Path(__file__).parent.parent
        output_path = root_path.joinpath("cache", "dataset")
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(args.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    process_and_save_ds(train_ds, test_ds, output_path, proc_fn, args.dataset)


if __name__ == "__main__":
    main()
