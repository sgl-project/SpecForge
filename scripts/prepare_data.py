import argparse
import json
import os
from pathlib import Path
from typing import Dict

from datasets import load_dataset
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
        choices=["ultrachat", "sharegpt", "opc"],
        help="The demo dataset to quickly run the training for speculative decoding",
    )
    parser.add_argument(
        "--output_path",
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
    return parser.parse_args()


def process_ultrachat_row(row) -> Dict:
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


def process_sharegpt_row(row) -> Dict:
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


def load_dataset_from_path(data_path: Path):
    suffix = data_path.suffix.split(".")[1]
    ds = load_dataset(suffix, data_files=str(data_path), split="train")
    return ds


import hashlib


def process_opc_sft_stage1(row) -> Dict:
    row_id = hashlib.md5((row["instruction"] + row["output"]).encode()).hexdigest()
    return {
        "id": row_id,
        "conversations": [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["output"]},
        ],
    }


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
    elif args.dataset == "opc":
        ds = load_dataset(
            "OpenCoder-LLM/opc-sft-stage1", "largescale_diverse_instruct"
        )["train"]
        proc_fn = process_opc_sft_stage1
    else:
        raise ValueError(
            f"This script only supports ultrachat_200k and sharegpt datasets for demo purpose, if you wish to use other datasets, please modify this script."
        )

    if args.output_path is None:
        root_path = Path(__file__).parent.parent
        output_path = root_path.joinpath("cache", "dataset")
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(args.output_path)

    output_jsonl_path = output_path.joinpath(f"{args.dataset}.jsonl")

    if output_jsonl_path.exists():
        print(
            f"The dataset {args.dataset} has already been processed and saved in {output_jsonl_path}, skipping..."
        )
        return

    total_skipped_count = 0
    with open(output_jsonl_path, "w") as f:
        for item in tqdm(ds, desc=f"Processing {args.dataset} dataset"):
            row, skipped_count = proc_fn(item)
            total_skipped_count += skipped_count
            f.write(json.dumps(row) + "\n")

    if total_skipped_count > 0:
        print(f"Skipped {total_skipped_count}/{len(ds)} messages for {args.dataset}")


if __name__ == "__main__":
    main()
