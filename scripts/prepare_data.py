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

ROLE_MAPPING = {"human": "user", "gpt": "assistant"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ultrachat", "sharegpt"],
        help="The demo dataset to quickly run the training for speculative decoding",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="The path to save the processed dataset, if not specified, the dataset will be saved in the cache/dataset/dataset_name directory of the root path",
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
    return row


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

    for message in conversations:
        new_role = ROLE_MAPPING[message["from"]]
        content = message["value"]
        formatted_conversations.append({"role": new_role, "content": content})

    row = {"id": row["id"], "conversations": formatted_conversations}
    return row


def main():
    args = parse_args()

    # load dataset
    if args.dataset == "ultrachat":
        ds = load_dataset("HuggingFaceH4/ultrachat_200k")["train_sft"]
        proc_fn = process_ultrachat_row
    elif args.dataset == "sharegpt":
        ds = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered")["train"]
        proc_fn = process_sharegpt_row
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

    with open(output_jsonl_path, "w") as f:
        for item in tqdm(ds, desc=f"Processing {args.dataset} dataset"):
            row = proc_fn(item)
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
