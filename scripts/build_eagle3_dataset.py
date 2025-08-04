"""
Preprocess data for dataset generation. This runs faster without c10d comms.
"""

import argparse
import hashlib
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from specforge.data import build_eagle3_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.distributed.init_process_group(backend="nccl")
    assert os.path.exists(
        args.data_path
    ), f"Dataset path {args.data_path} does not exist"
    if args.output_path is None:
        root_path = Path(__file__).parent.parent
        args.output_path = root_path.joinpath("cache", "hidden_states")

    dataset = load_dataset("json", data_files=args.data_path)["train"]
    if args.num_samples is not None:
        print(f"Selecting {args.num_samples} samples from {len(dataset)}")
        dataset = dataset.select(range(args.num_samples))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    cache_key = hashlib.md5(args.data_path.encode()).hexdigest()
    eagle3_dataset = build_eagle3_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
        cache_key=cache_key,
    )
    print(f"Built dataset")


if __name__ == "__main__":
    main()
