"""Convert the canonical demo datasets to SpecForge conversation JSONL."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

CANONICAL_DATASETS = ("ultrachat", "sharegpt")
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent.parent / "cache" / "dataset"
SUPPORTED_DATA_PATH_SUFFIXES = {".json", ".jsonl"}

ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
}

ProcessedRow = tuple[dict[str, Any], int]
RowProcessor = Callable[[Mapping[str, Any]], ProcessedRow]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a canonical SpecForge demo dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=CANONICAL_DATASETS,
        help="Dataset preset to prepare.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for <dataset>_train.jsonl (default: cache/dataset).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Custom ShareGPT JSON or JSONL file instead of the hosted preset.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Maximum number of source rows to process.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.data_path is not None:
        if args.dataset != "sharegpt":
            parser.error("--data-path is only supported with --dataset sharegpt")
        if args.data_path.suffix.lower() not in SUPPORTED_DATA_PATH_SUFFIXES:
            parser.error("--data-path must point to a .json or .jsonl file")
    if args.sample_size is not None and args.sample_size <= 0:
        parser.error("--sample-size must be greater than zero")

    return args


def process_ultrachat_row(row: Mapping[str, Any]) -> ProcessedRow:
    conversations = []
    for message in row["messages"]:
        role = message["role"]
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported UltraChat role: {role!r}")
        conversations.append({"role": role, "content": message["content"]})

    return {
        "id": str(row["prompt_id"]),
        "conversations": conversations,
    }, 0


def process_sharegpt_row(row: Mapping[str, Any]) -> ProcessedRow:
    conversations = []
    skipped_count = 0
    for message in row["conversations"]:
        role = ROLE_MAPPING.get(message["from"])
        if role is None:
            skipped_count += 1
            continue
        conversations.append({"role": role, "content": message["value"]})

    return {
        "id": str(row["id"]),
        "conversations": conversations,
    }, skipped_count


def _load_hf_dataset(*args: Any, **kwargs: Any) -> Any:
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def load_dataset_from_path(data_path: Path) -> Any:
    suffix = data_path.suffix.lower()
    if suffix not in SUPPORTED_DATA_PATH_SUFFIXES:
        raise ValueError(
            f"Unsupported ShareGPT data file {data_path}; expected .json or .jsonl"
        )
    return _load_hf_dataset("json", data_files=str(data_path), split="train")


def load_canonical_dataset(dataset_name: str, data_path: Path | None = None) -> Any:
    if dataset_name == "ultrachat":
        if data_path is not None:
            raise ValueError("Custom data paths are only supported for ShareGPT")
        return _load_hf_dataset(
            "HuggingFaceH4/ultrachat_200k",
            split="train_sft",
        )

    if dataset_name == "sharegpt":
        if data_path is not None:
            return load_dataset_from_path(data_path)
        return _load_hf_dataset(
            "Aeala/ShareGPT_Vicuna_unfiltered",
            split="train",
        )

    raise ValueError(
        f"Unsupported dataset preset {dataset_name!r}; choose from {CANONICAL_DATASETS}"
    )


def process_and_save_dataset(
    dataset: Iterable[Mapping[str, Any]],
    output_directory: Path,
    processor: RowProcessor,
    dataset_name: str,
) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{dataset_name}_train.jsonl"
    if output_path.exists():
        print(f"Dataset already exists at {output_path}; skipping conversion.")
        return output_path

    total_skipped_messages = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for item in dataset:
            row, skipped_count = processor(item)
            total_skipped_messages += skipped_count
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    if total_skipped_messages:
        print(
            f"Skipped {total_skipped_messages} unsupported messages while "
            f"processing {dataset_name}."
        )
    print(f"Saved {dataset_name} training data to {output_path}.")
    return output_path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = load_canonical_dataset(args.dataset, args.data_path)

    if args.sample_size is not None and args.sample_size < len(dataset):
        dataset = dataset.select(range(args.sample_size))

    processors: dict[str, RowProcessor] = {
        "ultrachat": process_ultrachat_row,
        "sharegpt": process_sharegpt_row,
    }
    process_and_save_dataset(
        dataset,
        args.output_path,
        processors[args.dataset],
        args.dataset,
    )


if __name__ == "__main__":
    main()
