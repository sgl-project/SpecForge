#!/usr/bin/env python3
# coding=utf-8
"""Prepare NVIDIA Nemotron Post-Training Dataset v2 for SpecForge training."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


DATASET_ID = "nvidia/Nemotron-Post-Training-Dataset-v2"
SUPPORTED_SPLITS = (
    "stem",
    "chat",
    "math",
    "code",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert nvidia/Nemotron-Post-Training-Dataset-v2 into SpecForge "
            "conversation JSONL files."
        )
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SUPPORTED_SPLITS),
        help="Nemotron v2 splits to mix before partitioning.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="Evaluation partition ratio. Use 0 to write only the train file.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional total sample count after mixing splits, useful for smoke tests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cache/dataset/nemotron-post-training-v2"),
        help="Directory for output JSONL files.",
    )
    parser.add_argument(
        "--train-output-path",
        type=Path,
        default=None,
        help="Optional explicit train JSONL path.",
    )
    parser.add_argument(
        "--eval-output-path",
        type=Path,
        default=None,
        help="Optional explicit eval JSONL path.",
    )
    return parser.parse_args()


def validate_splits(splits: Sequence[str]) -> List[str]:
    deduped_splits = list(dict.fromkeys(splits))
    invalid_splits = sorted(set(deduped_splits) - set(SUPPORTED_SPLITS))
    if invalid_splits:
        raise ValueError(
            f"Unsupported split(s): {', '.join(invalid_splits)}. "
            f"Supported splits: {', '.join(SUPPORTED_SPLITS)}."
        )
    return deduped_splits


def validate_eval_ratio(eval_ratio: float) -> None:
    if eval_ratio < 0 or eval_ratio >= 1:
        raise ValueError(f"--eval-ratio must be in [0, 1), got {eval_ratio}")


def normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def normalize_messages(messages: Any) -> List[Dict[str, str]]:
    if not isinstance(messages, list):
        return []

    conversations = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role is None:
            continue
        role = str(role)
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        conversations.append(
            {"role": role, "content": normalize_content(message.get("content"))}
        )
    return conversations


def row_to_specforge_conversation(row: Dict[str, Any]) -> Dict[str, Any]:
    conversations = normalize_messages(row.get("messages"))
    row_id = row.get("uuid") or row.get("id")
    if row_id is None:
        serialized = json.dumps(conversations, sort_keys=True, ensure_ascii=False)
        row_id = hashlib.md5(serialized.encode()).hexdigest()
    return {"id": str(row_id), "conversations": conversations}


def split_data_files(splits: Sequence[str]) -> Dict[str, str]:
    return {
        split: f"hf://datasets/{DATASET_ID}/data/{split}-*.parquet" for split in splits
    }


def load_nemotron_dataset(splits: Sequence[str]):
    try:
        from datasets import concatenate_datasets, load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required to prepare Nemotron v2 data."
        ) from exc

    split_arg: Any = list(splits) if len(splits) > 1 else splits[0]
    dataset = load_dataset(
        "parquet",
        data_files=split_data_files(splits),
        split=split_arg,
    )

    if isinstance(dataset, list):
        dataset = concatenate_datasets(dataset)
    return dataset


def maybe_sample_dataset(dataset, sample_size: Optional[int], seed: int):
    if sample_size is None:
        return dataset
    if sample_size < 1:
        raise ValueError("--sample-size must be positive when provided")
    sample_size = min(sample_size, len(dataset))
    return dataset.shuffle(seed=seed).select(range(sample_size))


def partition_dataset(dataset, eval_ratio: float, seed: int):
    validate_eval_ratio(eval_ratio)
    if eval_ratio == 0:
        return dataset, None
    split_dataset = dataset.train_test_split(
        test_size=eval_ratio,
        seed=seed,
        shuffle=True,
    )
    return split_dataset["train"], split_dataset["test"]


def default_output_paths(
    output_dir: Path,
    train_output_path: Optional[Path],
    eval_output_path: Optional[Path],
    eval_ratio: float,
) -> Tuple[Path, Optional[Path]]:
    train_path = train_output_path or output_dir / "nemotron_v2_train.jsonl"
    eval_path = None
    if eval_ratio > 0:
        eval_path = eval_output_path or output_dir / "nemotron_v2_eval.jsonl"
    return train_path, eval_path


def write_jsonl(dataset, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in dataset:
            item = row_to_specforge_conversation(row)
            if not item["conversations"]:
                continue
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    splits = validate_splits(args.splits)
    validate_eval_ratio(args.eval_ratio)

    dataset = load_nemotron_dataset(splits)
    dataset = maybe_sample_dataset(dataset, args.sample_size, args.seed)
    train_dataset, eval_dataset = partition_dataset(dataset, args.eval_ratio, args.seed)

    train_path, eval_path = default_output_paths(
        args.output_dir,
        args.train_output_path,
        args.eval_output_path,
        args.eval_ratio,
    )

    train_count = write_jsonl(train_dataset, train_path)
    print(f"Wrote {train_count} train examples to {train_path}")

    if eval_dataset is not None and eval_path is not None:
        eval_count = write_jsonl(eval_dataset, eval_path)
        print(f"Wrote {eval_count} eval examples to {eval_path}")


if __name__ == "__main__":
    main()
