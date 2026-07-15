"""Convert supported training datasets to SpecForge conversation JSONL.

All presets produce rows with a stable id and a conversations list. Heavy
dataset dependencies stay behind loader functions so row conversion helpers and
their tests remain usable in lightweight environments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SUPPORTED_DATASETS = (
    "ultrachat",
    "sharegpt",
    "eaglechat",
    "perfectblend",
    "perfectblend-llama3.1-8b-instruct",
    "perfectblend-llama3.3-70b-instruct",
    "perfectblend-llama4-scout-instruct",
    "perfectblend-llama4-maverick-instruct",
    "magpie-qwen2.5-pro-1m-v0.1",
    "sharegpt4v",
    "allava4v",
    "opc",
    "gsm8k",
    "hendrycks_math",
    "math_qa",
    "codealpaca-20k",
    "opencodeinstruct",
    "magicoder-evol-instruct",
    "sciq",
    "camel",
    "nebius-llama31-8b-infinity-instruct",
)
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent.parent / "cache" / "dataset"
SUPPORTED_DATA_PATH_SUFFIXES = {".json", ".jsonl"}
OPC_SUBSETS = (
    "largescale_diverse_instruct",
    "filtered_infinity_instruct",
    "realuser_instruct",
)

ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
}

ProcessedRow = tuple[dict[str, Any] | None, int]
RowProcessor = Callable[[Mapping[str, Any], str | None], ProcessedRow]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a supported SpecForge training dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=SUPPORTED_DATASETS,
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
    parser.add_argument(
        "--split-eval",
        action="store_true",
        help="Write a deterministic 5% evaluation split.",
    )
    parser.add_argument(
        "--opc-subset",
        choices=(*OPC_SUBSETS, "all"),
        default=OPC_SUBSETS[0],
        help="OpenCoder opc-sft-stage1 subset, or all supported subsets.",
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


def _load_hf_dataset(*args: Any, **kwargs: Any) -> Any:
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def _concatenate_hf_datasets(datasets: Sequence[Any]) -> Any:
    from datasets import concatenate_datasets

    return concatenate_datasets(list(datasets))


def _stable_id(*parts: str) -> str:
    return hashlib.md5("".join(parts).encode()).hexdigest()


def _conversation_row(
    row_id: Any,
    user_content: str,
    assistant_content: str,
) -> dict[str, Any]:
    return {
        "id": str(row_id),
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def process_ultrachat_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
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


def process_sharegpt_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
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


def process_nebius_infinity_instruct(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    conversation = row["conversation"][0]
    generated_message = row["generated_message"]
    return (
        _conversation_row(
            row["id"],
            conversation["content"],
            generated_message["content"],
        ),
        0,
    )


def process_opc_sft_stage1(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    instruction = row["instruction"]
    output = row["output"]
    return (
        _conversation_row(
            _stable_id(instruction, output),
            instruction,
            output,
        ),
        0,
    )


def process_codealpaca_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    return process_opc_sft_stage1(row, dataset_name)


def process_opencodeinstruct_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    row_id = row.get("id") or _stable_id(row["input"], row["output"])
    return _conversation_row(row_id, row["input"], row["output"]), 0


def process_magicoder_evol_instruct_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    instruction = row["instruction"]
    response = row["response"]
    return (
        _conversation_row(
            _stable_id(instruction, response),
            instruction,
            response,
        ),
        0,
    )


def process_gsm8k_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    question = row["question"]
    answer = row["answer"]
    return _conversation_row(_stable_id(question, answer), question, answer), 0


def process_hendrycks_math_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    problem = row["problem"]
    solution = row["solution"]
    return _conversation_row(_stable_id(problem, solution), problem, solution), 0


def process_math_qa_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    user_content = f"{row['Problem']}\n{row['options']}"
    rationale = row["Rationale"]
    return (
        _conversation_row(
            _stable_id(user_content, rationale),
            user_content,
            rationale,
        ),
        0,
    )


def process_sciq_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    answers = [
        row["distractor3"],
        row["distractor1"],
        row["distractor2"],
        row["correct_answer"],
    ]
    random.shuffle(answers)
    labels = ("a", "b", "c", "d")
    options = list(zip(labels, answers))
    correct_label = next(
        label for label, answer in options if answer == row["correct_answer"]
    )
    options_text = "\n".join(f"{label}) {answer}" for label, answer in options)
    user_content = f"{row['question']}\n{options_text}"
    assistant_content = (
        f"{row['support']}\nanswer: {correct_label}) {row['correct_answer']}"
    )
    return (
        _conversation_row(
            _stable_id(user_content, assistant_content),
            user_content,
            assistant_content,
        ),
        0,
    )


def process_camel_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    user_content = row["message_1"]
    assistant_content = row["message_2"]
    return (
        _conversation_row(
            _stable_id(user_content, assistant_content),
            user_content,
            assistant_content,
        ),
        0,
    )


def get_cache_dir(dataset_name: str) -> Path:
    if dataset_name == "sharegpt4v":
        raise ValueError("Downloading 'sharegpt4v' is not supported.")
    if dataset_name != "allava4v":
        raise ValueError(
            f"Dataset {dataset_name!r} is not a supported VLM download preset."
        )

    from datasets import config

    return Path(config.HF_DATASETS_CACHE) / "FreedomIntelligence" / "ALLaVA"


def download_vlm_dataset(dataset_name: str) -> None:
    """Download the external ALLaVA image archive when it is not cached."""
    cache_dir = get_cache_dir(dataset_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    first_archive = cache_dir / "allava_laion" / "image_chunks" / "images_0.zip"
    if first_archive.exists():
        print("##### allava4v dataset already exists.")
        return

    script_path = (
        Path(__file__).resolve().parent.parent / "datasets" / "download_laion.sh"
    )
    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=cache_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Download image dataset failed: {result.stderr}")
    print("##### allava4v dataset download complete.")


def process_sharegpt4v_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    if dataset_name is None:
        raise ValueError("VLM row processing requires a dataset name")
    image_path = get_cache_dir(dataset_name) / row["image"]
    if not image_path.exists():
        print(f"Image path {image_path} does not exist, skipping this sample.")
        return None, 0

    conversations = []
    skipped_count = 0
    for message in row["conversations"]:
        role = ROLE_MAPPING.get(message["from"])
        if role is None:
            skipped_count += 1
            continue
        content = message["value"]
        if role == "user":
            content = content.replace("<image>\n", "")
        conversations.append({"role": role, "content": content})
    return {
        "id": str(row["id"]),
        "image": str(image_path),
        "conversations": conversations,
    }, skipped_count


def _identity_row(
    row: Mapping[str, Any], dataset_name: str | None = None
) -> ProcessedRow:
    del dataset_name
    return dict(row), 0


def add_index(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    indexed = dict(row)
    indexed["id"] = index
    return indexed


def load_dataset_from_path(data_path: Path) -> Any:
    suffix = data_path.suffix.lower()
    if suffix not in SUPPORTED_DATA_PATH_SUFFIXES:
        raise ValueError(
            f"Unsupported ShareGPT data file {data_path}; expected .json or .jsonl"
        )
    return _load_hf_dataset("json", data_files=str(data_path), split="train")


def _train_split(*args: Any, **kwargs: Any) -> Any:
    return _load_hf_dataset(*args, **kwargs)["train"]


def _indexed(dataset: Any) -> Any:
    return dataset.map(add_index, with_indices=True)


def load_dataset_preset(
    dataset_name: str,
    *,
    data_path: Path | None = None,
    opc_subset: str = OPC_SUBSETS[0],
) -> tuple[Any, RowProcessor | None]:
    """Load one named preset and return its row processor."""
    if dataset_name == "ultrachat":
        return (
            _load_hf_dataset(
                "HuggingFaceH4/ultrachat_200k",
                split="train_sft",
            ),
            process_ultrachat_row,
        )
    if dataset_name == "sharegpt":
        dataset = (
            load_dataset_from_path(data_path)
            if data_path is not None
            else _load_hf_dataset(
                "Aeala/ShareGPT_Vicuna_unfiltered",
                split="train",
            )
        )
        return dataset, process_sharegpt_row
    if dataset_name == "eaglechat":
        return _train_split("zhaode/EagleChat"), _identity_row
    if dataset_name == "perfectblend":
        return (
            _indexed(_train_split("mlabonne/open-perfectblend")),
            process_sharegpt_row,
        )

    regenerated_presets = {
        "perfectblend-llama3.1-8b-instruct": (
            "frankleeeee/PerfectBlend-Regenerated-Llama-3.1-8B-Instruct"
        ),
        "perfectblend-llama3.3-70b-instruct": (
            "frankleeeee/PerfectBlend-Regenerated-Llama-3.3-70B-Instruct"
        ),
        "perfectblend-llama4-scout-instruct": (
            "frankleeeee/PerfectBlend-Regenerated-Llama-4-Scout-17B-16E-Instruct"
        ),
        "perfectblend-llama4-maverick-instruct": (
            "frankleeeee/PerfectBlend-Regenerated-Llama-4-Maverick-17B-128E-Instruct"
        ),
    }
    if dataset_name in regenerated_presets:
        return _indexed(_train_split(regenerated_presets[dataset_name])), _identity_row

    if dataset_name == "magpie-qwen2.5-pro-1m-v0.1":
        dataset = _train_split("Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1")
        return dataset.rename_column("uuid", "id"), process_sharegpt_row
    if dataset_name == "sharegpt4v":
        raise ValueError(
            "ShareGPT4V image downloading is not supported; use allava4v instead."
        )
    if dataset_name == "allava4v":
        dataset = _load_hf_dataset(
            "FreedomIntelligence/ALLaVA-4V",
            name="allava_laion",
        )["instruct"]
        download_vlm_dataset(dataset_name)
        return dataset, process_sharegpt4v_row
    if dataset_name == "nebius-llama31-8b-infinity-instruct":
        dataset = _load_hf_dataset(
            "nebius/Llama-3.1-8B-Instruct-Infinity-Instruct-0625",
            split="train",
        )
        return _indexed(dataset), process_nebius_infinity_instruct
    if dataset_name == "opc":
        if opc_subset == "all":
            dataset = _concatenate_hf_datasets(
                [
                    _train_split("OpenCoder-LLM/opc-sft-stage1", subset)
                    for subset in OPC_SUBSETS
                ]
            )
        else:
            dataset = _train_split("OpenCoder-LLM/opc-sft-stage1", opc_subset)
        return dataset, process_opc_sft_stage1
    if dataset_name == "gsm8k":
        return _train_split("openai/gsm8k", "main"), process_gsm8k_row
    if dataset_name == "hendrycks_math":
        subjects = (
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        )
        return (
            _concatenate_hf_datasets(
                [
                    _train_split("EleutherAI/hendrycks_math", subject)
                    for subject in subjects
                ]
            ),
            process_hendrycks_math_row,
        )
    if dataset_name == "math_qa":
        return (
            _train_split("allenai/math_qa", trust_remote_code=True),
            process_math_qa_row,
        )
    if dataset_name == "codealpaca-20k":
        return (
            _train_split("sahil2801/CodeAlpaca-20k", trust_remote_code=True),
            process_codealpaca_row,
        )
    if dataset_name == "opencodeinstruct":
        return (
            _train_split("nvidia/OpenCodeInstruct", trust_remote_code=True),
            process_opencodeinstruct_row,
        )
    if dataset_name == "magicoder-evol-instruct":
        return (
            _train_split(
                "ise-uiuc/Magicoder-Evol-Instruct-110K",
                trust_remote_code=True,
            ),
            process_magicoder_evol_instruct_row,
        )
    if dataset_name == "sciq":
        return (
            _train_split("allenai/sciq", trust_remote_code=True),
            process_sciq_row,
        )
    if dataset_name == "camel":
        return (
            _concatenate_hf_datasets(
                [
                    _load_hf_dataset(f"camel-ai/{subject}", split="train")
                    for subject in ("biology", "chemistry", "physics")
                ]
            ),
            process_camel_row,
        )
    raise ValueError(
        f"Unsupported dataset preset {dataset_name!r}; choose from {SUPPORTED_DATASETS}"
    )


def load_canonical_dataset(dataset_name: str, data_path: Path | None = None) -> Any:
    """Compatibility wrapper returning only the dataset for a named preset."""
    dataset, _ = load_dataset_preset(dataset_name, data_path=data_path)
    return dataset


def _write_split(
    dataset: Iterable[Mapping[str, Any]],
    output_path: Path,
    processor: RowProcessor | None,
    dataset_name: str,
) -> int:
    skipped_messages = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for item in dataset:
            if processor is None:
                row, skipped_count = dict(item), 0
            else:
                row, skipped_count = processor(item, dataset_name)
            if row is None:
                continue
            skipped_messages += skipped_count
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return skipped_messages


def process_and_save_dataset(
    dataset: Iterable[Mapping[str, Any]],
    output_directory: Path,
    processor: RowProcessor | None,
    dataset_name: str,
    *,
    eval_dataset: Iterable[Mapping[str, Any]] | None = None,
) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    train_output_path = output_directory / f"{dataset_name}_train.jsonl"
    if train_output_path.exists():
        print(f"Dataset already exists at {train_output_path}; skipping conversion.")
        return train_output_path

    skipped_messages = _write_split(
        dataset,
        train_output_path,
        processor,
        dataset_name,
    )
    if eval_dataset is not None:
        eval_output_path = output_directory / f"{dataset_name}_test.jsonl"
        skipped_messages += _write_split(
            eval_dataset,
            eval_output_path,
            processor,
            dataset_name,
        )

    if skipped_messages:
        print(
            f"Skipped {skipped_messages} unsupported messages while "
            f"processing {dataset_name}."
        )
    print(f"Saved {dataset_name} training data to {train_output_path}.")
    return train_output_path


def process_and_save_ds(
    train_ds: Iterable[Mapping[str, Any]],
    test_ds: Iterable[Mapping[str, Any]] | None,
    output_path: Path,
    proc_fn: RowProcessor | None,
    dataset_name: str,
) -> Path:
    """Backward-compatible wrapper for external data preparation callers."""
    return process_and_save_dataset(
        train_ds,
        output_path,
        proc_fn,
        dataset_name,
        eval_dataset=test_ds,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset, processor = load_dataset_preset(
        args.dataset,
        data_path=args.data_path,
        opc_subset=args.opc_subset,
    )

    if args.sample_size is not None and args.sample_size < len(dataset):
        dataset = dataset.select(range(args.sample_size))
        print(f"Processing {args.sample_size} samples from {args.dataset}.")

    eval_dataset = None
    if args.split_eval:
        split = dataset.train_test_split(test_size=0.05, seed=42)
        dataset = split["train"]
        eval_dataset = split["test"]

    process_and_save_dataset(
        dataset,
        args.output_path,
        processor,
        args.dataset,
        eval_dataset=eval_dataset,
    )


if __name__ == "__main__":
    main()
