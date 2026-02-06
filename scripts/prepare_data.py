import argparse
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Dict, Tuple

from tqdm import tqdm

from datasets import concatenate_datasets, config, load_dataset

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


def get_cache_dir(dataset_name):
    cache_dir = None
    if dataset_name == "sharegpt4v":
        raise ValueError("Downloading 'sharegpt4v' is not supported.")
    elif dataset_name == "allava4v":
        cache_dir = os.path.join(
            config.HF_DATASETS_CACHE, "FreedomIntelligence", "ALLaVA"
        )
    else:
        raise ValueError(
            f"Dataset '{dataset_name}' is not a supported VLM dataset for download."
        )
    return cache_dir


def download_vlm_dataset(dataset_name: str) -> None:
    """Download VLM's dataset such as sharegpt4v and allava4v"""
    if dataset_name == "sharegpt4v":
        raise Exception("Don't Support Download sharegpt4v.")
    elif dataset_name == "allava4v":
        cache_dir = get_cache_dir(dataset_name)
        os.makedirs(cache_dir, exist_ok=True)
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "datasets",
            "download_laion.sh",
        )
        os.chmod(script_path, 0o755)
        if not os.path.exists(
            os.path.join(cache_dir, "allava_laion", "image_chunks", "images_0.zip")
        ):
            result = subprocess.run(
                ["bash", script_path],
                cwd=cache_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Download image dataset failed: {result.stderr}")
            print("##### allava4v dataset Download Complete #####")
        else:
            print("##### allava4v dataset has existed.")
    else:
        raise Exception(f"Don't support {dataset_name}")


def process_ultrachat_row(row: Dict, dataset_name: str = None) -> Tuple[Dict, int]:
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


def process_sharegpt_row(row: Dict, dataset_name: str = None) -> Tuple[Dict, int]:
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


def process_sharegpt4v_row(row, dataset_name: str = None) -> Dict:
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
    cache_dir = get_cache_dir(dataset_name)
    conversations = row["conversations"]
    image = os.path.join(cache_dir, row["image"])
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
            if proc_fn is not None:
                row, skipped_count = proc_fn(item, dataset_name)
                if row is None:
                    continue
                total_skipped_count += skipped_count
            else:
                row = item
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if test_ds is not None:
        test_output_jsonl_path = output_path.joinpath(f"{dataset_name}_test.jsonl")
        with open(test_output_jsonl_path, "w") as f:
            for item in tqdm(test_ds, desc=f"Processing {dataset_name} test dataset"):
                if proc_fn is not None:
                    row, skipped_count = proc_fn(item, dataset_name)
                    if row is None:
                        continue
                    total_skipped_count += skipped_count
                else:
                    row = item
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


def process_codealpaca_row(row: Dict, dataset_name: str = None) -> Tuple[Dict, int]:
    """Process a row from the CodeAlpaca-20k dataset.
    
    The function expects a row with the following schema:
    {
        "instruction": str,
        "input": str,
        "output": str
    }
    """
    row_id = hashlib.md5((row["instruction"] + row["output"]).encode()).hexdigest()
    processed_row = {
        "id": row_id,
        "conversations": [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["output"]},
        ],
    }
    return processed_row, 0


def process_opencodeinstruct_row(row: Dict, dataset_name: str = None) -> Tuple[Dict, int]:
    """Process a row from the nvidia/OpenCodeInstruct dataset.
    
    The function expects a row with the following schema:
    {
        "id": str,
        "input": str,
        "output": str,
        "domain": str,
        "generation_algorithm": str,
        "llm_judgement": str,
        "unit_tests": str,
        "tests_execution_status": str,
        "average_test_score": float
    }
    """
    # Use the existing id if available, otherwise generate one
    row_id = row.get("id")
    if row_id is None:
        row_id = hashlib.md5((row["input"] + row["output"]).encode()).hexdigest()
    
    processed_row = {
        "id": row_id,
        "conversations": [
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["output"]},
        ],
    }
    return processed_row, 0


def process_magicoder_evol_instruct_row(row: Dict, dataset_name: str = None) -> Tuple[Dict, int]:
    """Process a row from the ise-uiuc/Magicoder-Evol-Instruct-110K dataset.
    
    The function expects a row with the following schema:
    {
        "instruction": str,
        "response": str
    }
    """
    row_id = hashlib.md5((row["instruction"] + row["response"]).encode()).hexdigest()
    processed_row = {
        "id": row_id,
        "conversations": [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["response"]},
        ],
    }
    return processed_row, 0


def process_gsm8k_row(row: Dict, dataset_name: str = None) -> Tuple[Dict, int]:
    """Process a row from the gsm8k dataset.
    
    The function expects a row with the following schema:
    {
        "question": str,
        "answer": str
    }
    """
    row_id = hashlib.md5((row["question"] + row["answer"]).encode()).hexdigest()
    processed_row = {
        "id": row_id,
        "conversations": [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ],
    }
    return processed_row, 0


def process_hendrycks_math_row(row: Dict, dataset_name: str = None) -> Tuple[Dict, int]:
    """Process a row from the hendrycks_math dataset.
    
    The function expects a row with the following schema:
    {
        "problem": str,
        "solution": str,
        "level": str,
        "type": str
    }
    """
    row_id = hashlib.md5((row["problem"] + row["solution"]).encode()).hexdigest()
    processed_row = {
        "id": row_id,
        "conversations": [
            {"role": "user", "content": row["problem"]},
            {"role": "assistant", "content": row["solution"]},
        ],
    }
    return processed_row, 0


def process_math_qa_row(row: Dict, dataset_name: str = None) -> Tuple[Dict, int]:
    """Process a row from the allenai/math_qa dataset.
    
    The function expects a row with the following schema:
    {
        "Problem": str,
        "Rationale": str,
        "options": str,  # format: "a) option1 b) option2 c) option3 d) option4"
        "correct": str,
        "annotated_formula": str,
        "linear_formula": str,
        "category": str
    }
    """
    # Combine Problem and options as user input
    problem = row["Problem"]
    options = row["options"]
    user_content = f"{problem}\n{options}"
    
    # Use Rationale as assistant response
    rationale = row["Rationale"]
    
    row_id = hashlib.md5((user_content + rationale).encode()).hexdigest()
    processed_row = {
        "id": row_id,
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": rationale},
        ],
    }
    return processed_row, 0


def process_sciq_row(row: Dict, dataset_name: str = None) -> Tuple[Dict, int]:
    """Process a row from the allenai/sciq dataset.
    
    The function expects a row with the following schema:
    {
        "question": str,
        "distractor3": str,
        "distractor1": str,
        "distractor2": str,
        "correct_answer": str,
        "support": str
    }
    """
    question = row["question"]
    correct_answer = row["correct_answer"]
    distractor1 = row["distractor1"]
    distractor2 = row["distractor2"]
    distractor3 = row["distractor3"]
    support = row["support"]
    
    # Create a list of all answers and randomly shuffle them
    answers_list = [distractor3, distractor1, distractor2, correct_answer]
    random.shuffle(answers_list)
    
    # Assign shuffled answers to labels a, b, c, d
    labels = ["a", "b", "c", "d"]
    options_list = [(labels[i], answers_list[i]) for i in range(4)]
    
    # Find the correct answer label after shuffling
    correct_label = None
    for label, answer in options_list:
        if answer == correct_answer:
            correct_label = label
            break
    
    # Format options as a string
    options_text = "\n".join([f"{label}) {answer}" for label, answer in options_list])
    user_content = f"{question}\n{options_text}"
    
    # Combine support with answer
    assistant_content = f"{support}\nanswer: {correct_label}) {correct_answer}"
    
    row_id = hashlib.md5((user_content + assistant_content).encode()).hexdigest()
    processed_row = {
        "id": row_id,
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }
    return processed_row, 0


def process_camel_row(row: Dict, dataset_name: str = None) -> Tuple[Dict, int]:
    """Process a row from the camel-ai dataset.
    
    The function expects a row with the following schema:
    {
        "message_1": str,  # user message
        "message_2": str,  # assistant message
    }
    """
    message_1 = row["message_1"]
    message_2 = row["message_2"]
    
    row_id = hashlib.md5((message_1 + message_2).encode()).hexdigest()
    processed_row = {
        "id": row_id,
        "conversations": [
            {"role": "user", "content": message_1},
            {"role": "assistant", "content": message_2},
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
        proc_fn = lambda row, name: (row, 0)
    elif args.dataset == "perfectblend":
        ds = load_dataset("mlabonne/open-perfectblend")["train"]
        ds = ds.map(add_index, with_indices=True)
        proc_fn = process_sharegpt_row
    elif args.dataset == "perfectblend-llama3.1-8b-instruct":
        ds = load_dataset("frankleeeee/PerfectBlend-Regenerated-Llama-3.1-8B-Instruct")[
            "train"
        ]
        ds = ds.map(add_index, with_indices=True)
        proc_fn = None
    elif args.dataset == "perfectblend-llama3.3-70b-instruct":
        ds = load_dataset(
            "frankleeeee/PerfectBlend-Regenerated-Llama-3.3-70B-Instruct"
        )["train"]
        ds = ds.map(add_index, with_indices=True)
        proc_fn = None
    elif args.dataset == "perfectblend-llama4-scout-instruct":
        ds = load_dataset(
            "frankleeeee/PerfectBlend-Regenerated-Llama-4-Scout-17B-16E-Instruct"
        )["train"]
        ds = ds.map(add_index, with_indices=True)
        proc_fn = None
    elif args.dataset == "perfectblend-llama4-maverick-instruct":
        ds = load_dataset(
            "frankleeeee/PerfectBlend-Regenerated-Llama-4-Maverick-17B-128E-Instruct"
        )["train"]
        ds = ds.map(add_index, with_indices=True)
        proc_fn = None
    elif args.dataset == "magpie-qwen2.5-pro-1m-v0.1":
        ds = load_dataset("Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1")["train"]
        ds = ds.rename_column("uuid", "id")
        proc_fn = process_sharegpt_row
    elif args.dataset == "sharegpt4v":
        ds = load_dataset("Lin-Chen/ShareGPT4V", "ShareGPT4V")["train"]
        raise Exception("Not supported sharegpt4v now")
        download_vlm_dataset(args.dataset)
        proc_fn = process_sharegpt4v_row
    elif args.dataset == "allava4v":
        ds = load_dataset("FreedomIntelligence/ALLaVA-4V", name="allava_laion")[
            "instruct"
        ]
        download_vlm_dataset(args.dataset)
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
    elif args.dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main")["train"]
        proc_fn = process_gsm8k_row
    elif args.dataset == "hendrycks_math":
        # Load all subjects and concatenate them
        subjects = [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ]
        datasets_list = [
            load_dataset("EleutherAI/hendrycks_math", subject)["train"]
            for subject in subjects
        ]
        ds = concatenate_datasets(datasets_list)
        proc_fn = process_hendrycks_math_row
    elif args.dataset == "math_qa":
        ds = load_dataset("allenai/math_qa", trust_remote_code=True)["train"]
        proc_fn = process_math_qa_row
    elif args.dataset == "codealpaca-20k":
        ds = load_dataset("sahil2801/CodeAlpaca-20k", trust_remote_code=True)["train"]
        proc_fn = process_codealpaca_row
    elif args.dataset == "opencodeinstruct":
        ds = load_dataset("nvidia/OpenCodeInstruct", trust_remote_code=True)["train"]
        proc_fn = process_opencodeinstruct_row
    elif args.dataset == "magicoder-evol-instruct":
        ds = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", trust_remote_code=True)["train"]
        proc_fn = process_magicoder_evol_instruct_row
    elif args.dataset == "sciq":
        ds = load_dataset("allenai/sciq", trust_remote_code=True)["train"]
        proc_fn = process_sciq_row
    elif args.dataset == "camel":
        # Load all three camel-ai datasets and concatenate them
        camel_datasets = [
            load_dataset("camel-ai/biology", split="train"),
            load_dataset("camel-ai/chemistry", split="train"),
            load_dataset("camel-ai/physics", split="train"),
        ]
        ds = concatenate_datasets(camel_datasets)
        proc_fn = process_camel_row
    else:
        raise ValueError(
            f"This script only supports ultrachat, sharegpt, sharegpt4v, allava4v, opc, gsm8k, hendrycks_math, math_qa, codealpaca-20k, opencodeinstruct, magicoder-evol-instruct, sciq, camel, and perfect-blend-gptoss-20B datasets for demo purpose, if you wish to use other datasets, please modify this script."
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
