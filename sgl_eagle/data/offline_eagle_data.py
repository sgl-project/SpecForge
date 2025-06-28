"""
usage:
python generate_data.py --outdir /path/to/outdir --model_path /path/to/model --dataset sharegpt
"""

import argparse
import multiprocessing
import os
import re
from itertools import accumulate
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer  # no need to format

DATASET_INFO = {
    "sharegpt": {
        "num_samples": 68000 - 1,
        "hf_name": "Aeala/ShareGPT_Vicuna_unfiltered",
        "split": "train",
    },
    "ultrachat": {
        "num_samples": 200000 - 1,
        "hf_name": "HuggingFaceH4/ultrachat_200k",
        "split": "train_sft",
    },
    "mixture_of_thoughts": {
        "num_samples": 100000 - 1,
        "hf_name": "open-r1/Mixture-of-Thoughts",
        "split": "all",
    },
}

system_message = {
    "role": "system",
    "content": "You are a helpful, respectful and honest assistant.",
}


def split_range(start, end, n, over=False):
    sizes = [
        (end - start + 1) // n + (1 if i < (end - start + 1) % n else 0)
        for i in range(n)
    ]
    boundaries = list(accumulate([start] + sizes))
    if over:  # the sub range is overlapped with the previous one
        return [(boundaries[i], boundaries[i + 1]) for i in range(n)]
    else:
        return [(boundaries[i], boundaries[i + 1] - 1) for i in range(n)]


def format_conversation_sharegpt(row, dataset_column="conversations"):
    messages = [system_message]
    current_role = None
    for message in row[dataset_column]:
        if message["from"] == "human":
            messages.append({"role": "user", "content": message["value"]})
        elif message["from"] == "gpt":
            messages.append({"role": "assistant", "content": message["value"]})
        else:
            raise ValueError(f"Unknown role: {message['from']}")

        if current_role is None:
            current_role = messages[-1]["role"]
        else:
            assert (
                current_role != messages[-1]["role"]
            ), "Conversation has incorrect role order"
            current_role = messages[-1]["role"]

    return {"messages": messages}


def format_conversation_ultrachat(row, dataset_column="messages"):
    messages = [system_message]
    for message in row[dataset_column]:
        messages.append(message)
    return {"messages": messages}


def tokenize_conversation(
    row, tokenizer, assistant_header, user_header, col="messages"
):
    formatted_conversation = tokenizer.apply_chat_template(
        row[col], tokenize=False, add_generation_prompt=False
    )

    encoding = tokenizer(formatted_conversation, return_offsets_mapping=True)
    input_ids = encoding.input_ids
    offsets = encoding.offset_mapping
    loss_mask = torch.zeros(len(input_ids), dtype=torch.long)

    # Find spans of assistant responses using regex
    assistant_pattern = (
        re.escape(assistant_header) + r"(.*?)(?=" + re.escape(user_header) + "|$)"
    )
    for match in re.finditer(assistant_pattern, formatted_conversation, re.DOTALL):
        # Assistant response text span (excluding assistant_header itself)
        assistant_start_char = match.start(1)
        assistant_end_char = match.end(1)

        # Mark tokens overlapping with assistant response
        for idx, (token_start, token_end) in enumerate(offsets):
            # Token is part of the assistant response span
            if token_end <= assistant_start_char:
                continue  # token before assistant text
            if token_start > assistant_end_char:
                continue  # token after assistant text
            loss_mask[idx] = 1

    return {
        "conversation_str": formatted_conversation,
        "input_ids": input_ids,
        "loss_mask": loss_mask,
    }


def generate_features(start, end, gpu_index, args):
    torch.cuda.set_device(gpu_index)
    dataset_info = DATASET_INFO[args.dataset]
    print(
        f"Generating data for {args.dataset} from {start} to {end} on GPU {gpu_index}, saving to {args.outdir}/{gpu_index}",
        flush=True,
    )
    dataset = load_dataset(dataset_info["hf_name"], split=dataset_info["split"])
    dataset = dataset.select(range(start, end))
    dataset = dataset.shuffle(seed=42)
    if args.dataset == "sharegpt":
        dataset = dataset.map(format_conversation_sharegpt)
    elif args.dataset == "ultrachat":
        dataset = dataset.map(format_conversation_ultrachat)

    if "llama" in args.model_path.lower():
        assistant_header = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_header = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
    elif "qwen" in args.model_path.lower():
        assistant_header = "<|im_start|>assistant\n"
        user_header = "<|im_start|>user\n"
    else:
        raise ValueError(f"Model name {args.model_path} not supported")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dataset = dataset.map(
        tokenize_conversation,
        fn_kwargs={
            "tokenizer": tokenizer,
            "assistant_header": assistant_header,
            "user_header": user_header,
        },
    )
    dataset = dataset.remove_columns(
        [
            col
            for col in dataset.column_names
            if col not in ["input_ids", "loss_mask", "conversation_str"]
        ]
    )
    dataset.set_format(type="torch")

    # multiple GPUs loading model from TCP at the the same time might cause issues
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map="cuda", torch_dtype=torch.bfloat16, local_files_only=True
    )
    model.eval()

    outdir = os.path.join(args.outdir, f"{gpu_index}")
    os.makedirs(outdir, exist_ok=True)

    num_layers = len(model.model.layers)
    low_layer_idx = 2
    mid_layer_idx = num_layers // 2
    high_layer_idx = num_layers - 3

    for idx, row in tqdm(enumerate(dataset), total=end - start):
        group_size = 5000
        start = (idx // group_size) * group_size
        end = start + group_size
        grouped_subdir = os.path.join(outdir, f"rows_{start}-{end}")
        output_file = os.path.join(grouped_subdir, f"data_{idx}.ckpt")
        os.makedirs(grouped_subdir, exist_ok=True)
        if os.path.exists(output_file):
            continue

        with torch.no_grad():
            outputs = model(
                row["input_ids"].unsqueeze(0)[:, : args.max_token_length].cuda(),
                output_hidden_states=True,
            )
            if args.enable_fused_features:
                low_layer = outputs.hidden_states[low_layer_idx].cpu()
                mid_layer = outputs.hidden_states[mid_layer_idx].cpu()
                high_layer = outputs.hidden_states[high_layer_idx].cpu()
                hidden_states = torch.concat([low_layer, mid_layer, high_layer], dim=2)
                target_hidden_states = outputs.hidden_states[-1].cpu()
            else:
                hidden_states = outputs.hidden_states[-1].cpu()
                target_hidden_states = None

        data_point = {
            "input_ids": row["input_ids"],
            "loss_mask": row["loss_mask"],
            "hidden_state": hidden_states,
        }
        if target_hidden_states is not None:
            data_point["target_hidden_states"] = target_hidden_states
        torch.save(data_point, output_file)


# this is copy from EAGLE3 official code
def generate_data(args, dataset_name: str, start: Optional[int] = None, end: Optional[int] = None):
    assert (
        hasattr(args, "model_path") and args.model_path is not None
    ), "model_path is required for args"
    dataset_info = DATASET_INFO[dataset_name]
    dataset = load_dataset(dataset_info["hf_name"], split=dataset_info["split"])
    if start is not None and end is not None:
        dataset = dataset.select(range(start, end))
    dataset = dataset.shuffle(seed=42)
    if args.dataset == "sharegpt":
        dataset = dataset.map(format_conversation_sharegpt)
    elif args.dataset == "ultrachat":
        dataset = dataset.map(format_conversation_ultrachat)

    if "llama" in args.model_path.lower():
        assistant_header = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        user_header = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
    elif "qwen" in args.model_path.lower():
        assistant_header = "<|im_start|>assistant\n"
        user_header = "<|im_start|>user\n"
    else:
        raise ValueError(f"Model name {args.model_path} not supported")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dataset = dataset.map(
        tokenize_conversation,
        fn_kwargs={
            "tokenizer": tokenizer,
            "assistant_header": assistant_header,
            "user_header": user_header,
        },
    )
    dataset = dataset.remove_columns(
        [
            col
            for col in dataset.column_names
            if col not in ["input_ids", "loss_mask", "conversation_str"]
        ]
    )
    dataset.set_format(type="torch")
    return dataset
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-token-length", type=int, default=2048)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["sharegpt", "ultrachat", "mixture_of_thoughts"],
        default="sharegpt",
    )
    parser.add_argument(
        "--enable-fused-features",
        action="store_true",
        help="enable fused features for eagle3",
    )
    args = parser.parse_args()
    num_gpus = torch.cuda.device_count()  # can be set by env "CUDA_VISIBLE_DEVICES"
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Saving to {args.outdir}", flush=True)

    s, e = 0, DATASET_INFO[args.dataset]["num_samples"]
    data_a = split_range(s, e, num_gpus, over=True)
    workers = []
    for gpu_index, (start, end) in enumerate(data_a):
        proc = multiprocessing.Process(
            target=generate_features,
            args=(
                start,
                end,
                gpu_index,
                args,
            ),
        )
        proc.start()
        workers.append(proc)

    for proc in workers:
        proc.join()
