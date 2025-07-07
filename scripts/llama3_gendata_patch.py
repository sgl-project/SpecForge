import argparse
import os
import re

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(description="sp")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=100)
parser.add_argument("--index", type=int, default=1)
parser.add_argument("--gpu_index", type=int, nargs="+", default=[0])
parser.add_argument("--outdir", type=str, default="/root/.cache/llama3/outdir0")
parser.add_argument(
    "--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct"
)  # "meta-llama/Meta-Llama-3.1-8B-Instruct"
parser.add_argument(
    "--dataset",
    type=str,
    choices=["sharegpt", "ultrachat", "mixture_of_thoughts"],
    default="sharegpt",
)
parser.add_argument("--group_size", type=int, default=50, help="Number of records per chunk (default: 50)")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]

max_token_length = 2048

# ------------------------ 1. Dataset ------------------------
# This step converts the dataset into a standard messages format

if args.dataset == "sharegpt":
    dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
elif args.dataset == "ultrachat":
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
elif args.dataset == "mixture_of_thoughts":
    dataset = load_dataset("open-r1/Mixture-of-Thoughts", "all", split="all")

dataset = dataset.select(range(args.start, args.end))
dataset = dataset.shuffle(seed=42)

# System message that will be prepended to all conversations
system_message = {
    "role": "system",
    "content": "You are a helpful, respectful and honest assistant.",
}


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
            ), f"Conversation has incorrect role order"
            current_role = messages[-1]["role"]

    return {"messages": messages}


def format_conversation_ultrachat(row, dataset_column="messages"):
    messages = [system_message]
    for message in row[dataset_column]:
        messages.append(message)
    return {"messages": messages}


if args.dataset == "sharegpt":
    dataset = dataset.map(format_conversation_sharegpt)
elif args.dataset == "ultrachat":
    dataset = dataset.map(format_conversation_ultrachat)
elif args.dataset == "mixture_of_thoughts":
    pass  # no need to format

# ------------------------ 2. Tokenizer ------------------------
# This step tokenizes the conversation and creates the loss mask
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# Special token sequences used to identify different parts of the conversation
# For Llama models
assistant_header = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
user_header = "<|eot_id|><|start_header_id|>user<|end_header_id|>"

# For Qwen models
# assistant_header = "<|im_start|>assistant\n"
# user_header = "<|im_start|>user\n"


def tokenize_conversation(row, tokenizer, col="messages"):
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


dataset = dataset.map(tokenize_conversation, fn_kwargs={"tokenizer": tokenizer})
dataset = dataset.remove_columns(
    [
        col
        for col in dataset.column_names
        if col not in ["input_ids", "loss_mask", "conversation_str"]
    ]
)
dataset.set_format(type="torch")

# ------------------------ 3. Compute hidden states ------------------------

model = AutoModelForCausalLM.from_pretrained(
    args.model_name, device_map="cuda", torch_dtype=torch.bfloat16
)
model.eval()

outdir = f"{args.outdir}/{args.index}"
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Define low, mid, and high layer indices based on https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/modeling_llama_kv.py#L1137-L1139
num_layers = len(model.model.layers)
low_layer_idx = 2
mid_layer_idx = num_layers // 2
high_layer_idx = num_layers - 3

records = []
group_size = args.group_size

for idx, row in tqdm(enumerate(dataset), total=args.end - args.start):
    start = (idx // group_size) * group_size
    end = start + group_size
    if not os.path.exists(f"{outdir}"):
        os.makedirs(f"{outdir}")

    try:
        with torch.no_grad():
            outputs = model(
                row["input_ids"].unsqueeze(0)[:, :max_token_length].cuda(),
                output_hidden_states=True,
            )
            low_layer = outputs.hidden_states[low_layer_idx].cpu()
            mid_layer = outputs.hidden_states[mid_layer_idx].cpu()
            high_layer = outputs.hidden_states[high_layer_idx].cpu()
            hidden_states = torch.concat([low_layer, mid_layer, high_layer], dim=2)
            target_hidden_states = outputs.hidden_states[-1].cpu()
        
        data_point = {
            "input_ids": row["input_ids"],
            "loss_mask": row["loss_mask"],
            "hidden_state": hidden_states.to(torch.float32).numpy(),
            "target_hidden_states": target_hidden_states.to(torch.float32).numpy(),
        }
        records.append(data_point)
        
        # 清理GPU内存
        del outputs, low_layer, mid_layer, high_layer, hidden_states, target_hidden_states
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        continue

    # 每group_size条保存一次
    if (idx + 1) % group_size == 0 or (idx + 1) == (args.end - args.start):
        if records:  # 确保有数据要保存
            try:
                chunk_idx = start // group_size
                dict_data = {k: [rec[k] for rec in records] for k in records[0]}
                ds = Dataset.from_dict(dict_data, writer_batch_size=10)  # 添加writer_batch_size参数
                ds.save_to_disk(f"{outdir}/chunk_{chunk_idx}")
                print(f"Saved chunk {chunk_idx} with {len(records)} records")
            except Exception as e:
                print(f"Error saving chunk {chunk_idx}: {e}")
                # 如果保存失败，尝试减小批次大小
                try:
                    ds = Dataset.from_dict(dict_data, writer_batch_size=1)
                    ds.save_to_disk(f"{outdir}/chunk_{chunk_idx}")
                    print(f"Saved chunk {chunk_idx} with reduced batch size")
                except Exception as e2:
                    print(f"Failed to save chunk {chunk_idx} even with reduced batch size: {e2}")
            finally:
                records = []
