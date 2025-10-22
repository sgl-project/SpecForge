import os
import hashlib
import json
import argparse
import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset 
from transformers import AutoConfig, AutoTokenizer
from datetime import datetime

from specforge.data import build_eagle3_dataset
from specforge.utils import print_with_rank, rank_0_priority

"""
# training data 생성
python postprocess_test.py \
    --data-path /cache/hidden_states_fixed/ \
    --model-path /models/gpt-oss-120b/ \
    --output-path /cache/dump_train_fixed

# evaluation data 생성
python postprocess_test.py \
    --data-path /cache/hidden_states_fixed/ \
    --model-path /models/gpt-oss-120b/ \
    --output-path /cache/dump_eval_fixed \
    --test-mode
"""

def aggregate_text_jsons(input_dir, output_dir, tokenizer):
    aggregated = []
    for fname in os.listdir(input_dir):
        if fname.endswith("_text.json"):
            request_id = fname.split("_text.json")[0]
            with open(os.path.join(input_dir, fname), "r", encoding="utf-8") as f:
                data = json.load(f)

            conversations = []
            if "input_ids" in data and "output_ids" in data:
                #detokenized_text = tokenizer.decode(data["input_ids"] + data["output_ids"])
                input_text = tokenizer.decode(data["input_ids"])
                output_text = tokenizer.decode(data["output_ids"])
                conversations.append({"role": "user", "content": input_text})
                conversations.append({"role": "assistant", "content": output_text})
                text = tokenizer.decode(data["input_ids"] + data["output_ids"])
                aggregated.append({
                    "request_id": request_id,
                    "text": text,
                    "input_ids" : data["input_ids"],
                    "output_ids" : data["output_ids"],
                    "conversations": conversations
                })
            else:
                continue
            
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "aggregated_text.jsonl")
    with open(out_path, "w", encoding="utf-8") as fout:
        for item in aggregated:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved: {out_path}")
    return out_path

# 사용 예시
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate text JSON files")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--chat-template", type=str, default="gpt-oss")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--enable-aux-hidden-states", action="store_true")
    parser.add_argument("--aux-hidden-states-layers", type=str, default=None)
    parser.add_argument("--build-dataset-num-proc", type=int, default=8)
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args()

    assert os.path.exists(
        args.data_path
    ), f"Dataset path {args.data_path} does not exist"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    out_path = aggregate_text_jsons(args.data_path, args.output_path, tokenizer)
    dataset = load_dataset("json", data_files=out_path)["train"]
    total_len = len(dataset)
    ten_percent = int(total_len * 0.1)
    
    if args.test_mode:
        # 마지막 10%만 가져오려면
        dataset = dataset.select(range(total_len - ten_percent, total_len))
    else:
        # 마지막 10%만 가져오려면
        dataset = dataset.select(range(0, total_len - ten_percent))
    print(dataset)
    print("Built dataset")
        
    group_size = 5000
    diff_counter = {0: 0, 1: 0, 2: 0}
    diff_examples = {0: [], 1: [], 2: []}
    for idx, row in tqdm(enumerate(dataset), total=len(dataset)):
        group_start = (idx // group_size) * group_size
        group_end = group_start + group_size
        grouped_subdir = f"rows_{group_start}-{group_end}"
        subdir_path = os.path.join(args.output_path, grouped_subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path, exist_ok=True)

        output_file = os.path.join(subdir_path, f"data_{idx}.ckpt")

        # hidden_states 파일 로드
        request_id = row["request_id"]
        hs_path = os.path.join(args.data_path, f"{request_id}_hidden_states.pt")
        if not os.path.exists(hs_path):
            print(f"skip {hs_path} (not found)")
            continue
        extracted_hidden_states = torch.load(hs_path)  # (N, 4*K)
        seq_len, hidden_dim_times_4 = extracted_hidden_states.shape
        hidden_dim = hidden_dim_times_4 // 4
        
        # 마지막 K: hidden_state, 앞 3K: aux_hidden_state
        hidden_state = extracted_hidden_states[:, -hidden_dim:]
        aux_hidden_state = extracted_hidden_states[:, :-hidden_dim]

        # .view(-1) 및 .unsqueeze(0) 적용
        input_ids_only = torch.tensor(row["input_ids"],dtype=torch.long).view(-1)
        output_ids_only = torch.tensor(row["output_ids"][:-1],dtype=torch.long).view(-1)
        input_ids = torch.cat([input_ids_only, output_ids_only], dim=0)
        loss_mask = torch.zeros_like(input_ids)
        loss_mask[len(input_ids_only):] = 1
        diff = abs(input_ids.shape[0] - seq_len)
        if diff in diff_counter:
            diff_counter[diff] += 1
            if len(diff_examples[diff]) < 10:  # 예시 10개만 저장
                diff_examples[diff].append({
                    "request_id": request_id,
                    "input_ids_len": int(input_ids.shape[0]),
                    "hidden_states_len": int(seq_len),
                    "diff": int(diff),
                    "hs_path": hs_path
                })
        else:
            diff_counter[diff] = 1
            diff_examples[diff] = [{
                "request_id": request_id,
                "input_ids_len": int(input_ids.shape[0]),
                "hidden_states_len": int(seq_len),
                "diff": int(diff),
                "hs_path": hs_path
            }]
        
        hidden_state = hidden_state.unsqueeze(0).cpu()
        aux_hidden_state = aux_hidden_state.unsqueeze(0).cpu()
        # assert 대신에 warning 메세지 출력하고 저장은 하지 않는 것으로 해줄래?
        if input_ids.shape[0] != hidden_state.shape[1]:
            print(f"Warning: input_ids length {input_ids.shape[0]} != hidden_state length {hidden_state.shape[1]}, seq_len: {seq_len}, request_id: {request_id}")
            continue
        # 저장
        save_dict = {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "hidden_state": hidden_state,
            "aux_hidden_state": aux_hidden_state,
        }
        torch.save(save_dict, output_file)
        
    # 결과를 json으로 저장
    diff_report = {
        "diff_counter": diff_counter,
        "diff_examples": diff_examples
    }
    with open(os.path.join(args.output_path, "input_hidden_length_diff_report.json"), "w", encoding="utf-8") as f:
        json.dump(diff_report, f, ensure_ascii=False, indent=2)
    print(f"Diff report saved to {os.path.join(args.output_path, 'input_hidden_length_diff_report.json')}")