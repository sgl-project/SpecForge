import os
import json
import torch
from transformers import AutoTokenizer

base_dir = "/cache/hidden_states_fixed"
tokenizer = AutoTokenizer.from_pretrained("/models/gpt-oss-120b")
count = 0
for fname in os.listdir(base_dir):
    if fname.endswith("_text.json"):
        request_id = fname.replace("_text.json", "")
        text_path = os.path.join(base_dir, fname)
        pt_path = os.path.join(base_dir, f"{request_id}_hidden_states.pt")

        # 텍스트 파일 읽기 및 토크나이즈
        with open(text_path, "r", encoding="utf-8") as f:
            text = json.load(f)

        input_ids = text.get("input_ids", [])
        output_ids = text.get("output_ids", [])
        input_texts_len = len(input_ids)
        output_texts_len = len(output_ids)-1
        input_len = input_texts_len + output_texts_len
        # 히든스테이트 텐서 읽기
        if os.path.exists(pt_path):
            hidden_states = torch.load(pt_path)
            hs_len = hidden_states.shape[0]
            if input_len != hs_len:
                count += 1
                print(f"[Match: {input_len == hs_len}] {request_id}: input_text_len = {input_texts_len}, output_text_len = {output_texts_len}, input_ids len = {input_len}, hidden_states.shape[0] = {hs_len}")
        else:
            print(f"{request_id}: {pt_path} not found")

print(f"Total unmatched cases: {count}")