import os
import torch
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer
from sgl_eagle.data.data_utils import preprocess_conversations

user_template = "<|header_start|>user<|header_end|>"
assistant_template = "<|header_start|>assistant<|header_end|>\n\n"

def do_word_count(dataset, num_processes=8):
    def _process_data(rows):
        input_ids = torch.cat([torch.LongTensor(i) for i in rows["input_ids"]], dim=-1).view(-1)
        loss_mask = torch.cat([torch.LongTensor(i) for i in rows["loss_mask"]], dim=-1).view(-1)
        masked_ids = input_ids[loss_mask == 1]
        unique_ids, counts = masked_ids.unique(return_counts=True)
        return {
            "unique_ids": [unique_ids.tolist()],
            "counts": [counts.tolist()]
        }
    new_ds = dataset.map(
        _process_data,
        batched=True,
        num_proc=num_processes,
        remove_columns=dataset.column_names,
    )
    token_dict = Counter()
    for row in new_ds:
        if len(row["unique_ids"]) > 0:
            token_dict.update(dict(zip(row["unique_ids"], row["counts"])))
    return token_dict


def build_dataset_rank(tokenizer=None,ds=None, assistant_header: str = assistant_template, user_header: str = user_template, max_length: int = 2048):
    
    ds = ds.shuffle(seed=42)
    original_cols = ds.column_names
    num_proc = 8
    def preprocess_function(examples):
        return preprocess_conversations(
            tokenizer,
            examples["conversations"],
            return_attention_mask=False,
            assistant_header=assistant_header,
            user_header=user_header,
            max_length=max_length)
    dataset = ds.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_cols,
        load_from_cache_file=False,
    )
    token_dict=do_word_count(dataset)
    ds.set_format(type="torch")
    return ds,token_dict