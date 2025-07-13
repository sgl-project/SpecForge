import re
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset

from sgl_spec.data.config import DataConfig


@contextmanager
def rank_0_priority():
    rank = dist.get_rank()

    if rank == 0:
        yield
        dist.barrier()
    else:
        dist.barrier()
        yield

def preprocess_conversations(
    tokenizer,
    conversations,
    return_attention_mask=True,
    *,
    system_prompt: Optional[str] = None,
    assistant_header: Optional[str] = None,
    user_header: Optional[str] = None,
    max_length=2048,
    config: Optional[DataConfig] = None,
):
    """Preprocess a batch of ShareGPT style conversations."""
    # Get chat template from config if not provided
    if config is None:
        config = DataConfig()

    template = config.get_chat_template()

    # Use provided parameters or fall back to config template
    if system_prompt is None:
        system_prompt = template["system_prompt"]
    if assistant_header is None:
        assistant_header = template["assistant_header"]
    if user_header is None:
        user_header = template["user_header"]

    results = {"input_ids": [], "loss_mask": []}
    if return_attention_mask:
        results["attention_mask"] = []

    for source in conversations:
        messages = [{"role": "system", "content": system_prompt}]
        if not source:
            continue
        if source[0]["role"] != "user":
            source = source[1:]

        convroles = ["user", "assistant"]
        for j, sentence in enumerate(source):
            role = sentence["role"]
            assert role == convroles[j % 2], f"unexpected role {role}"
            messages.append({"role": role, "content": sentence["content"]})

        conversation = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.unk_token_id

        encoding = tokenizer(
            conversation,
            return_tensors="pt",
            max_length=max_length,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=True,
        )

        ids = encoding.input_ids[0]
        offsets = encoding.offset_mapping[0]
        
        # 模拟原始代码：初始化为全1，然后逐步设为0
        mask = torch.ones_like(ids)
        
        # 对应原始代码：cur_len = 1, loss_mask[:cur_len] = 0
        mask[0] = 0  # 跳过第0个token

        # 模拟原始代码的turn处理逻辑
        sep = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        sep2 = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
        turns = conversation.split(sep2)
        
        if len(turns) > 1:
            turns[1] = turns[0] + sep2 + turns[1]
            turns = turns[1:]
            
            cur_len = 1  # 对应原始代码的起始位置
            
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                    
                turn_len = len(tokenizer(turn).input_ids)
                parts = turn.split(sep)
                
                if len(parts) != 2:
                    break
                    
                parts[0] += sep
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1
                
                # 模拟原始代码的边界处理
                if i == 0:
                    mask[cur_len: cur_len + instruction_len - 2] = 0
                else:
                    mask[cur_len - 3: cur_len + instruction_len + 1] = 0
                    
                cur_len += turn_len
                if i != 0:
                    cur_len += 3
                    
            # 关键：模拟原始代码的结尾处理
            mask[cur_len:] = 0

        results["input_ids"].append(ids[None, :])
        results["loss_mask"].append(mask[None, :])

        if return_attention_mask:
            results["attention_mask"].append(torch.ones_like(mask)[None, :])
        
    return results