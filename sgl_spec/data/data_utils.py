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

        input_ids = encoding.input_ids[0]
        
        turns = conversation.split(user_header)

        turns[1] = turns[0] + user_header + turns[1]
        turns = turns[1:]

        cur_len = 1
        loss_mask = torch.ones_like(input_ids)
        loss_mask[:cur_len] = 0
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(assistant_header)
            if len(parts) != 2:
                break
            parts[0] += assistant_header
            instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i == 0:
                loss_mask[cur_len: cur_len + instruction_len - 2] = 0
            else:
                loss_mask[cur_len - 3: cur_len + instruction_len + 1] = 0
            cur_len += turn_len
            if i != 0:
                cur_len += 3


        loss_mask[cur_len:] = 0


        results["input_ids"].append(input_ids[None, :])
        results["loss_mask"].append(loss_mask[None, :])

        if return_attention_mask:
            results["attention_mask"].append(torch.ones_like(loss_mask)[None, :])

    return results