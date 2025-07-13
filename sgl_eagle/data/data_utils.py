from contextlib import contextmanager
import torch.distributed as dist
import torch
import re
import numpy as np
from datasets import Dataset
from sgl_eagle.data.config import DataConfig
from typing import Optional


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as "
    "helpfully as possible, while being safe.  Your answers should not "
    "include any harmful, unethical, racist, sexist, toxic, dangerous, or "
    "illegal content. Please ensure that your responses are socially unbiased "
    "and positive in nature.\n\nIf a question does not make any sense, or is "
    "not factually coherent, explain why instead of answering something not "
    "correct. If you don't know the answer to a question, please don't share "
    "false information."
)

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
    """Preprocess a batch of ShareGPT style conversations.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer used to convert text to tokens.
    conversations : list
        List of conversation items, each a list of {"role", "content"} dicts.
    return_attention_mask : bool, optional
        Whether to also return attention masks.
    system_prompt : str, optional
        System prompt prepended to every conversation. If None, use config template.
    assistant_header : str, optional
        Token sequence that marks the assistant role in the conversation. If None, use config template.
    user_header : str, optional
        Token sequence that marks the user role in the conversation. If None, use config template.
    max_length : int, optional
        Maximum sequence length.
    config : DataConfig, optional
        Configuration object containing chat template settings.

    Returns
    -------
    dict
        Dictionary containing lists of ``input_ids`` and ``loss_mask``. If
        ``return_attention_mask`` is True an ``attention_mask`` list is also
        included.
    """
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
        mask = torch.zeros_like(ids)

        pattern = (
            re.escape(assistant_header) + r"(.*?)(?=" + re.escape(user_header) + "|$)"
        )

        # Vectorized mask construction
        matches = list(re.finditer(pattern, conversation, re.DOTALL))
        if matches:
            match_starts = np.array([m.start(1) for m in matches])
            match_ends = np.array([m.end(1) for m in matches])
            
            token_starts = offsets[:, 0].numpy()
            token_ends = offsets[:, 1].numpy()
            
            overlaps = (token_ends[:, None] > match_starts[None, :]) & \
                      (token_starts[:, None] < match_ends[None, :])
            
            mask = torch.tensor(overlaps.any(axis=1), dtype=torch.long)

        results["input_ids"].append(ids[None, :])
        results["loss_mask"].append(mask[None, :])
        if return_attention_mask:
            results["attention_mask"].append(torch.ones_like(mask)[None, :])

    return results
