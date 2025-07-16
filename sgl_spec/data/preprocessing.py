import os
import warnings
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset as HFDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from .template import TEMPLATE_REGISTRY, ChatTemplate

# define a type called conversation
Conversation = List[Dict[str, str]]


# ==============================
# This file is for preprocessing the data
# ==============================
def preprocess_conversations(
    tokenizer: PreTrainedTokenizer,
    conversations: List[Conversation],
    chat_template: ChatTemplate,
    max_length: int = 2048,
) -> Dict[str, List[torch.Tensor]]:
    """Preprocess a batch of ShareGPT style conversations."""
    system_prompt = chat_template.system_prompt
    user_message_separator = (
        f"{chat_template.end_of_turn_token}{chat_template.user_header}"
    )
    assistant_message_separator = (
        f"{chat_template.end_of_turn_token}{chat_template.assistant_header}"
    )

    # prepare result
    results = {"input_ids": [], "loss_mask": [], "attention_mask": []}

    for source in conversations:
        messages = [{"role": "system", "content": system_prompt}]
        if not source:
            # if the source is None, skip it
            continue

        if source[0]["role"] != "user":
            # if the first message is not from user, skip it
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
        turns = conversation.split(user_message_separator)
        turns[1] = turns[0] + user_message_separator + turns[1]
        turns = turns[1:]

        cur_len = 1
        loss_mask = torch.ones_like(input_ids)
        loss_mask[:cur_len] = 0
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(assistant_message_separator)
            if len(parts) != 2:
                break
            parts[0] += assistant_message_separator
            instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i == 0:
                loss_mask[cur_len : cur_len + instruction_len - 2] = 0
            else:
                loss_mask[cur_len - 3 : cur_len + instruction_len + 1] = 0
            cur_len += turn_len
            if i != 0:
                cur_len += 3

        loss_mask[cur_len:] = 0
        results["input_ids"].append(input_ids[None, :])
        results["loss_mask"].append(loss_mask[None, :])
        results["attention_mask"].append(torch.ones_like(loss_mask)[None, :])
    return results


def build_eagle3_dataset(
    dataset: HFDataset,
    tokenizer: PreTrainedTokenizer,
    chat_template: str,
    max_length: Optional[int] = 2048,
    shuffle_seed: Optional[int] = 0,
    num_proc: Optional[int] = 8,
    cache_dir: Optional[str] = None,
    cache_key: Optional[str] = None,
):
    # Get chat template
    assert (
        chat_template in TEMPLATE_REGISTRY.get_all_template_names()
    ), f"Chat template {chat_template} not found in TEMPLATE_REGISTRY, you may need to register it first"
    template: ChatTemplate = TEMPLATE_REGISTRY.get(chat_template)

    dataset = dataset.shuffle(seed=shuffle_seed)
    original_cols = dataset.column_names

    def preprocess_function(examples):
        # Always do preprocessing
        processed = preprocess_conversations(
            tokenizer,
            examples["conversations"],
            template,
            max_length,
        )
        return processed

    # Process dataset only once
    if cache_dir and cache_key:
        load_from_cache_file = True
        os.makedirs(cache_dir, exist_ok=True)
        cache_file_name = os.path.join(cache_dir, f"{cache_key}.pkl")
        print(f"dataset is cached at {cache_file_name}")
    elif cache_dir is None and cache_key is None:
        load_from_cache_file = False
        cache_file_name = None
        print(f"dataset is not cached")
    else:
        warnings.warn(
            f"cache_dir and cache_key must be provided together to make caching work"
        )

    dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_cols,
        load_from_cache_file=load_from_cache_file,
        cache_file_name=cache_file_name,
    )

    dataset.set_format(type="torch")
    return dataset


# ==============================
# Vocab Mapping
# ==============================
def generate_vocab_mapping_file(
    dataset: HFDataset,
    target_vocab_size: int,
    draft_vocab_size: int,
    cache_dir: str = "./cache/vocab_mapping",
    cache_key: str = "vocab_mapping",
):
    # prepare cache direcotory
    os.makedirs(cache_dir, exist_ok=True)
    vocab_mapping_path = os.path.join(cache_dir, f"{cache_key}.pkl")

    if os.path.exists(vocab_mapping_path):
        print(f"Loading vocab mapping from the cached file at: {vocab_mapping_path}")
        return vocab_mapping_path

    # we first count the frequency of effectiev tokens in the dataset
    token_dict = Counter()
    for item in tqdm(dataset, desc="Counting tokens for vocab mapping"):
        input_ids = item["input_ids"]
        loss_mask = item["loss_mask"]
        masked_ids = input_ids[loss_mask == 1]
        unique_ids, counts = masked_ids.unique(return_counts=True)
        batch_token_dict = dict(zip(unique_ids.tolist(), counts.tolist()))
        token_dict.update(batch_token_dict)

    # generate the d2t and t2d mapping
    d2t, t2d = process_token_dict_to_mappings(
        token_dict,
        draft_vocab_size,
        target_vocab_size,
    )

    vocab_mapping = {
        "d2t": d2t,
        "t2d": t2d,
    }
    torch.save(vocab_mapping, vocab_mapping_path)
    print(f"Saved vocab mapping to: {vocab_mapping_path}")
    return vocab_mapping_path


def process_token_dict_to_mappings(
    token_dict: Counter,
    draft_vocab_size: int,
    target_vocab_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process token_dict to create d2t and t2d mappings, with optional caching.
    """
    total_frequency = sum(token_dict.values())
    top_N = token_dict.most_common(draft_vocab_size)
    top_N_frequency_sum = sum(freq for key, freq in top_N)
    top_N_ratio = top_N_frequency_sum / total_frequency
    print(f"top {draft_vocab_size} token frequency ratio: {top_N_ratio:.2%}")

    used_tokens = [key for key, freq in top_N]
    used_tokens.sort()
    used_tokens_set = set(used_tokens)

    # Create d2t mapping: draft token index -> target token index
    d2t = torch.tensor(
        [used_tokens[i] - i for i in range(len(used_tokens))], dtype=torch.long
    )

    # Create t2d mapping: target token index -> boolean (whether in draft vocab)
    t2d = torch.tensor(
        [i in used_tokens_set for i in range(target_vocab_size)], dtype=torch.bool
    )
    return d2t, t2d
