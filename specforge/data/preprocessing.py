# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in HuggingFace Transformers.
# Portions of this code are adapted from:
#   - https://github.com/EleutherAI/gpt-neox (Apache License 2.0)
#   - https://github.com/huggingface/transformers (Apache License 2.0)
#   - https://github.com/SafeAILab/EAGLE (Apache License 2.0)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import warnings
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset as HFDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from specforge.utils import padding

from .template import TEMPLATE_REGISTRY, ChatTemplate

# define a type called conversation
Conversation = List[Dict[str, str]]


# ==============================
# This file is for preprocessing the data
# ==============================
# Copied from https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/cnets.py
def preprocess_conversations(
    tokenizer: PreTrainedTokenizerFast,
    conversations: List[Conversation],
    chat_template: ChatTemplate,
    max_length: int = 2048,
) -> Dict[str, List[torch.Tensor]]:
    """
    Preprocess a batch of ShareGPT style conversations.

    Args:
        tokenizer: The tokenizer to use for tokenization.
        conversations: A list of conversations, where each conversation is a list of messages.
        chat_template: The chat template to use for formatting the conversations.
        max_length: The maximum length of the tokenized input.

    Returns:
        A dictionary containing:
            - input_ids: List of tokenized input IDs.
            - loss_mask: List of loss masks indicating which tokens should contribute to the loss.
            - attention_mask: List of attention masks.
    """
    system_prompt = chat_template.system_prompt
    # This template is only suitable for other models. For kimi_k2, use the modified conversation template.
 
    # user_message_separator = (
    #     f"{chat_template.end_of_turn_token}{chat_template.user_header}"
    # )
    # assistant_message_separator = (
    #     f"{chat_template.end_of_turn_token}{chat_template.assistant_header}"
    # )

    user_message_separator = (
        chat_template.user_header
    )
    assistant_message_separator = (
        chat_template.assistant_header
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
            return_offsets_mapping=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoding.input_ids[0]
        offsets = encoding.offset_mapping[0]
        loss_mask = torch.zeros(len(input_ids), dtype=torch.long)

        # Find spans of assistant responses using regex
        assistant_pattern = (
            re.escape(assistant_message_separator)
            + r"(.*?)(?="
            + re.escape(user_message_separator)
            + "|$)"
        )
        for match in re.finditer(assistant_pattern, conversation, re.DOTALL):
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

        results["input_ids"].append(input_ids[None, :])
        results["loss_mask"].append(loss_mask[None, :])
        results["attention_mask"].append(torch.ones_like(loss_mask)[None, :])
    return results


def build_eagle3_dataset(
    dataset: HFDataset,
    tokenizer: PreTrainedTokenizer,
    chat_template: str,
    max_length: Optional[int] = 2048,
    shuffle_seed: Optional[int] = 42,
    num_proc: Optional[int] = 8,
    cache_dir: Optional[str] = None,
    cache_key: Optional[str] = None,
) -> HFDataset:
    """
    build eagle3 dataset

    Args:
        dataset: HF dataset to process.
        tokenizer: The tokenizer to use for tokenization.
        chat_template: The chat template to use for formatting the conversations.
        max_length: The maximum length of the tokenized input.
        shuffle_seed: The seed for shuffling the dataset.
        num_proc: The number of processes to use for multiprocessing.
        cache_dir: The directory to use for caching the processed dataset.
        cache_key: The key to use for caching the processed dataset.

    Returns:
        The processed HF dataset.
    """
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
# Offline Eagle3 Dataset
# ==============================
# modified from https://github.com/NickL77/BaldEagle/blob/master/train/modules/data/data.py
def list_local_files(path, suffixes=[".ckpt"]):
    datapaths = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapaths.append(file_path)
    for suffix in suffixes:
        datapaths = [f_name for f_name in datapaths if f_name.endswith(suffix)]
    return datapaths


class OfflineEagle3Dataset(torch.utils.data.Dataset):
    def __init__(self, datapath, transform=None, max_len=2048):
        self.datapaths = datapath
        self.transform = transform
        self._epoch = 0
        self.max_len = max_len

    def __len__(self):
        return len(self.datapaths)

    def _open_file(self, index):
        return torch.load(self.datapaths[index], weights_only=False)

    def __getitem__(self, index):
        try:
            data = self._open_file(index)
        except Exception as e:
            print(f"ERROR Failed to load {self.datapaths[index]} with error {e}")
            data = self._open_file(0)
            # raise e
        new_data = {}

        # Squeeze due to our data generation script adding a batch dimension
        hidden_state = data["aux_hidden_state"].squeeze(0)[: self.max_len][None, :]
        target = data["hidden_state"].squeeze(0)[: self.max_len][None, :]

        input_ids = data["input_ids"][: self.max_len][None, :]
        loss_mask = data["loss_mask"][: self.max_len][None, :]
        loss_mask[0, -1] = 0

        new_data["attention_mask"] = torch.ones_like(loss_mask, dtype=torch.long)
        new_data["loss_mask"] = loss_mask
        new_data["target"] = padding(target, left=False)
        new_data["hidden_state"] = hidden_state
        new_data["input_ids"] = padding(input_ids, left=False)
        if self.transform:
            new_data = self.transform(new_data)

        return new_data

    def set_epoch(self, epoch):
        self._epoch = epoch


def build_offline_eagle3_dataset(
    hidden_states_path: str,
    max_len: int = 2048,
) -> torch.utils.data.Dataset:
    return OfflineEagle3Dataset(
        list_local_files(hidden_states_path),
        max_len=max_len,
    )


# ==============================
# Vocab Mapping
# ==============================
def generate_vocab_mapping_file(
    dataset: HFDataset,
    target_vocab_size: int,
    draft_vocab_size: int,
    cache_dir: str = "./cache/vocab_mapping",
    cache_key: str = "vocab_mapping",
) -> str:
    """
    Generate a vocab mapping file for the dataset.

    Args:
        dataset: The dataset to process.
        target_vocab_size: The target vocabulary size.
        draft_vocab_size: The draft vocabulary size.
        cache_dir: The directory to use for caching the vocab mapping file.
        cache_key: The key to use for caching the vocab mapping file.

    Returns:
        The path to the vocab mapping file.
    """
    # prepare cache direcotory
    os.makedirs(cache_dir, exist_ok=True)
    vocab_mapping_path = os.path.join(cache_dir, f"{cache_key}.pt")

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

    Args:
        token_dict: A Counter object mapping token ids to their frequencies.
        draft_vocab_size: The size of the draft vocabulary.
        target_vocab_size: The size of the target vocabulary.

    Returns:
        A tuple containing:
            - d2t: A tensor mapping draft token ids to target token ids.
            - t2d: A tensor mapping target token ids to draft token ids.
    """
    if len(token_dict) < draft_vocab_size:
        existing_tokens = set(token_dict.keys())
        missing_tokens = set(range(draft_vocab_size)) - existing_tokens
        for token in missing_tokens:
            token_dict[token] = 0
            if len(token_dict) >= draft_vocab_size:
                break

    total_frequency = sum(token_dict.values())
    top_N = token_dict.most_common(draft_vocab_size)
    top_N_frequency_sum = sum(freq for key, freq in top_N)
    top_N_ratio = top_N_frequency_sum / total_frequency

    print(f"top {draft_vocab_size} token frequency ratio: {top_N_ratio:.2%}")
    used_tokens = [key for key, freq in top_N]
    used_tokens.sort()

    d2t = [used_tokens[i] - i for i in range(len(used_tokens))]
    t2d = [i in used_tokens for i in range(target_vocab_size)]
    d2t = torch.tensor(d2t)
    t2d = torch.tensor(t2d)

    return d2t, t2d
