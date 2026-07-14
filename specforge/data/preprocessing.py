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

import gzip
import io
import json
import os
import warnings
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizer

from .parse import GeneralParser, HarmonyParser, ThinkingParser
from .template import TEMPLATE_REGISTRY, ChatTemplate

# define a type called conversation
Conversation = List[Dict[str, str]]


# ==============================
# This file is for preprocessing the data
# ==============================


# Copied from https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/cnets.py
def preprocess_conversations(
    tokenizer: PreTrainedTokenizer,
    conversations: Union[List[Conversation], List[str]],
    chat_template: ChatTemplate,
    max_length: int = 2048,
    is_preformatted: bool = False,
    train_only_last_turn: bool = False,
    tools: Optional[List[List[Dict]]] = [[]],
    **kwargs,
) -> Dict[str, List[torch.Tensor]]:
    """
    Preprocess a batch of ShareGPT style conversations or pre-formatted text.

    Args:
        tokenizer: The tokenizer to use for tokenization.
        conversations: A list of conversations (if is_preformatted=False) or
                      a list of pre-formatted text strings (if is_preformatted=True).
        chat_template: The chat template to use for formatting/identifying spans.
        max_length: The maximum length of the tokenized input.
        is_preformatted: Whether the input is already formatted text strings.
        train_only_last_turn: If True, only the last assistant turn contributes to the loss.
        tools: Optional list of tools information corresponding to each conversation, used for tool-use conversations.

    Returns:
        A dictionary containing:
            - input_ids: List of tokenized input IDs.
            - loss_mask: List of loss masks indicating which tokens should contribute to the loss.
            - attention_mask: List of attention masks.
    """
    # prepare result
    results = {"input_ids": [], "loss_mask": [], "attention_mask": []}
    if chat_template.parser_type == "general":
        parser = GeneralParser(tokenizer, chat_template)
    elif chat_template.parser_type == "thinking":
        parser = ThinkingParser(tokenizer, chat_template)
    elif chat_template.parser_type == "openai-harmony":
        parser = HarmonyParser(tokenizer, chat_template)
    else:
        raise ValueError(f"Invalid parser type: {chat_template.parser_type}")
    kwargs_list = [{} for _ in range(len(conversations))]
    for key, value_list in kwargs.items():
        for i, value in enumerate(value_list):
            kwargs_list[i][key] = value
    for source, tool, kwargs_item in zip(conversations, tools, kwargs_list):
        if not source:
            # if the source is None, skip it
            continue
        input_ids, loss_mask = parser.parse(
            source,
            max_length,
            preformatted=is_preformatted,
            train_only_last_turn=train_only_last_turn,
            tool=tool,
            **kwargs_item,
        )
        results["input_ids"].append(input_ids[None, :])
        results["loss_mask"].append(loss_mask[None, :])
        results["attention_mask"].append(torch.ones_like(loss_mask)[None, :])
    return results


def build_eagle3_dataset(
    dataset: HFDataset,
    tokenizer: PreTrainedTokenizer,
    chat_template: Optional[str] = None,
    max_length: Optional[int] = 2048,
    shuffle_seed: Optional[int] = 42,
    num_proc: Optional[int] = 8,
    cache_dir: Optional[str] = None,
    cache_key: Optional[str] = None,
    is_preformatted: Optional[bool] = False,
    train_only_last_turn: Optional[bool] = False,
    minimum_valid_tokens: Optional[int] = None,
) -> HFDataset:
    """
    build eagle3 dataset

    Args:
        dataset: HF dataset to process.
        tokenizer: The tokenizer to use for tokenization.
        chat_template: The chat template to use for formatting conversations.
                        This includes the system prompt and user/assistant tokens
                        required to delineate different parts of the conversation
                        for loss mask generation.
        max_length: The maximum length of the tokenized input.
        shuffle_seed: The seed for shuffling the dataset.
        num_proc: The number of processes to use for multiprocessing.
        cache_dir: The directory to use for caching the processed dataset.
        cache_key: The key to use for caching the processed dataset.
        is_preformatted: Whether the dataset contains preformatted text of the conversation
                        (e.g. includes system prompt, user and assistant start and end tokens)
                        and doesn't need to have the chat template applied.
                        Note that the chat_template still needs to be specified to determine
                        the assistant spans for loss mask generation.
                        If True, expects "text" column with ready-to-train text.
                        If False, expects "conversations" column with ShareGPT format.
        train_only_last_turn: If True, only the last assistant turn contributes to the loss.
                             Useful for thinking models where history may not contain thoughts.
        minimum_valid_tokens: If set, drops samples with fewer trainable tokens.

    Returns:
        The processed HF dataset.
    """
    if minimum_valid_tokens is not None and minimum_valid_tokens < 0:
        raise ValueError("minimum_valid_tokens must be >= 0")

    # Validate chat_template requirement
    if chat_template is None:
        raise ValueError("chat_template must be provided for all dataset types")

    assert (
        chat_template in TEMPLATE_REGISTRY.get_all_template_names()
    ), f"Chat template {chat_template} not found in TEMPLATE_REGISTRY, you may need to register it first"

    template: ChatTemplate = TEMPLATE_REGISTRY.get(chat_template)

    dataset = dataset.shuffle(seed=shuffle_seed)
    original_cols = dataset.column_names

    def preprocess_function(examples):
        # Handle different dataset formats
        if is_preformatted:
            # Handle pre-formatted text (should be in "text" column)
            if "text" not in examples:
                raise ValueError(
                    f"Expected 'text' column for is_preformatted=True, but found columns: {list(examples.keys())}"
                )
            processed = preprocess_conversations(
                tokenizer,
                examples["text"],
                template,
                max_length,
                is_preformatted=True,
                train_only_last_turn=train_only_last_turn,
            )
        else:
            # Handle ShareGPT conversations
            if "conversations" not in examples:
                raise ValueError(
                    f"Expected 'conversations' column for is_preformatted=False, but found columns: {list(examples.keys())}"
                )
            conversations = examples.pop("conversations")
            if "id" in examples:
                examples.pop("id")
            if "tools" in examples:
                tools_raw = examples.pop("tools")
                # Parse tools: handle JSON strings from safe_conversations_generator
                tools = []
                for tool_item in tools_raw:
                    if isinstance(tool_item, (str, list)):
                        try:
                            tools.append(json.loads(tool_item))
                        except json.JSONDecodeError:
                            warnings.warn(
                                f"Failed to parse tools JSON string: {tool_item[:100]}..."
                            )
                            tools.append([])
                    elif isinstance(tool_item, list):
                        tools.append(tool_item)
                    elif tool_item is None:
                        tools.append([])
                    else:
                        warnings.warn(
                            f"Unexpected tools type: {type(tool_item)}, using empty list"
                        )
                        tools.append([])
            else:
                tools = [[] for _ in range(len(conversations))]
            processed = preprocess_conversations(
                tokenizer,
                conversations,
                template,
                max_length,
                is_preformatted=False,
                train_only_last_turn=train_only_last_turn,
                tools=tools,
                **examples,
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

    # Disable tokenizers internal parallelism when using multiprocessing to avoid
    # deadlocks caused by forked Rust threads (see huggingface/tokenizers#1391).
    if num_proc is not None and num_proc > 1:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        batch_size=1000,
        remove_columns=original_cols,
        # keep_in_memory=True,
        load_from_cache_file=load_from_cache_file,
        cache_file_name=cache_file_name,
    )

    if minimum_valid_tokens is not None:
        before_filter = len(dataset)

        def has_minimum_valid_tokens(example):
            loss_mask = example["loss_mask"]
            if isinstance(loss_mask, torch.Tensor):
                valid_tokens = int(loss_mask.sum().item())
            else:
                valid_tokens = sum(
                    int(token)
                    for row in loss_mask
                    for token in (row if isinstance(row, list) else [row])
                )
            return valid_tokens >= minimum_valid_tokens

        dataset = dataset.filter(
            has_minimum_valid_tokens,
            num_proc=num_proc,
            desc=f"Filtering samples with >= {minimum_valid_tokens} trainable tokens",
        )
        print(
            f"Filtered dataset by trainable tokens: {before_filter} -> {len(dataset)}"
        )

    dataset.set_format(type="torch")
    return dataset


# ==============================
# Offline Eagle3 Dataset
# ==============================
# modified from https://github.com/NickL77/BaldEagle/blob/master/train/modules/data/data.py
def list_local_files(path, suffixes=None):
    if suffixes is None:
        suffixes = [".ckpt", ".ckpt.gz"]
    datapaths = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapaths.append(file_path)
    if suffixes:
        datapaths = [
            f_name
            for f_name in datapaths
            if any(f_name.endswith(suffix) for suffix in suffixes)
        ]
    datapaths.sort()  # Sort to ensure deterministic order across ranks
    return datapaths


class OfflineEagle3Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datapath,
        transform=None,
        max_len=2048,
    ):
        """
        Args:
            datapath: List of file paths.
            transform: Optional transform to apply.
            max_len: Maximum sequence length to load.
        """
        self.datapaths = datapath
        self.transform = transform
        self._epoch = 0
        self.max_len = max_len

    @staticmethod
    def process_data(data, max_len, transform=None):
        new_data = {}
        # Squeeze due to our data generation script adding a batch dimension
        hidden_state = data["aux_hidden_state"].squeeze(0)[:max_len][None, :]
        target = data["hidden_state"].squeeze(0)[:max_len][None, :]

        input_ids = data["input_ids"][:max_len][None, :]
        loss_mask = data["loss_mask"][:max_len][None, :]
        loss_mask[0, -1] = 0

        new_data["attention_mask"] = torch.ones_like(loss_mask, dtype=torch.long)
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state"] = hidden_state
        new_data["input_ids"] = input_ids
        if transform:
            new_data = transform(new_data)
        return new_data

    def __len__(self):
        return len(self.datapaths)

    def _open_file(self, index):
        """
        Opens the file with memory mapping.
        This operation is virtually instant and consumes negligible RAM
        because no data is actually read from disk yet.
        """
        data_path = self.datapaths[index]
        if data_path.endswith(".gz"):
            with gzip.open(data_path, "rb") as f:
                return torch.load(io.BytesIO(f.read()), weights_only=False)
        return torch.load(data_path, weights_only=False, mmap=True)

    def __getitem__(self, index):
        try:
            data = self._open_file(index)
        except Exception as e:
            print(f"ERROR Failed to load {self.datapaths[index]} with error {e}")
            data = self._open_file(0)

        return self.process_data(
            data,
            self.max_len,
            self.transform,
        )

    def set_epoch(self, epoch):
        self._epoch = epoch


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
    print(f"Added missing tokens to reach draft vocab size: {draft_vocab_size}")
    print(f"Total tokens after addition: {len(token_dict)}")
    total_frequency = sum(token_dict.values())
    top_N = token_dict.most_common(draft_vocab_size)
    top_N_frequency_sum = sum(freq for key, freq in top_N)

    if total_frequency == 0:
        print(
            "Warning: Total token frequency is zero. All tokens will have zero ratio."
        )
        top_N_ratio = 0.0
    else:
        top_N_ratio = top_N_frequency_sum / total_frequency

    print(f"top {draft_vocab_size} token frequency ratio: {top_N_ratio:.2%}")
    used_tokens = [key for key, freq in top_N]
    used_tokens.sort()

    d2t = [used_tokens[i] - i for i in range(len(used_tokens))]
    t2d = [i in used_tokens for i in range(target_vocab_size)]
    d2t = torch.tensor(d2t)
    t2d = torch.tensor(t2d)

    return d2t, t2d
