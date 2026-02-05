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
import os
import re
import warnings
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import ImageProcessingMixin, PreTrainedTokenizer

from datasets import Dataset as HFDataset

from ..distributed import get_draft_sp_group, get_sp_ring_group

try:
    from qwen_vl_utils import process_vision_info

    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False
    process_vision_info = None


from .parse import GeneralParser, HarmonyParser, ThinkingParser
from .template import TEMPLATE_REGISTRY, ChatTemplate

# define a type called conversation
Conversation = List[Dict[str, str]]


# ==============================
# This file is for preprocessing the data
# ==============================


def _apply_loss_mask_from_chat_template(
    text: str,
    offsets: torch.Tensor,
    chat_template: ChatTemplate,
) -> torch.Tensor:
    """
    Apply loss mask to identify assistant response spans using chat template.

    Args:
        text: The formatted conversation text.
        offsets: Token offset mapping from tokenizer.
        chat_template: The chat template to use for identifying assistant spans.

    Returns:
        A tensor indicating which tokens should contribute to the loss (1) or not (0).
    """
    loss_mask = torch.zeros(len(offsets), dtype=torch.long)

    user_message_separator = (
        f"{chat_template.end_of_turn_token}{chat_template.user_header}"
    )
    assistant_message_separator = (
        f"{chat_template.end_of_turn_token}{chat_template.assistant_header}"
    )

    # Find spans of assistant responses using regex
    assistant_pattern = (
        re.escape(assistant_message_separator)
        + r"(.*?)(?="
        + re.escape(user_message_separator)
        + "|$)"
    )

    matches_found = 0

    for match in re.finditer(assistant_pattern, text, re.DOTALL):
        matches_found += 1
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

    if matches_found == 0:
        print("WARNING: No assistant response spans found in the conversation text.")

    return loss_mask


# Copied from https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/cnets.py
def preprocess_conversations(
    tokenizer: PreTrainedTokenizer,
    conversations: Union[List[Conversation], List[str]],
    chat_template: ChatTemplate,
    max_length: int = 2048,
    is_preformatted: bool = False,
    train_only_last_turn: bool = False,
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
    for source, kwargs_item in zip(conversations, kwargs_list):
        if not source:
            # if the source is None, skip it
            continue
        input_ids, loss_mask = parser.parse(
            source,
            max_length,
            preformatted=is_preformatted,
            train_only_last_turn=train_only_last_turn,
            **kwargs_item,
        )
        results["input_ids"].append(input_ids[None, :])
        results["loss_mask"].append(loss_mask[None, :])
        results["attention_mask"].append(torch.ones_like(loss_mask)[None, :])
    return results


def preprocess_vlm_conversations(
    processor: ImageProcessingMixin,
    examples: List[Conversation],
    chat_template: ChatTemplate,
    max_length: int = 2048,
) -> Dict[str, List[torch.Tensor]]:
    """
    Preprocess a batch of ShareGPT style conversations.

    Args:
        processor: The image processor to use for processing images.
        examples: A list of examples, where each example is a dictionary containing:
            - image: The image in the conversation.
            - conversations: A list of conversations, where each conversation is a list of messages.
        chat_template: The chat template to use for formatting the conversations.
        max_length: The maximum length of the tokenized input.

    Returns:
        A dictionary containing:
            - input_ids: List of tokenized input IDs.
            - loss_mask: List of loss masks indicating which tokens should contribute to the loss.
            - attention_mask: List of attention masks.
            - pixel_values: List of pixel values for images in the examples.
            - image_grid_thw: List of image grid tensors.
    """
    system_prompt = chat_template.system_prompt

    # prepare result
    results = {
        "input_ids": [],
        "loss_mask": [],
        "attention_mask": [],
        "pixel_values": [],
        "image_grid_thw": [],
    }

    # Note: currently, we assume that each example has only one image
    for i, image in enumerate(examples["image"]):
        source = examples["conversations"][i]
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
            if role == "user":
                # if the message is from user and has image, process the image
                messages.append(
                    {
                        "role": role,
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                            },
                            {"type": "text", "text": sentence["content"]},
                        ],
                    }
                )
            else:
                messages.append({"role": role, "content": sentence["content"]})

        conversation = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        # get vision infor use qwen_vl_utils
        if not HAS_QWEN_VL_UTILS:
            raise ImportError(
                "qwen_vl_utils is required for VLM preprocessing but is not installed. "
                "Please install it to use VLM features."
            )
        image_inputs, video_inputs = process_vision_info(messages)
        assert image_inputs is not None, "image_inputs must not be None"

        encoding = processor(
            text=[conversation],
            images=image_inputs,
            videos=video_inputs,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        input_ids = encoding.input_ids[0]
        offsets = encoding.offset_mapping[0]
        pixel_values = encoding.pixel_values
        image_grid_thw = encoding.image_grid_thw[0]

        # get conversation with image info for loss mask generation
        decoded_conversation = processor.tokenizer.decode(
            encoding.input_ids[0], skip_special_tokens=False
        )

        # Apply loss mask
        loss_mask = _apply_loss_mask_from_chat_template(
            decoded_conversation, offsets, chat_template
        )

        results["input_ids"].append(input_ids[None, :])
        results["loss_mask"].append(loss_mask[None, :])
        results["attention_mask"].append(torch.ones_like(loss_mask)[None, :])
        results["pixel_values"].append(pixel_values)
        results["image_grid_thw"].append(image_grid_thw[None, :])
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
    is_vlm: Optional[bool] = False,
    processor: Optional[ImageProcessingMixin] = None,
    is_preformatted: Optional[bool] = False,
    train_only_last_turn: Optional[bool] = False,
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
        is_vlm: Whether the dataset is for VLM models.
        processor: The image processor to use for processing images.
        is_preformatted: Whether the dataset contains preformatted text of the conversation
                        (e.g. includes system prompt, user and assistant start and end tokens)
                        and doesn't need to have the chat template applied.
                        Note that the chat_template still needs to be specified to determine
                        the assistant spans for loss mask generation.
                        If True, expects "text" column with ready-to-train text.
                        If False, expects "conversations" column with ShareGPT format.
        train_only_last_turn: If True, only the last assistant turn contributes to the loss.
                             Useful for thinking models where history may not contain thoughts.

    Returns:
        The processed HF dataset.
    """
    if is_vlm:
        assert processor is not None, "processor must be provided when is_vlm is True"

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
        if is_vlm:
            processed = preprocess_vlm_conversations(
                processor,
                examples,
                template,
                max_length,
            )
        elif is_preformatted:
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
            processed = preprocess_conversations(
                tokenizer,
                conversations,
                template,
                max_length,
                is_preformatted=False,
                train_only_last_turn=train_only_last_turn,
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

    # adjust batch size based on dataset type
    if is_vlm:
        batch_size = (
            200  # reduce batch size for VLM datasets to avoid PyArrow offset overflow
        )
    else:
        batch_size = 1000  # default for conversations
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        batch_size=batch_size,
        remove_columns=original_cols,
        # keep_in_memory=True,
        load_from_cache_file=load_from_cache_file,
        cache_file_name=cache_file_name,
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
    return datapaths


class OfflineEagle3Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datapath,
        transform=None,
        max_len=2048,
        ttt_length=1,
        use_usp_preprocess=False,
    ):
        """
        Args:
            datapath: List of file paths.
            transform: Optional transform to apply.
            max_len: Maximum sequence length to load.
            ttt_length: TTT overlap length used in USP preprocessing.
            use_usp_preprocess: Whether to shard all sequences with USP overlap in preprocessing.
        """
        self.datapaths = datapath
        self.transform = transform
        self._epoch = 0
        self.max_len = max_len
        self.ttt_length = ttt_length
        self.use_usp_preprocess = use_usp_preprocess
        if use_usp_preprocess:
            sp_group = get_draft_sp_group()
            self.sp_rank = torch.distributed.get_rank(sp_group)
            self.sp_size = torch.distributed.get_world_size(sp_group)
            ring_group = get_sp_ring_group()
            self.ring_rank = torch.distributed.get_rank(ring_group)
            self.sp_ring_size = torch.distributed.get_world_size(ring_group)

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

    @staticmethod
    def process_data_usp(
        data,
        max_len,
        ttt_length=1,
        transform=None,
        sp_rank=0,
        sp_size=1,
        ring_rank=0,
        sp_ring_size=1,
    ):
        """
        USP preprocess: shard all sequences by sp_rank and add TTT overlap.
        Each local sequence length = ceil(max_len / sp_size) + ttt_length.
        """
        new_data = {}

        input_ids = data["input_ids"]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        global_len = min(max_len, input_ids.shape[1])
        chunk_size = (global_len + sp_size - 1) // sp_size
        start = sp_rank * chunk_size
        local_len = chunk_size + ttt_length

        end = min(start + local_len, global_len)

        def _slice_and_pad(tensor):
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            tensor = tensor[:, :global_len]
            sliced = tensor[:, start : min(end, tensor.shape[1])]
            valid_len = sliced.shape[1]
            if valid_len < local_len:
                pad_len = local_len - valid_len
                if tensor.ndim == 2:
                    sliced = F.pad(sliced, (0, pad_len))
                else:
                    sliced = F.pad(sliced, (0, 0, 0, pad_len))
            return sliced.contiguous(), valid_len

        if "aux_hidden_state" not in data or data["aux_hidden_state"] is None:
            raise KeyError("aux_hidden_state is required for OfflineEagle3Dataset")
        new_data["hidden_state"], _ = _slice_and_pad(data["aux_hidden_state"])
        new_data["target"], _ = _slice_and_pad(data["hidden_state"])

        new_data["input_ids"], valid_len = _slice_and_pad(input_ids)

        full_loss_mask = data["loss_mask"]
        if full_loss_mask.ndim == 1:
            full_loss_mask = full_loss_mask.unsqueeze(0)

        full_loss_mask = full_loss_mask[:, :global_len].clone()
        if full_loss_mask.numel() > 0:
            full_loss_mask[0, -1] = 0
        new_data["loss_mask"], _ = _slice_and_pad(full_loss_mask)

        local_len = new_data["input_ids"].shape[1]
        attention_mask = torch.zeros((1, local_len), dtype=torch.long)
        attention_mask[:, :valid_len] = 1
        new_data["attention_mask"] = attention_mask

        # Position ids should align with Ulysses all2all-expanded sequence length.
        # Local seq_len (per sp_rank) = local_len; attention uses (local_len - ttt_length).
        sp_ulysses_size = max(1, sp_size // sp_ring_size)
        usp_chunk_size = max(local_len - ttt_length, 0)
        ring_chunk = usp_chunk_size * sp_ulysses_size
        ring_start = ring_rank * ring_chunk
        new_data["position_ids"] = torch.arange(
            ring_start, ring_start + ring_chunk, dtype=torch.long
        ).unsqueeze(0)

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

        # 2. Read only specific bytes from disk
        if self.use_usp_preprocess:
            return self.process_data_usp(
                data,
                self.max_len,
                ttt_length=self.ttt_length,
                transform=self.transform,
                sp_rank=self.sp_rank,
                sp_size=self.sp_size,
                ring_rank=self.ring_rank,
                sp_ring_size=self.sp_ring_size,
            )
        return self.process_data(
            data,
            self.max_len,
            self.transform,
        )

    def set_epoch(self, epoch):
        self._epoch = epoch


def build_offline_eagle3_dataset(
    hidden_states_path: str,
    max_len: int = 2048,
    ttt_length: int = 1,
    use_usp_preprocess: bool = False,
) -> torch.utils.data.Dataset:

    return OfflineEagle3Dataset(
        list_local_files(hidden_states_path),
        max_len=max_len,
        ttt_length=ttt_length,
        use_usp_preprocess=use_usp_preprocess,
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
    # prepare cache directory
    os.makedirs(cache_dir, exist_ok=True)
    vocab_mapping_path = os.path.join(cache_dir, f"{cache_key}.pt")

    if os.path.exists(vocab_mapping_path):
        print(f"Loading vocab mapping from the cached file at: {vocab_mapping_path}")
        return vocab_mapping_path

    # we first count the frequency of effective tokens in the dataset
    token_dict = Counter()
    for input_ids, loss_mask in tqdm(
        zip(dataset["input_ids"], dataset["loss_mask"]),
        total=len(dataset),
        desc="Counting tokens for vocab mapping",
    ):
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
