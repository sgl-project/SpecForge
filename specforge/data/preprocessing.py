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

import logging
import os
import re
import warnings
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm
from transformers import ImageProcessingMixin, PreTrainedTokenizer

from datasets import Dataset as HFDataset

try:
    from qwen_vl_utils import process_vision_info

    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False
    process_vision_info = None

from specforge.utils import padding

from .parse import GeneralParser, HarmonyParser, ThinkingParser
from .template import TEMPLATE_REGISTRY, ChatTemplate

# define a type called conversation
Conversation = List[Dict[str, str]]

# Module-level processor cache for multiprocessing (MiniCPM-V)
_MINICPM_PROCESSOR_CACHE = {}


def _load_minicpm_processor(processor_path: str):
    """Load MiniCPM-V processor in worker process to avoid pickle issues."""
    global _MINICPM_PROCESSOR_CACHE
    if processor_path not in _MINICPM_PROCESSOR_CACHE:
        from transformers import AutoProcessor
        _MINICPM_PROCESSOR_CACHE[processor_path] = AutoProcessor.from_pretrained(
            processor_path, trust_remote_code=True
        )
    return _MINICPM_PROCESSOR_CACHE[processor_path]


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


def preprocess_minicpm_vlm_conversations(
    processor: Optional[ImageProcessingMixin],
    examples: List[Conversation],
    chat_template: ChatTemplate,
    max_length: int = 2048,
    processor_path: Optional[str] = None,
) -> Dict[str, List[torch.Tensor]]:
    """
    Preprocess a batch of ShareGPT style conversations for MiniCPM-V-4.0.

    This function handles MiniCPM-V's unique image processing:
    - Uses slice-based image encoding instead of grid
    - Outputs pixel_values, tgt_sizes, image_bound instead of image_grid_thw
    - Supports single image or multiple images per example
    - Supports standard <image> placeholder (auto-converted to MiniCPM-V format)
    - Supports lazy loading of processor via processor_path for multiprocessing

    Args:
        processor: The MiniCPM-V processor for processing images.
        examples: A list of examples, where each example is a dictionary containing:
            - image: Single image path/PIL.Image or list of image paths/PIL.Images
            - conversations: A list of conversations. For multi-image, use
              <image> placeholder to specify image positions in content.
        chat_template: The chat template to use for formatting.
        max_length: The maximum length of the tokenized input.

    Returns:
        A dictionary containing:
            - input_ids: List of tokenized input IDs.
            - loss_mask: List of loss masks.
            - attention_mask: List of attention masks.
            - pixel_values: List of pixel values (list of slice tensors per image).
            - tgt_sizes: List of target sizes for each slice.
            - image_bound: List of image token boundaries in the sequence.
            - position_ids: List of position IDs.

    Example data formats:
        Single image (auto-add placeholder):
        {
            "image": "path/to/image.jpg",
            "conversations": [
                {"role": "user", "content": "描述这张图片"},
                {"role": "assistant", "content": "这是..."}
            ]
        }

        Multiple images (use <image> to specify positions):
        {
            "image": ["path/to/img1.jpg", "path/to/img2.jpg"],
            "conversations": [
                {"role": "user", "content": "<image>第一张图\n<image>第二张图\n请比较这两张图片"},
                {"role": "assistant", "content": "第一张图片...第二张图片..."}
            ]
        }

    Note:
        The standard <image> placeholder will be automatically converted to
        MiniCPM-V's internal format (<image>./</image>).
    """
    from PIL import Image

    # Lazy load processor if not provided but path is given (for multiprocessing)
    if processor is None and processor_path is not None:
        processor = _load_minicpm_processor(processor_path)
    elif processor is None:
        raise ValueError("Either processor or processor_path must be provided")

    system_prompt = chat_template.system_prompt
    STANDARD_PLACEHOLDER = "<image>"            # 行业标准格式
    MINICPM_PLACEHOLDER = "(<image>./</image>)"  # MiniCPM-V 内部格式

    # prepare result
    results = {
        "input_ids": [],
        "loss_mask": [],
        "attention_mask": [],
        "pixel_values": [],
        "tgt_sizes": [],
        "image_bound": [],
        "position_ids": [],
    }

    for i, image_input in enumerate(examples["image"]):
        source = examples["conversations"][i]

        if not source:
            continue

        if source[0]["role"] != "user":
            source = source[1:]

        # Normalize image input to list
        if image_input is None:
            image_list = []
        elif isinstance(image_input, list):
            image_list = image_input
        else:
            image_list = [image_input]

        # Load all images
        images = []
        for img in image_list:
            if isinstance(img, str):
                loaded_img = Image.open(img).convert("RGB")
            elif isinstance(img, Image.Image):
                loaded_img = img.convert("RGB")
            else:
                loaded_img = img
            images.append(loaded_img)

        # Build messages in MiniCPM-V format
        messages = []
        
        # Check if content already has image placeholders (standard or MiniCPM format)
        first_user_content = source[0]["content"] if source else ""
        has_standard = STANDARD_PLACEHOLDER in first_user_content
        has_minicpm = MINICPM_PLACEHOLDER in first_user_content
        has_placeholder = has_standard or has_minicpm

        for j, sentence in enumerate(source):
            role = sentence["role"]
            content = sentence["content"]

            # Convert standard placeholder to MiniCPM-V format
            if STANDARD_PLACEHOLDER in content:
                content = content.replace(STANDARD_PLACEHOLDER, MINICPM_PLACEHOLDER)

            if role == "user" and j == 0 and images and not has_placeholder:
                # First user message: add image placeholders if not already present
                # Add one placeholder per image
                placeholders = "\n".join([MINICPM_PLACEHOLDER] * len(images))
                content = f"{placeholders}\n{content}"

            messages.append({"role": role, "content": content})

        # Apply chat template using processor's tokenizer
        if system_prompt:
            sys_msg = {"role": "system", "content": system_prompt}
            messages = [sys_msg] + messages

        # Format conversation text
        conversation = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Process with MiniCPM-V processor
        # Note: MiniCPM-V processor has a bug where images=None causes UnboundLocalError
        # Pass [[]] (empty list) instead of None when no images
        try:
            encoding = processor(
                text=[conversation],
                images=[images] if images else [[]],
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
        except RuntimeError as e:
            # Skip samples that cause errors (e.g., truncation causing mismatched image bounds)
            logging.warning(f"Skipping sample {i} due to processor error: {e}")
            continue

        input_ids = encoding.input_ids[0]
        offsets = encoding.offset_mapping[0] if hasattr(encoding, "offset_mapping") else None

        # Extract MiniCPM-V specific fields
        pixel_values = encoding.pixel_values if hasattr(encoding, "pixel_values") else None
        tgt_sizes = encoding.tgt_sizes if hasattr(encoding, "tgt_sizes") else None
        image_bound = encoding.image_bound if hasattr(encoding, "image_bound") else None
        position_ids = encoding.position_ids if hasattr(encoding, "position_ids") else None

        # Generate loss mask based on token IDs (MiniCPM-V doesn't support offset_mapping)
        # Pattern: <|im_start|> assistant \n ... <|im_end|>
        IM_START = 73441  # <|im_start|>
        IM_END = 73440    # <|im_end|>
        ASSISTANT = 16434  # assistant
        NEWLINE = 5        # \n
        
        loss_mask = torch.zeros(len(input_ids), dtype=torch.long)
        is_truncated = False
        idx = 0
        while idx < len(input_ids) - 2:
            # Check for <|im_start|> assistant \n sequence
            if (input_ids[idx] == IM_START and 
                input_ids[idx + 1] == ASSISTANT and 
                input_ids[idx + 2] == NEWLINE):
                # Found assistant header, content starts at idx+3
                start = idx + 3
                # Find ending <|im_end|>
                end = start
                while end < len(input_ids) and input_ids[end] != IM_END:
                    end += 1
                # Check if truncated (no <|im_end|> found)
                if end >= len(input_ids):
                    is_truncated = True
                    break
                # Mark assistant response tokens (including <|im_end|> for EOS learning)
                loss_mask[start:end+1] = 1
                idx = end + 1
            else:
                idx += 1

        # Skip truncated samples (assistant response cut off)
        if is_truncated:
            logging.warning(f"Skipping sample {i}: assistant response truncated")
            continue

        # Skip samples with no assistant response
        if loss_mask.sum() == 0:
            logging.warning(f"Skipping sample {i}: no assistant response found")
            continue

        results["input_ids"].append(input_ids[None, :])
        results["loss_mask"].append(loss_mask[None, :])
        results["attention_mask"].append(torch.ones_like(loss_mask)[None, :])
        # Remove batch dimension from processor outputs (take first element)
        # pixel_values: [[[tensor], ...]] -> [[tensor], ...] -> list of tensors per image
        if pixel_values is not None and len(pixel_values) > 0:
            # pixel_values[0] is the first (and only) batch item
            results["pixel_values"].append(pixel_values[0])
        else:
            results["pixel_values"].append([])
        if tgt_sizes is not None and len(tgt_sizes) > 0:
            results["tgt_sizes"].append(tgt_sizes[0])
        else:
            results["tgt_sizes"].append(None)
        if image_bound is not None and len(image_bound) > 0:
            results["image_bound"].append(image_bound[0])
        else:
            results["image_bound"].append(None)
        if position_ids is not None:
            results["position_ids"].append(position_ids)
        else:
            # Generate position_ids if not provided
            results["position_ids"].append(
                torch.arange(len(input_ids), dtype=torch.long)[None, :]
            )

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
) -> HFDataset:
    """
    Build eagle3 dataset for LLM models.
    
    Note: For VLM models, use build_vlm_dataset_with_template() instead.

    Args:
        dataset: HF dataset to process.
        tokenizer: The tokenizer to use for tokenization.
        chat_template: The chat template to use for formatting conversations.
        max_length: The maximum length of the tokenized input.
        shuffle_seed: The seed for shuffling the dataset.
        num_proc: The number of processes to use for multiprocessing.
        cache_dir: The directory to use for caching the processed dataset.
        cache_key: The key to use for caching the processed dataset.
        is_preformatted: Whether the dataset contains preformatted text.
        train_only_last_turn: If True, only the last assistant turn contributes to the loss.

    Returns:
        The processed HF dataset.
    """

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

    batch_size = 1000  # default for conversations
    
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        batch_size=batch_size,
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
        new_data["target"] = padding(target, left=False)
        new_data["hidden_state"] = hidden_state
        new_data["input_ids"] = padding(input_ids, left=False)
        if transform:
            new_data = transform(new_data)
        return new_data

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
        return self.process_data(data, self.max_len, self.transform)

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
def generate_vocab_mapping_for_vlm(
    hf_dataset: HFDataset,
    tokenizer_path: str,
    target_vocab_size: int,
    draft_vocab_size: int,
    cache_dir: str = "./cache/vocab_mapping",
    cache_key: str = "vocab_mapping",
    num_proc: int = 32,
) -> str:
    """
    Generate vocab mapping for VLM dataset using TEXT-ONLY processing.
    
    Simply extracts assistant response text, tokenizes it, and counts token frequencies.
    Much faster than processing images.
    
    Args:
        hf_dataset: Raw HuggingFace dataset with 'conversations' column.
        tokenizer_path: Path to load tokenizer from.
        target_vocab_size: The target model vocabulary size.
        draft_vocab_size: The draft model vocabulary size.
        cache_dir: Directory for caching.
        cache_key: Cache key for the vocab mapping file.
        num_proc: Number of processes for tokenization.
    
    Returns:
        Path to the vocab mapping file.
    """
    from transformers import AutoTokenizer
    
    os.makedirs(cache_dir, exist_ok=True)
    vocab_mapping_path = os.path.join(cache_dir, f"{cache_key}.pt")
    
    if os.path.exists(vocab_mapping_path):
        print(f"Loading vocab mapping from cached file: {vocab_mapping_path}")
        return vocab_mapping_path
    
    print("Generating vocab mapping (text-only, fast mode)...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    def tokenize_assistant_responses(examples):
        """Extract and tokenize assistant responses only."""
        all_token_ids = []
        
        for conversations in examples["conversations"]:
            if not conversations:
                continue
            
            # Simply collect all assistant response text
            for conv in conversations:
                if conv["role"] == "assistant":
                    try:
                        # Tokenize assistant response directly
                        tokens = tokenizer.encode(conv["content"], add_special_tokens=False)
                        all_token_ids.extend(tokens)
                    except Exception:
                        continue
        
        return {"assistant_tokens": [all_token_ids]}
    
    # Process dataset with multiprocessing
    print(f"Tokenizing assistant responses with {num_proc} processes...")
    processed = hf_dataset.map(
        tokenize_assistant_responses,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=hf_dataset.column_names,
    )
    
    # Count token frequencies
    token_dict = Counter()
    for item in tqdm(processed, desc="Counting token frequencies"):
        token_dict.update(item["assistant_tokens"])
    
    # Generate mappings
    d2t, t2d = process_token_dict_to_mappings(
        token_dict, draft_vocab_size, target_vocab_size
    )
    
    vocab_mapping = {"d2t": d2t, "t2d": t2d}
    torch.save(vocab_mapping, vocab_mapping_path)
    print(f"Saved vocab mapping to: {vocab_mapping_path}")
    
    return vocab_mapping_path


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

    # we first count the frequency of effective tokens in the dataset
    token_dict = Counter()
    for item in tqdm(dataset, desc="Counting tokens for vocab mapping"):
        # Skip None items (VLMDataset may return None for invalid samples)
        if item is None:
            continue
        
        input_ids = item["input_ids"]
        loss_mask = item["loss_mask"]
        
        # Handle different tensor shapes:
        # - HF dataset: (seq_len,) or (1, seq_len)
        # - VLMDataset: (1, seq_len)
        if input_ids.dim() == 2:
            input_ids = input_ids.squeeze(0)
        if loss_mask.dim() == 2:
            loss_mask = loss_mask.squeeze(0)
        
        masked_ids = input_ids[loss_mask == 1]
        if len(masked_ids) == 0:
            continue
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


# ==============================
# VLM Template-based Dataset Building
# ==============================
def build_vlm_dataset_with_template(
    dataset: HFDataset,
    vlm_template,
    shuffle_seed: Optional[int] = 42,
    num_proc: Optional[int] = 32,
):
    """
    Build VLM dataset using the new Template architecture.
    
    This function uses VLMDataset wrapper which applies template encoding
    on-the-fly during DataLoader iteration. The dataset.map() phase only
    does basic formatting (multiprocessing-safe).
    
    Args:
        dataset: HuggingFace dataset with 'image' and 'conversations' columns
        vlm_template: VLMTemplate instance (MiniCPMVTemplate, QwenVLTemplate, etc.)
        shuffle_seed: Seed for shuffling
        num_proc: Number of processes for formatting (not used for encoding)
    
    Returns:
        VLMDataset wrapper ready for DataLoader
    """
    from .dataset import VLMDataset
    
    # Shuffle dataset
    dataset = dataset.shuffle(seed=shuffle_seed)
    
    # Basic format validation (can be done with multiprocessing)
    def validate_format(example):
        """Validate that example has required fields."""
        has_image = "image" in example and example["image"] is not None
        has_conversations = "conversations" in example and example["conversations"]
        return {"valid": has_image and has_conversations}
    
    # Optional: filter invalid samples (this can use multiprocessing)
    # dataset = dataset.map(validate_format, num_proc=num_proc)
    # dataset = dataset.filter(lambda x: x["valid"])
    
    # Wrap with VLMDataset - encoding happens in __getitem__
    return VLMDataset(dataset, vlm_template)


def get_vlm_template(
    vlm_type: str,
    processor_path: str,
    max_length: int = 2048,
    system_prompt: Optional[str] = None,
):
    """
    Get VLM template by type.
    
    Args:
        vlm_type: Type of VLM ("minicpm_v_4", "qwen2_5_vl")
        processor_path: Path to processor/model
        max_length: Maximum sequence length
        system_prompt: Optional system prompt
    
    Returns:
        VLMTemplate instance
    """
    from .vlm_template import MiniCPMVTemplate, QwenVLTemplate
    
    if vlm_type == "minicpm_v_4":
        return MiniCPMVTemplate(
            processor_path=processor_path,
            max_length=max_length,
            system_prompt=system_prompt or "You are a helpful assistant.",
        )
    elif vlm_type in ("qwen2_5_vl", "qwen2_vl"):
        return QwenVLTemplate(
            processor_path=processor_path,
            max_length=max_length,
            system_prompt=system_prompt or "You are a helpful assistant.",
        )
    else:
        raise ValueError(f"Unknown VLM type: {vlm_type}. Supported: minicpm_v_4, qwen2_5_vl")
