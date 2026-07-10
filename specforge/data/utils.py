# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in HuggingFace Transformers.
# Portions of this code are adapted from:
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

import re
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from datasets import Dataset
from specforge.distributed import get_draft_sp_group, get_sp_ulysses_group


class DataCollatorWithPadding:
    """
    Datacollator that will dynamically pad the inputs for batching.
    """

    def __init__(self):
        self.sp_degree = torch.distributed.get_world_size(get_draft_sp_group())
        self.ulysses_degree = torch.distributed.get_world_size(get_sp_ulysses_group())

    def paddingtensor(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        """
        Pad to the longest sequence in the batch.

        Args:
            intensors: (B, n, S)
            N: the length to pad to, N >= n

        Returns:
            outtensors: (B, N, S)
        """
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(
            B, N - n, S, dtype=intensors.dtype, device=intensors.device
        )
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        """
        Pad 2D tensor to the longest sequence in the batch.

        Args:
            intensors: (B, n)
            N: the length to pad to, N >= n

        Returns:
            outtensors: (B, N)
        """
        B, n = intensors.shape
        padding_tensor = torch.zeros(
            B, N - n, dtype=intensors.dtype, device=intensors.device
        )
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of features.

        Args:
            features: A list of features, where each feature is a dictionary containing:
                - input_ids: torch.Tensor of shape (n,)
                - attention_mask: torch.Tensor of shape (n,)
                - loss_mask: torch.Tensor of shape (n,)

        Returns:
            A dictionary containing:
                - input_ids: torch.Tensor of shape (B, N)
                - attention_mask: torch.Tensor of shape (B, N)
                - loss_mask: torch.Tensor of shape (B, N)
        """
        max_length = max(item["input_ids"].shape[1] for item in features)

        # pad for sequence parrel
        max_length = (
            (max_length + self.sp_degree - 1) // self.sp_degree
        ) * self.sp_degree
        # position max len, ulysses do not need chuck position ids
        position_max_len = max_length * self.ulysses_degree

        batch_input_ids = torch.cat(
            [self.paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [
                self.paddingtensor2D(item["attention_mask"], max_length)
                for item in features
            ]
        )
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item["loss_mask"], max_length) for item in features]
        )
        if "position_ids" in features[0]:
            batch_position_ids = torch.cat(
                [
                    self.paddingtensor2D(item["position_ids"], position_max_len)
                    for item in features
                ]
            )
        else:
            batch_position_ids = None
        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "hidden_state": None,
            "target": None,
        }
        if batch_position_ids is not None:
            batch["position_ids"] = batch_position_ids
        if all("hidden_state" in item for item in features):
            assert all(
                "target" in item for item in features
            ), "target is required when hidden_state is provided"
            if self.sp_degree > 1:  # USP mode
                batch["hidden_state"] = torch.cat(
                    [item["hidden_state"] for item in features]
                )
            else:
                batch["hidden_state"] = torch.cat(
                    [
                        self.paddingtensor(item["hidden_state"], max_length)
                        for item in features
                    ]
                )
            batch["target"] = torch.cat(
                [self.paddingtensor(item["target"], max_length) for item in features]
            )
        return batch


class VlmDataCollatorWithPadding:
    """
    Datacollator that will dynamically pad the inputs for batching.
    """

    def paddingtensor(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        """
        Pad to the longest sequence in the batch.

        Args:
            intensors: (B, n, S)
            N: the length to pad to, N >= n

        Returns:
            outtensors: (B, N, S)
        """
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        """
        Pad 2D tensor to the longest sequence in the batch.

        Args:
            intensors: (B, n)
            N: the length to pad to, N >= n

        Returns:
            outtensors: (B, N)
        """
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of features.

        Args:
            features: A list of features, where each feature is a dictionary containing:
                - input_ids: torch.Tensor of shape (n,)
                - attention_mask: torch.Tensor of shape (n,)
                - loss_mask: torch.Tensor of shape (n,)
                - pixel_values: torch.Tensor of shape (grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size)
                - image_grid_thw: torch.Tensor of shape (3,)

        Returns:
            A dictionary containing:
                - input_ids: torch.Tensor of shape (B, N)
                - attention_mask: torch.Tensor of shape (B, N)
                - loss_mask: torch.Tensor of shape (B, N)
        """
        assert len(features) == 1, (
            f"VlmDataCollatorWithPadding requires batch_size=1, got {len(features)}. "
            "Set per_device_train_batch_size=1 in your training config."
        )
        max_length = max(item["input_ids"].shape[1] for item in features)
        batch_input_ids = torch.cat(
            [self.paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [
                self.paddingtensor2D(item["attention_mask"], max_length)
                for item in features
            ]
        )
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item["loss_mask"], max_length) for item in features]
        )
        # Collect pixel_values and image_grid_thw per sample.
        # Image samples have non-empty pixel_values; text-only samples have empty tensors.
        all_pixel_values = []
        all_image_grid_thw = []
        for item in features:
            pv = item.get("pixel_values")
            thw = item.get("image_grid_thw")
            if pv is not None and isinstance(pv, torch.Tensor) and pv.numel() > 0:
                all_pixel_values.append(pv)
                all_image_grid_thw.append(thw)
            else:
                all_image_grid_thw.append(None)

        if all_pixel_values:
            batch_pixel_values = torch.cat(all_pixel_values, dim=0)
        else:
            batch_pixel_values = None

        # If all samples are text-only, set image_grid_thw to None
        if all(thw is None for thw in all_image_grid_thw):
            batch_image_grid_thw = None
        else:
            batch_image_grid_thw = all_image_grid_thw

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "pixel_values": batch_pixel_values,
            "image_grid_thw": batch_image_grid_thw,
            "hidden_state": None,
            "target": None,
        }
        if all("hidden_state" in item for item in features):
            assert all(
                "target" in item for item in features
            ), "target is required when hidden_state is provided"
            batch["hidden_state"] = torch.cat(
                [
                    self.paddingtensor(item["hidden_state"], max_length)
                    for item in features
                ]
            )
            batch["target"] = torch.cat(
                [self.paddingtensor(item["target"], max_length) for item in features]
            )
        return batch


def prepare_dp_dataloaders(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    process_group: Optional[dist.ProcessGroup] = None,
    pin_memory: Optional[bool] = False,
    shuffle: Optional[bool] = False,
    is_vlm: Optional[bool] = False,
    prefetch_factor: Optional[int] = 2,
    **dataloader_kwargs,
) -> DataLoader:
    """
    Prepare dataloader for distributed data parallel training.

    Args:
        dataset: The dataset to load data from.
        batch_size: The batch size for each GPU.
        num_workers: The number of workers for data loading.
        process_group: The process group for distributed training.
        pin_memory: Whether to pin memory for data loading.
        shuffle: Whether to shuffle the dataset.
        is_vlm: Whether the dataset is a vision-language model dataset.
        **dataloader_kwargs: Additional keyword arguments for the DataLoader.

    Returns:
        A DataLoader for the dataset.
    """
    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
    )
    if is_vlm:
        datacollator_cls = VlmDataCollatorWithPadding
    else:
        datacollator_cls = DataCollatorWithPadding

    if num_workers == 0:
        prefetch_factor = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        collate_fn=datacollator_cls(),
        drop_last=True,
        **dataloader_kwargs,
    )
    return dataloader


def parse_harmony_message_content(content):
    pattern = r"<\|channel\|>(.*?)<\|message\|>(.*?)<\|end|>"
    matches = re.findall(pattern, content, re.DOTALL)

    if not matches:
        return [{"channel": "text", "content": content}]

    results = []
    for channel, msg_body in matches:
        results.append({"channel": channel.strip(), "content": msg_body.strip()})
    return results


def process_harmony_conversations(conversation):
    new_conversation = []
    for msg in conversation:
        role = msg.get("role")
        original_content = msg.get("content", "")

        segments = parse_harmony_message_content(original_content)

        for seg in segments:
            new_msg = {
                "role": role,
                "channel": seg["channel"],
                "content": seg["content"],
            }
            new_conversation.append(new_msg)

    return new_conversation
