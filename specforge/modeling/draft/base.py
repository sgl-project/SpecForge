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

import glob
import json
import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import PreTrainedModel
from transformers.cache_utils import Cache

from specforge.distributed import get_tp_group
from specforge.layers.linear import ColumnParallelLinear, RowParallelLinear
from specforge.modeling._mask_utils import _expand_mask, _make_causal_mask


class Eagle3DraftModel(PreTrainedModel, ABC):
    """
    This is the base class for the Eagle3 draft model implementation. The child class needs to implement
    the abstract methods to support training with TTT.
    """

    @abstractmethod
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed the input ids.
        """
        pass

    @abstractmethod
    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project the concatenated hidden states from the high, medium and low layers to the target hidden size.
        """
        pass

    @abstractmethod
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute the logits of the draft model.
        """
        pass

    def prepare_decoder_attention_mask(
        self,
        attention_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        batch_size: int,
        seq_length: int,
        past_key_values_length: int,
    ) -> torch.Tensor:
        """
        Prepare the attention mask of the draft model.
        """
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if seq_length > 1:
            combined_attention_mask = _make_causal_mask(
                (batch_size, seq_length),
                hidden_states.dtype,
                device=hidden_states.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, hidden_states.dtype, tgt_len=seq_length
            ).to(hidden_states.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask

    @abstractmethod
    def backbone(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        The backbone of the draft model.
        """
        pass

    def freeze_embedding(self) -> None:
        """
        Freeze the embeddings of the draft model so that they are not updated during training.
        """
        self.embed_tokens.weight.requires_grad = False

    @torch.no_grad()
    def load_embedding(
        self, model_path: str, embedding_key: str = "model.embed_tokens.weight"
    ) -> None:
        """
        Load the embedding of the draft model.

        Args:
            model_path (str): The path to the huggingface repository.
        """
        if os.path.exists(model_path):
            # model_path is a local directory
            # check if there is file ending with index.json
            glob_path = os.path.join(model_path, "*.index.json")
            index_json_path = glob.glob(glob_path)

            if len(index_json_path) == 0:
                raise FileNotFoundError(f"No index.json file found in {model_path}")
            if len(index_json_path) > 1:
                raise FileNotFoundError(
                    f"Multiple index.json files found in {model_path}"
                )
            index_json_path = index_json_path[0]

            with open(index_json_path, "r") as f:
                index_json = json.load(f)
            ckpt_file = index_json["weight_map"][embedding_key]

            if ckpt_file.endswith(".safetensors"):
                with safe_open(
                    os.path.join(model_path, ckpt_file), framework="pt"
                ) as f:
                    emb_tokens = f.get_tensor(embedding_key)
            else:
                state_dict = torch.load(os.path.join(model_path, ckpt_file))
                emb_tokens = state_dict[embedding_key]
            self.embed_tokens.weight.copy_(emb_tokens)
        else:
            # this is the case where model_path is a huggingface repository
            # we first need to locate its local cache
            local_cache_path = snapshot_download(repo_id=model_path)
            self.load_embedding(local_cache_path, embedding_key)

    def load_vocab_mapping(self, file_path: str) -> None:
        """
        Load the vocab buffers of the draft model.

        Args:
            file_path (str): The path to the vocab mapping file.
        """
        assert hasattr(self, "t2d") and hasattr(
            self, "d2t"
        ), "t2d and d2t buffersare not found in the draft model, please check your draft model implementation"
        vocab_mapping = torch.load(file_path)
        self.t2d.copy_(vocab_mapping["t2d"])
        self.d2t.copy_(vocab_mapping["d2t"])

    def save_pretrained(self, save_directory, state_dict=None, **kwargs):
        """
        Overrides save_pretrained to handle TP weight aggregation robustly.
        This method gathers sharded weights from all TP ranks and saves a single,
        complete checkpoint from the main process.
        """
        if not dist.is_initialized():
            # Standard non-distributed save
            super().save_pretrained(save_directory, state_dict=state_dict, **kwargs)
            return

        # Use the provided state_dict or get it from the model
        if state_dict is None:
            state_dict = self.state_dict()

        # Get distributed process groups and ranks
        global_rank = dist.get_rank()
        tp_group = get_tp_group()
        tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
        tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0

        # If not using TP, only rank 0 saves and others do nothing.
        if tp_size <= 1:
            if global_rank == 0:
                super().save_pretrained(save_directory, state_dict=state_dict, **kwargs)
            dist.barrier()
            return

        # --- Aggregation Logic for TP > 1 ---
        # Step 1: Each TP rank's leader (tp_rank == 0) will reconstruct the full state dict.
        reconstructed_state_dict = None
        if tp_rank == 0:
            reconstructed_state_dict = {}

        # All ranks in a TP group participate in gathering shards for each parameter.
        modules = dict(self.named_modules())
        for name, param in state_dict.items():
            # Gather shards from all TP ranks into a list
            tensor_list = [torch.empty_like(param) for _ in range(tp_size)]
            dist.all_gather(tensor_list, param.contiguous(), group=tp_group)

            # Let the tp_rank 0 process handle the concatenation
            if tp_rank == 0:
                module_name = ".".join(name.split(".")[:-1])
                module = modules.get(module_name)

                if isinstance(module, ColumnParallelLinear) and name.endswith(
                    ".weight"
                ):
                    # Concat along dimension 0 for ColumnParallel
                    reconstructed_state_dict[name] = torch.cat(tensor_list, dim=0)
                elif isinstance(module, RowParallelLinear) and name.endswith(".weight"):
                    # Concat along dimension 1 for RowParallel
                    reconstructed_state_dict[name] = torch.cat(tensor_list, dim=1)
                else:
                    # Non-parallel layers (biases, norms, etc.) are identical across ranks
                    reconstructed_state_dict[name] = tensor_list[0]

        # Step 2: Only the global rank 0 process saves the final model.
        if global_rank == 0:
            print(f"Rank {global_rank} saving aggregated model checkpoint...")
            super().save_pretrained(
                save_directory, state_dict=reconstructed_state_dict, **kwargs
            )

        # Step 3: Barrier to ensure all processes wait until saving is complete.
        dist.barrier()
