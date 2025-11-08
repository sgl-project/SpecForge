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
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import PretrainedConfig
from transformers.cache_utils import Cache
from transformers.modeling_utils import PreTrainedModel

from specforge.distributed import (
    gather_tensor,
    get_draft_tp_group,
    get_draft_tp_rank,
    get_draft_tp_size,
)
from specforge.model.linear import ColumnParallelLinear, RowParallelLinear
from specforge.utils import print_with_rank


@torch.no_grad()
def load_param_from_target_model(
    target_model_path: str, param_key: str, param: nn.Parameter
) -> None:
    """
    Load the parameter from the target model.

    Args:
        target_model_path (str): Path to the target model. Can be either a Hugging Face
        repository ID or a local directory path containing the model files.
        param_key (str): The key of the parameter to load.
        param (nn.Parameter): The parameter to load.
    """
    if os.path.exists(target_model_path):
        # model_path is a local directory
        # check if there is file ending with index.json
        glob_path = os.path.join(target_model_path, "*.index.json")
        index_json_path = glob.glob(glob_path)

        if len(index_json_path) == 0:
            # No index.json found, look for single model file
            safetensors_path = os.path.join(target_model_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                with safe_open(safetensors_path, framework="pt") as f:
                    param.copy_(f.get_tensor(param_key))
                return

            pytorch_model_path = os.path.join(target_model_path, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                state_dict = torch.load(pytorch_model_path, map_location="cpu")
                param.copy_(state_dict[param_key])
                return

            raise FileNotFoundError(
                f"No index.json, model.safetensors or pytorch_model.bin found in {target_model_path}"
            )
        if len(index_json_path) > 1:
            raise FileNotFoundError(
                f"Multiple index.json files found in {target_model_path}"
            )
        index_json_path = index_json_path[0]

        with open(index_json_path, "r") as f:
            index_json = json.load(f)
        ckpt_file = index_json["weight_map"][param_key]

        if ckpt_file.endswith(".safetensors"):
            with safe_open(
                os.path.join(target_model_path, ckpt_file), framework="pt"
            ) as f:
                param.copy_(f.get_tensor(param_key))
        else:
            state_dict = torch.load(os.path.join(target_model_path, ckpt_file))
            param.copy_(state_dict[param_key])
    else:
        # this is the case where model_path is a huggingface repository
        # we first need to locate its local cache
        local_cache_path = snapshot_download(repo_id=target_model_path)
        load_param_from_target_model(local_cache_path, param_key, param)


class Eagle3DraftModel(PreTrainedModel, ABC):
    """
    This is the base class for the Eagle3 draft model implementation. The child class needs to implement
    the abstract methods to support training with TTT.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = config

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

    def get_embedding_param(self) -> nn.Parameter:
        """
        Get the embedding parameter of the draft model.
        """
        return self.embed_tokens.weight

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
        self.vocab_mapping_loaded = True

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
        tp_group = get_draft_tp_group()
        tp_size = get_draft_tp_size()
        tp_rank = get_draft_tp_rank()

        if tp_size > 1:
            # --- Aggregation Logic for TP > 1 ---
            reconstructed_state_dict = {}
            # All ranks in a TP group participate in gathering shards for each parameter.
            modules = dict(self.named_modules())
            state_dict = dict(sorted(state_dict.items()))
            for name, param in state_dict.items():
                # Gather shards from all TP ranks into a list
                module_name = ".".join(name.split(".")[:-1])
                module = modules.get(module_name)

                tensor = param
                if name.endswith(".weight"):
                    if isinstance(module, ColumnParallelLinear):
                        tensor = gather_tensor(tensor, tp_group, 0)
                    elif isinstance(module, RowParallelLinear):
                        tensor = gather_tensor(tensor, tp_group, 1)

                reconstructed_state_dict[name] = tensor
            state_dict = reconstructed_state_dict

        #  Only the TP Rank 0 process saves the final model.
        if tp_rank == 0:
            print_with_rank(f"TP Rank {tp_rank} saving aggregated model checkpoint...")
            super().save_pretrained(save_directory, state_dict=state_dict, **kwargs)

        # Barrier to ensure all processes wait until saving is complete.
        dist.barrier(group=tp_group)
