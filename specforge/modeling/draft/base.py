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
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers.cache_utils import Cache
from transformers.modeling_utils import PreTrainedModel

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

    @abstractmethod
    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project the concatenated hidden states from the high, medium and low layers to the target hidden size.
        """

    @abstractmethod
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute the logits of the draft model.
        """

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

    def freeze_embedding(self) -> None:
        """
        Freeze the embeddings of the draft model so that they are not updated during training.
        """
        self.embed_tokens.weight.requires_grad = False

    def freeze_lm_head(self) -> None:
        """
        Freeze the lm_head of the draft model so that it is not updated during training.
        """
        self.lm_head.weight.requires_grad = False

    @torch.no_grad()
    def _load_tensor_from_checkpoint(
        self, model_path: str, tensor_key: str
    ) -> torch.Tensor:
        """
        Load a single tensor from a target model checkpoint.

        Args:
            model_path (str): Path to the target model. Can be either a Hugging Face
                repository ID or a local directory path containing the model files.
            tensor_key (str): The key of the tensor to load from the checkpoint.

        Returns:
            torch.Tensor: The loaded tensor.
        """
        if not os.path.exists(model_path):
            # model_path is a huggingface repository, download first
            model_path = snapshot_download(repo_id=model_path)

        glob_path = os.path.join(model_path, "*.index.json")
        index_json_paths = glob.glob(glob_path)

        if len(index_json_paths) == 0:
            # No index.json found, look for single model file
            safetensors_path = os.path.join(model_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                with safe_open(safetensors_path, framework="pt") as f:
                    return f.get_tensor(tensor_key)

            pytorch_model_path = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                state_dict = torch.load(pytorch_model_path, map_location="cpu")
                return state_dict[tensor_key]

            raise FileNotFoundError(
                f"No index.json, model.safetensors or pytorch_model.bin found in {model_path}"
            )

        if len(index_json_paths) > 1:
            raise FileNotFoundError(f"Multiple index.json files found in {model_path}")

        with open(index_json_paths[0], "r") as f:
            index_json = json.load(f)
        ckpt_file = index_json["weight_map"][tensor_key]

        if ckpt_file.endswith(".safetensors"):
            with safe_open(os.path.join(model_path, ckpt_file), framework="pt") as f:
                return f.get_tensor(tensor_key)
        else:
            state_dict = torch.load(os.path.join(model_path, ckpt_file))
            return state_dict[tensor_key]

    @torch.no_grad()
    def load_embedding(
        self, model_path: str, embedding_key: str = "language_model.embed_tokens.weight"
    ) -> None:
        """
        Load the embedding of the draft model from the target model checkpoint.

        Args:
            model_path (str): Path to the target model. Can be either a Hugging Face
                repository ID or a local directory path containing the model files.
            embedding_key (str): The key of the embedding weight in the checkpoint.
        """
        tensor = self._load_tensor_from_checkpoint(model_path, embedding_key)
        self.embed_tokens.weight.copy_(tensor)

    @torch.no_grad()
    def load_lm_head(
        self, model_path: str, lm_head_key: str, embedding_key: str
    ) -> None:
        """
        Load the lm_head of the draft model from the target model checkpoint.

        For models with tied weights (embed_tokens == lm_head), the lm_head key
        may not exist in the checkpoint. In that case, falls back to loading
        from the embedding key.

        Args:
            model_path (str): Path to the target model. Can be either a Hugging Face
                repository ID or a local directory path containing the model files.
            lm_head_key (str): The key of the lm_head weight in the checkpoint.
            embedding_key (str): Fallback key if lm_head_key is not found (for
                models with tie_word_embeddings=True).
        """
        try:
            tensor = self._load_tensor_from_checkpoint(model_path, lm_head_key)
        except KeyError:
            # Target model ties weights -- lm_head key doesn't exist in checkpoint,
            # fall back to embedding key
            tensor = self._load_tensor_from_checkpoint(model_path, embedding_key)
        self.lm_head.weight.copy_(tensor)

    def load_vocab_mapping(self, file_path: str) -> None:
        """
        Load the vocab buffers of the draft model.

        Args:
            file_path (str): The path to the vocab mapping file.
        """
        assert hasattr(self, "t2d") and hasattr(self, "d2t"), (
            "t2d and d2t buffersare not found in the draft model, please check your draft model implementation"
        )
        vocab_mapping = torch.load(file_path)
        self.t2d.copy_(vocab_mapping["t2d"])
        self.d2t.copy_(vocab_mapping["d2t"])
        self.vocab_mapping_loaded = True
