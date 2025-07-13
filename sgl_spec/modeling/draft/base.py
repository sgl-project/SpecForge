import glob
import json
import os
from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import PreTrainedModel

from sgl_spec.modeling._mask_utils import _expand_mask, _make_causal_mask


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

    def load_embedding(
        self, model_path: str, embedding_name: str = "embed_tokens"
    ) -> None:
        """
        Load the embedding of the draft model.

        Args:
            model_path (str): The path to the huggingface repository.
        """
        embedding_weight_name = f"{embedding_name}.weight"

        if os.path.exists(model_path):
            # model_path is a local directory
            # check if there is file ending with index.json
            glob_path = os.path.join(model_path, "**", "*.safetensors.index.json")
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
            ckpt_file = index_json["weight_map"][embedding_weight_name]

            if ckpt_file.endswith(".safetensors"):
                with safe_open(
                    os.path.join(model_path, ckpt_file), framework="pt"
                ) as f:
                    emb_tokens = f.get_tensor(embedding_weight_name)
            else:
                state_dict = torch.load(os.path.join(model_path, ckpt_file))
                emb_tokens = state_dict[embedding_name]
            self.embed_tokens.weight.copy_(emb_tokens)
        else:
            # this is the case where model_path is a huggingface repository
            # we first need to locate its local cache
            local_cache_path = snapshot_download(repo_id=model_path)
            self.load_embedding(local_cache_path, embedding_name)

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
