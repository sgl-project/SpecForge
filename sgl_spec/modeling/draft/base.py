import torch
import torch.nn as nn
from typing import Tuple
from abc import ABC, abstractmethod
from transformers import PreTrainedModel


class Eagle3DraftModel(PreTrainedModel, ABC):
    """
    This is the base class for the Eagle3 draft model implementation. The child class needs to implement
    the abstract methods to support training with TTT.
    """


    @abstractmethod
    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
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
    def prepare_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Prepare the attention mask of the draft model.
        """
        pass

    @abstractmethod
    def backbone(self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        use_cache: bool = True
        ) -> torch.Tensor:
        """
        The backbone of the draft model.
        """
        pass