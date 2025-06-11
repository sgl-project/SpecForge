import torch
import torch.nn as nn
from typing import Optional, Tuple

class TrainableKVCache(nn.Module):
    """
    A version of KVCache that supports backpropagation.
    It preserves the computation graph by returning new, concatenated tensors
    instead of performing in-place modifications. This module itself is stateless;
    the actual cache tensors are managed and passed externally.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        k_new: torch.Tensor, 
        v_new: torch.Tensor, 
        past_k: Optional[torch.Tensor] = None, 
        past_v: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the KV cache. If a past cache exists, it appends the new k/v values.

        Args:
            k_new (torch.Tensor): The new key tensor for the current step,
                                  with shape [B, H, S_new, D].
            v_new (torch.Tensor): The new value tensor for the current step,
                                  with shape [B, H, S_new, D].
            past_k (Optional[torch.Tensor]): The accumulated key cache from all previous steps.
            past_v (Optional[torch.Tensor]): The accumulated value cache from all previous steps.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The updated, complete key and value cache tensors.
        """
        # If this is the first update (no past cache exists)
        if past_k is None:
            return k_new, v_new

        # Use torch.cat for concatenation, which is a differentiable operation.
        # We concatenate along the sequence length dimension (dim=2).
        updated_k = torch.cat([past_k, k_new], dim=2)
        updated_v = torch.cat([past_v, v_new], dim=2)

        return updated_k, updated_v