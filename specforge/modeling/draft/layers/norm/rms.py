"""RMS normalization with explicit stored-weight and casting semantics."""

import torch
from torch import Tensor, nn
from transformers import initialization as init

from specforge.config.draft._validation import positive_int
from specforge.config.draft.norm import RMSNormSpec

from ..registry import register_component_builder


class RMSNorm(nn.Module):
    def __init__(
        self, config: RMSNormSpec, *, hidden_size: int, device=None, dtype=None
    ):
        super().__init__()
        if type(config) is not RMSNormSpec:
            raise TypeError("RMSNorm requires RMSNormSpec")
        positive_int("hidden_size", hidden_size)
        self.config = config
        self.use_liger_kernel = False
        self.weight = (
            nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))
            if config.has_weight
            else None
        )
        self.reset_draft_parameters()

    def reset_draft_parameters(self):
        # HF restoration can mark a loaded tensor before initializing missing ones.
        if self.weight is not None:
            init.constant_(self.weight, 1 - self.config.weight_offset)

    def forward(self, hidden: Tensor) -> Tensor:
        if self.use_liger_kernel and self.weight is not None:
            from liger_kernel.ops.rms_norm import LigerRMSNormFunction

            return LigerRMSNormFunction.apply(
                hidden,
                self.weight,
                self.config.eps,
                self.config.weight_offset,
                "gemma" if self.config.weight_offset else "llama",
                False,
                None,
            )
        input_dtype = hidden.dtype
        hidden = hidden.to(torch.float32)
        variance = hidden.pow(2).mean(-1, keepdim=True)
        hidden = hidden * torch.rsqrt(variance + self.config.eps)
        if self.weight is None:
            return hidden.to(input_dtype)
        if self.config.weight_offset == 0:
            # Ordinary RMSNorm casts the normalized activations before scaling.
            return self.weight * hidden.to(input_dtype)
        # The 1+w parameterization scales in FP32 before the final cast.
        return (hidden * (1.0 + self.weight.float())).to(input_dtype)


register_component_builder("norm", RMSNormSpec, RMSNorm)
