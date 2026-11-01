"""Block-local dynamic convolution used around DFlash2 attention and FFN."""

from __future__ import annotations

import functools
import os
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

# Set to "0" to force the eager convolution on CUDA as well.
FUSED_CONV_ENV = "SPECFORGE_DFLASH2_FUSED_CONV"
_FUSED_CONV_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@functools.lru_cache(maxsize=1)
def _load_fused_grouped_conv() -> Optional[Tuple[Callable, Callable]]:
    try:
        from specforge.modeling.draft.models.layers.convolution.dflash2_triton import (
            dflash2_grouped_conv_fused,
            supports_group_size,
        )
    except ModuleNotFoundError as exc:
        name = exc.name or ""
        if name == "triton" or name.startswith("triton."):
            return None
        raise
    return dflash2_grouped_conv_fused, supports_group_size


class DFlashGroupedConv(nn.Module):
    """Grouped dynamic depthwise convolution within each proposal block.

    One projection of the sublayer input produces the dynamic kernel deltas
    for both the input and output side.  ``base_kernel`` starts as an identity,
    which makes enabling the module a stable extension of a DFlash backbone.
    """

    def __init__(
        self,
        hidden_size: int,
        block_size: int,
        taps: int,
        group_size: int,
    ) -> None:
        super().__init__()
        if taps < 1:
            raise ValueError("DFlash2 conv_kernel_size must be >= 1")
        if taps > block_size:
            raise ValueError(
                "DFlash2 conv_kernel_size must not exceed block_size, got "
                f"conv_kernel_size={taps}, block_size={block_size}"
            )
        if group_size < 1 or hidden_size % group_size:
            raise ValueError(
                f"DFlash2 conv_group_size={group_size} must divide "
                f"hidden_size={hidden_size}"
            )

        self.block_size = int(block_size)
        self.taps = int(taps)
        self.group_size = int(group_size)
        self.num_groups = int(hidden_size) // self.group_size

        # [input/output side, tap, channel], matching SGLang's loader contract.
        base_kernel = torch.zeros(2, self.taps, int(hidden_size))
        base_kernel[:, 0] = 1.0
        self.base_kernel = nn.Parameter(base_kernel)
        self.kernel_projection = nn.Linear(
            int(hidden_size),
            2 * self.taps * self.num_groups,
            bias=False,
        )
        nn.init.zeros_(self.kernel_projection.weight)

    def _convolve(
        self,
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
        *,
        side: int,
    ) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        if sequence_length % self.block_size:
            raise ValueError(
                "DFlash2 convolution sequence length must be divisible by "
                f"block_size={self.block_size}, got {sequence_length}"
            )

        base_kernel = self.base_kernel[side]
        fused = self._fused_convolution(hidden_states, delta, base_kernel)
        if fused is not None:
            return fused(
                hidden_states,
                delta,
                base_kernel,
                self.block_size,
                self.group_size,
            )

        num_blocks = sequence_length // self.block_size
        blocks = hidden_states.reshape(
            batch_size,
            num_blocks,
            self.block_size,
            self.num_groups,
            self.group_size,
        )
        dynamic = delta.reshape(
            batch_size,
            num_blocks,
            self.block_size,
            self.taps,
            self.num_groups,
        )
        base = base_kernel.reshape(
            1,
            1,
            1,
            self.taps,
            self.num_groups,
            self.group_size,
        )
        coefficients = base + dynamic.unsqueeze(-1)

        output = coefficients[:, :, :, 0] * blocks
        for tap in range(1, self.taps):
            shifted = F.pad(
                blocks[:, :, : self.block_size - tap],
                (0, 0, 0, 0, tap, 0),
            )
            output = output + coefficients[:, :, :, tap] * shifted
        return output.reshape(batch_size, sequence_length, hidden_size)

    def _fused_convolution(
        self,
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
        base_kernel: torch.Tensor,
    ) -> Optional[Callable]:
        """Return the Triton convolution when it can replace the eager path.

        The fused kernels compute the same sum with FP32 accumulation and one
        rounding per output. They require CUDA tensors of one floating dtype
        and a power-of-two group size; everything else stays eager.
        """

        if os.environ.get(FUSED_CONV_ENV, "1") == "0":
            return None
        if not (hidden_states.is_cuda and delta.is_cuda and base_kernel.is_cuda):
            return None
        if hidden_states.numel() == 0:
            return None
        if hidden_states.dtype not in _FUSED_CONV_DTYPES or not (
            hidden_states.dtype == delta.dtype == base_kernel.dtype
        ):
            return None
        if torch.compiler.is_compiling():
            return None
        fused = _load_fused_grouped_conv()
        if fused is None:
            return None
        convolve, supports_group_size = fused
        if not supports_group_size(self.group_size):
            return None
        return convolve

    def prepare(self, hidden_states: torch.Tensor):
        coefficients = self.kernel_projection(hidden_states).reshape(
            *hidden_states.shape[:-1],
            2,
            self.taps,
            self.num_groups,
        )
        return (
            self._convolve(hidden_states, coefficients[..., 0, :, :], side=0),
            coefficients[..., 1, :, :],
        )

    def finish(
        self,
        hidden_states: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, side=1)
