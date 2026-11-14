"""Dynamic, block-local convolution around one attention or FFN sublayer."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import initialization as init

from specforge.config.draft._validation import positive_int
from specforge.config.draft.conv import GroupedCausalConvSpec

from ..registry import register_component_builder


class _KernelProjection(nn.Linear):
    def reset_draft_parameters(self):
        init.zeros_(self.weight)


class GroupedCausalConv(nn.Module):
    """An input/output convolution pair with kernels derived from the input.

    The caller supplies the resolved block width; no algorithm configuration is
    interpreted here. Each block is independent. ``prepare`` returns the input
    convolution and the output-side coefficients; pass those coefficients to
    ``finish`` after executing the sublayer. They stay in the autograd graph,
    never in module state, so overlapping forwards and checkpointing are safe.

    Eager execution preserves per-tap rounding. Training may opt into the fused
    kernel, which accumulates in FP32 and rounds once without changing weights.
    """

    def __init__(
        self,
        config: GroupedCausalConvSpec,
        *,
        hidden_size: int,
        block_size: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if type(config) is not GroupedCausalConvSpec:
            raise TypeError("GroupedCausalConv requires GroupedCausalConvSpec")
        positive_int("hidden_size", hidden_size)
        positive_int("block_size", block_size)
        if config.kernel_size > block_size:
            raise ValueError("kernel_size must not exceed block_size")
        if hidden_size % config.group_size:
            raise ValueError("group_size must divide hidden_size")
        self.config = config
        self.use_fused = False
        self.hidden_size = hidden_size
        self.block_size = block_size
        self.num_groups = hidden_size // config.group_size
        # [input/output side, tap, channel], matching the existing tensor layout.
        self.base_kernel = nn.Parameter(
            torch.empty(2, config.kernel_size, hidden_size, device=device, dtype=dtype)
        )
        self.kernel_projection = _KernelProjection(
            hidden_size,
            2 * config.kernel_size * self.num_groups,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.reset_draft_parameters()

    def reset_draft_parameters(self):
        # Check the full tensor before initializing a view: the view does not
        # inherit HF's loaded-tensor flag during partial checkpoint restoration.
        if not getattr(self.base_kernel, "_is_hf_initialized", False):
            init.zeros_(self.base_kernel)
            init.ones_(self.base_kernel[:, 0])
        self.kernel_projection.reset_draft_parameters()

    def _validate_hidden(self, hidden: Tensor) -> None:
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError("hidden must have shape [batch, sequence, hidden_size]")
        if hidden.shape[1] % self.block_size:
            raise ValueError("sequence length must be divisible by block_size")

    def _convolve(self, hidden: Tensor, delta: Tensor, *, side: int) -> Tensor:
        if self.use_fused:
            if torch.compiler.is_compiling():
                raise ValueError("use_fused_conv does not support torch.compile")
            if (
                not hidden.is_cuda
                or hidden.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                or hidden.dtype != delta.dtype
                or hidden.dtype != self.base_kernel.dtype
            ):
                raise ValueError(
                    "use_fused_conv requires matching CUDA/ROCm FP16/BF16/FP32 inputs and weights"
                )
            from .triton import dflash2_grouped_conv_fused

            if hidden.numel():
                return dflash2_grouped_conv_fused(
                    hidden,
                    delta,
                    self.base_kernel[side],
                    self.block_size,
                    self.config.group_size,
                )
            # Empty inputs use eager arithmetic only to preserve zero gradients.
        batch, length, width = hidden.shape
        blocks = hidden.reshape(
            batch,
            length // self.block_size,
            self.block_size,
            self.num_groups,
            self.config.group_size,
        )
        dynamic = delta.reshape(
            batch,
            length // self.block_size,
            self.block_size,
            self.config.kernel_size,
            self.num_groups,
        )
        base = self.base_kernel[side].reshape(
            1, 1, 1, self.config.kernel_size, self.num_groups, self.config.group_size
        )
        coefficients = base + dynamic.unsqueeze(-1)
        output = coefficients[:, :, :, 0] * blocks
        for tap in range(1, self.config.kernel_size):
            shifted = F.pad(blocks[:, :, : self.block_size - tap], (0, 0, 0, 0, tap, 0))
            output = output + coefficients[:, :, :, tap] * shifted
        return output.reshape(batch, length, width)

    def prepare(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        self._validate_hidden(hidden)
        coefficients = self.kernel_projection(hidden).reshape(
            *hidden.shape[:-1], 2, self.config.kernel_size, self.num_groups
        )
        return (
            self._convolve(hidden, coefficients[..., 0, :, :], side=0),
            coefficients[..., 1, :, :],
        )

    def finish(self, hidden: Tensor, coefficients: Tensor) -> Tensor:
        self._validate_hidden(hidden)
        expected = (*hidden.shape[:-1], self.config.kernel_size, self.num_groups)
        if coefficients.shape != expected:
            raise ValueError("output coefficients must match the sublayer output shape")
        return self._convolve(hidden, coefficients, side=1)


register_component_builder("conv", GroupedCausalConvSpec, GroupedCausalConv)
