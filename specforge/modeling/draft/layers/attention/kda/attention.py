"""Kimi Delta Attention over context blocks, starting from the scanned context.

A block reading ``L`` target-context rows starts its recurrence from the KDA
state after rows ``[0, L)``, the rows its softmax siblings attend to, and its
short convolutions see the last context rows before the block. Parameter names
follow Kimi's ``KimiDeltaAttention`` so serving can load them unchanged.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import initialization as init

from specforge.config.draft._validation import boolean, positive_int
from specforge.config.draft.attention import KDASpec
from specforge.config.draft.norm import NormSpec, RMSNormSpec
from specforge.config.draft.rope import RopeSpec

from ...registry import register_component_builder
from ..backends import AttentionPreparationScope, PreparedLayout
from ..masks import AttentionLayout
from .kernels import fla_kda, reference_kda
from .scan import (
    ContextScanPlan,
    build_context_scan_plan,
    gather_left_context_rows,
    scan_context_states,
)


@dataclass(frozen=True, eq=False)
class PreparedContextScan(PreparedLayout):
    """The scan plan every KDA layer of one forward shares."""

    plan: ContextScanPlan


class KDAShortConvolution(nn.Module):
    """Causal depthwise convolution with SiLU; weight ``[channels, kernel]``."""

    def __init__(self, channels: int, kernel_size: int, *, device=None, dtype=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.empty(channels, kernel_size, device=device, dtype=dtype)
        )
        self.reset_draft_parameters()

    def reset_draft_parameters(self):
        init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, inputs: Tensor) -> Tensor:
        channels_first = F.pad(inputs.transpose(1, 2), (self.kernel_size - 1, 0))
        convolved = F.conv1d(
            channels_first,
            self.weight.unsqueeze(1),
            bias=None,
            groups=self.weight.shape[0],
        )
        return F.silu(convolved.transpose(1, 2))


class KDAGatedRMSNorm(nn.Module):
    """Per-head RMSNorm followed by KDA's sigmoid output gate."""

    def __init__(self, hidden_size: int, eps: float, *, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))
        self.eps = eps
        self.reset_draft_parameters()

    def reset_draft_parameters(self):
        init.ones_(self.weight)

    def forward(self, inputs: Tensor, gate: Tensor) -> Tensor:
        variance = inputs.float().square().mean(dim=-1, keepdim=True)
        normalized = inputs * torch.rsqrt(variance + self.eps).to(inputs.dtype)
        return normalized * self.weight.to(inputs.dtype) * torch.sigmoid(gate)


class KDAAttention(nn.Module):
    """KDA for layouts made of context blocks.

    The recurrence runs on FLA ``chunk_kda`` for CUDA tensors and on the
    reference recurrence on CPU. KDA has no rotary subspace and no softmax
    backend; the shared builder arguments ``rope`` and ``backend`` do not apply.
    """

    def __init__(
        self,
        config: KDASpec,
        *,
        input_size: int,
        hidden_size: int,
        norm: NormSpec,
        rope: RopeSpec,
        backend,
        training: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if type(config) is not KDASpec:
            raise TypeError("KDAAttention requires KDASpec")
        positive_int("input_size", input_size)
        positive_int("hidden_size", hidden_size)
        if type(norm) is not RMSNormSpec:
            raise TypeError("KDA's gated output norm takes its eps from an RMSNorm")
        del rope, backend
        self.config = config
        width = config.projection_size
        options = dict(bias=False, device=device, dtype=dtype)
        self.q_proj = nn.Linear(input_size, width, **options)
        self.k_proj = nn.Linear(input_size, width, **options)
        self.v_proj = nn.Linear(input_size, width, **options)
        conv = dict(device=device, dtype=dtype)
        self.q_conv1d = KDAShortConvolution(
            width, config.short_conv_kernel_size, **conv
        )
        self.k_conv1d = KDAShortConvolution(
            width, config.short_conv_kernel_size, **conv
        )
        self.v_conv1d = KDAShortConvolution(
            width, config.short_conv_kernel_size, **conv
        )
        self.A_log = nn.Parameter(
            torch.empty(config.num_heads, device=device, dtype=dtype)
        )
        self.f_a_proj = nn.Linear(input_size, config.head_dim, **options)
        self.f_b_proj = nn.Linear(config.head_dim, width, **options)
        self.dt_bias = nn.Parameter(torch.empty(width, device=device, dtype=dtype))
        self.b_proj = nn.Linear(input_size, config.num_heads, **options)
        if config.full_rank_output_gate:
            self.g_proj = nn.Linear(input_size, width, **options)
        else:
            self.g_a_proj = nn.Linear(input_size, config.head_dim, **options)
            self.g_b_proj = nn.Linear(config.head_dim, width, **options)
        self.o_norm = KDAGatedRMSNorm(config.head_dim, norm.eps, **conv)
        self.o_proj = nn.Linear(width, hidden_size, **options)
        self.reset_draft_parameters()
        self.train(training)

    def reset_draft_parameters(self):
        # Respect HF's loaded flag: a restored A_log must not be log-transformed.
        if not getattr(self.A_log, "_is_hf_initialized", False):
            with torch.no_grad():
                decay = torch.empty(self.A_log.shape, dtype=torch.float32)
                self.A_log.copy_(decay.uniform_(1, 16).log())
        init.zeros_(self.dt_bias)

    def train(self, mode: bool = True):
        boolean("training", mode)
        return super().train(mode)

    def prepare(
        self,
        layout: AttentionLayout,
        preparation: AttentionPreparationScope,
        *,
        batch_size: int,
    ) -> PreparedContextScan:
        """Plan the context scan once per forward; equal layouts share it."""
        blocks = layout.context_blocks
        if blocks is None:
            raise ValueError("KDA requires a layout described by context_blocks")
        layout.validate_dimensions(batch_size, self.config.num_heads)
        if blocks.context_lengths.shape[0] != batch_size:
            raise ValueError("context blocks must describe every row of the batch")
        return preparation.share(
            (PreparedContextScan, blocks),
            lambda: PreparedContextScan(layout, build_context_scan_plan(blocks)),
        )

    def _gates(self, rows: Tensor) -> tuple[Tensor, Tensor]:
        """Raw decay gate ``[..., H, D]`` and raw beta ``[..., H]``."""
        leading = rows.shape[:-1]
        node = self.config
        gate = self.f_b_proj(self.f_a_proj(rows)).view(
            *leading, node.num_heads, node.head_dim
        )
        return gate, self.b_proj(rows).view(*leading, node.num_heads)

    def _output_gate(self, blocks: Tensor) -> Tensor:
        if self.config.full_rank_output_gate:
            return self.g_proj(blocks)
        return self.g_b_proj(self.g_a_proj(blocks))

    def _context_states(self, context: Tensor, plan: ContextScanPlan) -> Tensor:
        batch, rows = context.shape[:2]
        shape = (batch, rows, self.config.num_heads, self.config.head_dim)
        k = self.k_conv1d(self.k_proj(context)).view(shape)
        v = self.v_conv1d(self.v_proj(context)).view(shape)
        gate, beta = self._gates(context)
        return scan_context_states(
            self._kda_kernel(context),
            k,
            v,
            gate,
            beta,
            self.A_log,
            self.dt_bias,
            self.config.gate_lower_bound,
            plan,
        )

    @staticmethod
    def _kda_kernel(tensor: Tensor):
        return fla_kda if tensor.is_cuda else reference_kda

    @staticmethod
    def _convolve_after(
        conv: KDAShortConvolution, proj: nn.Linear, left_rows: Tensor, rows: Tensor
    ) -> Tensor:
        """Convolve ``rows`` as if ``left_rows`` immediately preceded them."""
        joined = torch.cat((proj(left_rows), proj(rows)), dim=1)
        return conv(joined)[:, left_rows.shape[1] :]

    def forward(
        self,
        hidden: Tensor,
        *,
        attention: PreparedContextScan,
        position_embeddings: tuple[Tensor, Tensor] | None,
        target_context: Tensor | None = None,
    ) -> Tensor:
        """Fused target context is ``[batch, context rows, input_size]``."""
        if not isinstance(attention, PreparedContextScan):
            raise TypeError("KDAAttention requires a PreparedContextScan")
        if position_embeddings is not None:
            raise ValueError(
                "KDA has no rotary subspace; pass position_embeddings=None"
            )
        blocks, plan = attention.layout.context_blocks, attention.plan
        input_size = self.q_proj.in_features
        if (
            hidden.ndim != 3
            or hidden.shape[-1] != input_size
            or hidden.shape[:2] != (plan.batch_size, blocks.num_queries)
        ):
            raise ValueError("hidden must be [batch, block queries, input_size]")
        if target_context is None or target_context.shape != (
            plan.batch_size,
            plan.num_context_rows,
            input_size,
        ):
            raise ValueError(
                "KDA requires target_context [batch, context rows, input_size]"
            )
        node = self.config
        size = blocks.block_size
        count = plan.batch_size * plan.num_blocks
        states = self._context_states(target_context, plan)
        left_rows = gather_left_context_rows(
            target_context, plan.context_lengths, node.short_conv_kernel_size - 1
        ).reshape(count, -1, input_size)
        rows = hidden.reshape(count, size, input_size)
        shape = (count, size, node.num_heads, node.head_dim)
        q, k, v = (
            self._convolve_after(conv, proj, left_rows, rows).view(shape)
            for conv, proj in (
                (self.q_conv1d, self.q_proj),
                (self.k_conv1d, self.k_proj),
                (self.v_conv1d, self.v_proj),
            )
        )
        gate, beta = self._gates(rows)
        output = self._kda_kernel(rows)(
            q,
            k,
            v,
            gate,
            beta,
            self.A_log,
            self.dt_bias,
            node.gate_lower_bound,
            initial_state=states.reshape(
                count, node.num_heads, node.head_dim, node.head_dim
            ),
        )
        output = self.o_norm(output, self._output_gate(rows).view(shape))
        output = self.o_proj(output.flatten(-2)).reshape(*hidden.shape[:2], -1)
        # Inactive blocks stay zero, as for every attention kind.
        return output.masked_fill(~attention.layout.active_queries.unsqueeze(-1), 0)


register_component_builder("attention", KDASpec, KDAAttention)
