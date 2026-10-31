"""Compose attention and FFN with independently injected residual sublayers."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from torch import nn
from transformers.modeling_layers import GradientCheckpointingLayer

from .attention.base import Attention, AttentionContext, AttentionOutput
from .factory import build_residual
from .ffn.moe.layer import MoELayer
from .residual.base import Residual, ResidualStream

State = TypeVar("State")


@dataclass(frozen=True)
class DecoderOutput(Generic[State]):
    hidden_states: ResidualStream
    attention: AttentionOutput[State]


class DecoderLayer(GradientCheckpointingLayer, Generic[State]):
    """Own modules at their historical paths; algorithms supply the context.

    Special input preparation (EAGLE3) and sublayer convolutions (DFlash2)
    remain in the algorithm layers instead of adding branches to this flow.
    """

    def __init__(
        self,
        attention: Attention[State],
        ffn: nn.Module,
        input_layernorm: nn.Module,
        post_attention_layernorm: nn.Module,
        *,
        attention_residual: Residual | None = None,
        ffn_residual: Residual | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = ffn
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.attention_residual = (
            attention_residual if attention_residual is not None else build_residual()
        )
        self.ffn_residual = (
            ffn_residual if ffn_residual is not None else build_residual()
        )

    def forward(
        self,
        hidden_states: ResidualStream,
        context: AttentionContext,
        state: State | None = None,
        moe_token_mask: torch.Tensor | None = None,
        **backend_kwargs,
    ) -> DecoderOutput[State]:
        hidden_states, residual = self.attention_residual.read(
            hidden_states, self.input_layernorm
        )
        output = self.self_attn(hidden_states, context, state, **backend_kwargs)
        hidden_states = self.attention_residual.write(output.hidden_states, residual)
        hidden_states, residual = self.ffn_residual.read(
            hidden_states, self.post_attention_layernorm
        )
        hidden_states = self.ffn_residual.write(
            self._ffn(hidden_states, moe_token_mask), residual
        )
        return DecoderOutput(hidden_states, output)

    def _ffn(
        self, hidden_states: torch.Tensor, moe_token_mask: torch.Tensor | None
    ) -> torch.Tensor:
        # Only an MoELayer takes the mask; the dense MLP keeps its one-argument call.
        if moe_token_mask is None or not isinstance(self.mlp, MoELayer):
            return self.mlp(hidden_states)
        return self.mlp(hidden_states, moe_token_mask)
