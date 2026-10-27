"""Compose attention and FFN as a pre-norm residual decoder layer."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from torch import nn
from transformers.modeling_layers import GradientCheckpointingLayer

from .attention.base import Attention, AttentionContext, AttentionOutput

State = TypeVar("State")


@dataclass(frozen=True)
class DecoderOutput(Generic[State]):
    hidden_states: torch.Tensor
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
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = ffn
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: AttentionContext,
        state: State | None = None,
        **backend_kwargs,
    ) -> DecoderOutput[State]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        output = self.self_attn(hidden_states, context, state, **backend_kwargs)
        hidden_states = residual + output.hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return DecoderOutput(hidden_states, output)
