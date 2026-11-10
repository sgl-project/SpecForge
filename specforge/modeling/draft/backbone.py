"""One decoder execution path, usable sequentially or from a model-owned loop."""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask
from torch.utils.checkpoint import checkpoint

from .layers.attention.backends import PreparedAttention
from .layers.residual import ResidualStreams
from .layers.rope.base import RotaryEmbedding


@dataclass(frozen=True, eq=False, kw_only=True)
class LayerInputs:
    """Prepared runtime inputs, without algorithm settings or mutable KV caches.

    Attention is prepared once by the model and reused across equivalent layers.
    The legacy attention_mask field accepts tensor or FlexAttention masks.
    RoPE can be supplied directly or computed from this layer's position_ids.
    Tensor contents must not be mutated while their autograd graph is live.
    """

    attention: PreparedAttention | None = None
    target_context: Tensor | None = None
    attention_mask: Tensor | BlockMask | None = None
    position_ids: Tensor | None = None
    position_embeddings: tuple[Tensor, Tensor] | None = None


@dataclass(frozen=True, eq=False)
class BackboneState:
    hidden: Tensor
    layer_inputs: tuple[LayerInputs, ...]
    next_layer: int
    _owner: object = field(repr=False)

    def with_hidden(self, hidden: Tensor) -> "BackboneState":
        """Replace a layer result without detaching its graph or changing layout."""
        if hidden.shape != self.hidden.shape:
            raise ValueError(
                "replacement hidden state must preserve the residual layout"
            )
        return replace(self, hidden=hidden)


class DraftBackbone(nn.Module):
    """Own modules and execution order; architectures own fusion and injections.

    Each layer implements layer(hidden, inputs: LayerInputs) -> Tensor. Builders
    provide validated components; this class does not interpret configuration.
    Mutable serving caches are not part of this initial execution interface.
    """

    def __init__(
        self,
        layers: Sequence[nn.Module],
        *,
        final_norm: nn.Module,
        rotary: Sequence[nn.Module | None],
        residual: ResidualStreams,
    ):
        super().__init__()
        if not layers or len(rotary) != len(layers):
            raise ValueError("backbone requires layers and one rotary entry per layer")
        self.layers = nn.ModuleList(layers)
        self.final_norm = final_norm
        self.rotary = nn.ModuleList(rotary)
        self.residual = residual
        self._state_owner = object()
        self.gradient_checkpointing = False

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def begin(
        self, hidden: Tensor, layer_inputs: Sequence[LayerInputs]
    ) -> BackboneState:
        if len(layer_inputs) != self.num_layers:
            raise ValueError("layer_inputs must contain one entry per layer")
        prepared = []
        rotary_tables = {}
        hidden = self.residual.expand(hidden)
        for inputs, rotary in zip(layer_inputs, self.rotary):
            if not isinstance(inputs, LayerInputs):
                raise TypeError("each layer input must be a LayerInputs instance")
            if rotary is not None and inputs.position_embeddings is None:
                if inputs.position_ids is None:
                    raise ValueError("rotary preparation requires position_ids")
                # Built-in rotary modules are determined by their frozen node.
                # Unknown modules share only when the module itself is shared.
                rule = (
                    (type(rotary), rotary.config, rotary.head_dim)
                    if isinstance(rotary, RotaryEmbedding)
                    else id(rotary)
                )
                key = (rule, id(inputs.position_ids))
                if key not in rotary_tables:
                    rotary_tables[key] = rotary(hidden, inputs.position_ids)
                inputs = replace(inputs, position_embeddings=rotary_tables[key])
            prepared.append(inputs)
        return BackboneState(hidden, tuple(prepared), 0, self._state_owner)

    def _check_state(self, state: BackboneState) -> None:
        if state._owner is not self._state_owner:
            raise ValueError("state belongs to another backbone")

    def run_next_layer(self, state: BackboneState) -> BackboneState:
        self._check_state(state)
        if not 0 <= state.next_layer < self.num_layers:
            raise ValueError("no next layer remains")
        layer = self.layers[state.next_layer]
        inputs = state.layer_inputs[state.next_layer]
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            # Non-reentrant checkpointing retains gradients for nested context/RoPE
            # tensors, and works when inputs are frozen but layer weights are not.
            # Pass the selected module directly: recomputation must not close over
            # an advancing layer index. Normal module hooks still run.
            hidden = checkpoint(layer, state.hidden, inputs, use_reentrant=False)
        else:
            hidden = layer(state.hidden, inputs)
        return replace(state, hidden=hidden, next_layer=state.next_layer + 1)

    def finalize(self, state: BackboneState) -> Tensor:
        self._check_state(state)
        if state.next_layer != self.num_layers:
            raise ValueError("finalize requires every layer to have run")
        return self.final_norm(self.residual.reduce(state.hidden))

    def forward(self, hidden: Tensor, layer_inputs: Sequence[LayerInputs]) -> Tensor:
        state = self.begin(hidden, layer_inputs)
        for _ in self.layers:
            state = self.run_next_layer(state)
        return self.finalize(state)
