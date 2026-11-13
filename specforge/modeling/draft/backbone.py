"""One decoder execution path, usable sequentially or from a model-owned loop."""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .layers.inputs import LayerInputs, LayerOutput
from .layers.residual import ResidualStreams
from .layers.rope import NODE_DETERMINED_ROTARY_TYPES


def _call_layer(layer, hidden, inputs, layer_kwargs):
    """Keep layer keyword arguments separate from checkpoint control options."""
    return layer(hidden, inputs, **layer_kwargs)


@dataclass(frozen=True, eq=False)
class BackboneState:
    hidden: Tensor
    layer_inputs: tuple[LayerInputs, ...]
    next_layer: int
    _owner: object = field(repr=False)
    layer_output: LayerOutput | None = None

    def with_hidden(self, hidden: Tensor) -> "BackboneState":
        """Replace a layer result without detaching its graph or changing layout."""
        if hidden.shape != self.hidden.shape:
            raise ValueError(
                "replacement hidden state must preserve the residual layout"
            )
        output = self.layer_output
        return replace(
            self,
            hidden=hidden,
            layer_output=replace(output, hidden=hidden) if output is not None else None,
        )


class DraftBackbone(nn.Module):
    """Own modules and execution order; architectures own fusion and injections.

    Layers return a Tensor or a LayerOutput with additional typed tensors. Builders
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
        rotary_tables = []
        hidden = self.residual.expand(hidden)
        for inputs, rotary in zip(layer_inputs, self.rotary):
            if not isinstance(inputs, LayerInputs):
                raise TypeError("each layer input must be a LayerInputs instance")
            if rotary is not None and inputs.position_embeddings is None:
                if inputs.position_ids is None:
                    raise ValueError("rotary preparation requires position_ids")
                # Built-in rotary modules are determined by their frozen node.
                # Any other module shares only when the module itself is shared.
                for cached, positions, tables in rotary_tables:
                    same_rule = cached is rotary or (
                        type(rotary) in NODE_DETERMINED_ROTARY_TYPES
                        and type(cached) is type(rotary)
                        and cached.config == rotary.config
                        and cached.head_dim == rotary.head_dim
                    )
                    if same_rule and positions is inputs.position_ids:
                        break
                else:
                    tables = rotary(hidden, inputs.position_ids)
                    rotary_tables.append((rotary, inputs.position_ids, tables))
                inputs = replace(inputs, position_embeddings=tables)
            prepared.append(inputs)
        return BackboneState(hidden, tuple(prepared), 0, self._state_owner)

    def _check_state(self, state: BackboneState) -> None:
        if state._owner is not self._state_owner:
            raise ValueError("state belongs to another backbone")

    def run_next_layer(self, state: BackboneState, **layer_kwargs) -> BackboneState:
        """Run one stage; optional arguments and extra results belong to that stage."""
        self._check_state(state)
        if not 0 <= state.next_layer < self.num_layers:
            raise ValueError("no next layer remains")
        layer = self.layers[state.next_layer]
        inputs = state.layer_inputs[state.next_layer]
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            # Non-reentrant checkpointing retains gradients for nested context/RoPE
            # tensors, and works when inputs are frozen but layer weights are not.
            # Pass the module and its kwargs as positional inputs, so checkpoint
            # cannot consume a layer argument such as debug or use_reentrant.
            result = checkpoint(
                _call_layer,
                layer,
                state.hidden,
                inputs,
                layer_kwargs,
                use_reentrant=False,
            )
        else:
            result = _call_layer(layer, state.hidden, inputs, layer_kwargs)
        output = result if isinstance(result, LayerOutput) else None
        return replace(
            state,
            hidden=output.hidden if output is not None else result,
            next_layer=state.next_layer + 1,
            layer_output=output,
        )

    def finalize(self, state: BackboneState) -> Tensor:
        self._check_state(state)
        if state.next_layer != self.num_layers:
            raise ValueError("finalize requires every layer to have run")
        return self.final_norm(self.residual.reduce(state.hidden))

    def forward(
        self,
        hidden: Tensor,
        layer_inputs: Sequence[LayerInputs],
        *,
        with_layer_outputs: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[LayerOutput | None, ...]]:
        """The layer loop; `with_layer_outputs` also returns every layer's typed output, None for a plain layer."""
        state = self.begin(hidden, layer_inputs)
        outputs = []
        for _ in self.layers:
            state = self.run_next_layer(state)
            # The state keeps only the latest layer's output, so collect each one as it completes.
            if with_layer_outputs:
                outputs.append(state.layer_output)
        hidden = self.finalize(state)
        return (hidden, tuple(outputs)) if with_layer_outputs else hidden
