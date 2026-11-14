"""Shared context preparation for independently composed draft models."""

from collections.abc import Sequence
from dataclasses import replace

from torch import Tensor

from ..layers.inputs import LayerInputs, MoELayerOutput
from .target_features import TargetFeatureFusion


def prepare_context_inputs(
    fusion: TargetFeatureFusion,
    features: Tensor,
    inputs: Sequence[LayerInputs],
    *,
    num_layers: int,
) -> list[LayerInputs]:
    if len(inputs) != num_layers:
        raise ValueError("layer_inputs must contain one entry per layer")
    for entry in inputs:
        if not isinstance(entry, LayerInputs):
            raise TypeError("each layer input must be LayerInputs")
        if entry.target_context is not None:
            raise ValueError(
                "target_context must be computed through the model's trainable fusion"
            )
    context = fusion(features)
    return [replace(entry, target_context=context) for entry in inputs]


def run_backbone(backbone, hidden: Tensor, layer_inputs: Sequence[LayerInputs]):
    """The backbone's forward through its hooks, and each MoE layer's routing (or None)."""
    hidden, outputs = backbone(hidden, layer_inputs, with_layer_outputs=True)
    routings = tuple(o.routing for o in outputs if isinstance(o, MoELayerOutput))
    return hidden, routings or None
