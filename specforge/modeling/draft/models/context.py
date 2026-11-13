"""Shared context preparation for independently composed draft models."""

from collections.abc import Sequence
from dataclasses import replace

from torch import Tensor

from ..layers.inputs import ExpertRouting, LayerInputs, LayerOutput, MoELayerOutput
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


def collect_expert_routing(
    outputs: Sequence[LayerOutput | None],
) -> tuple[ExpertRouting, ...] | None:
    """Each MoE layer's expert routing among the typed layer outputs, in layer order;
    None without MoE."""
    routings = tuple(o.routing for o in outputs if isinstance(o, MoELayerOutput))
    return routings or None


def run_backbone(backbone, hidden: Tensor, layer_inputs: Sequence[LayerInputs]):
    """The backbone's forward, through its hooks, and each MoE layer's expert routing."""
    hidden, outputs = backbone(hidden, layer_inputs, with_layer_outputs=True)
    return hidden, collect_expert_routing(outputs)
