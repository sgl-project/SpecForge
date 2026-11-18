"""The ordinary single-stream residual layout needs no boundary transforms."""

from collections.abc import Callable
from typing import Literal

from torch import Tensor

from specforge.config.draft._validation import positive_int
from specforge.config.draft.residual import StandardResidualSpec

from ..registry import register_component_builder
from .base import ResidualConnection, ResidualStreams


class StandardResidualConnection(ResidualConnection):
    def __init__(self, config: StandardResidualSpec):
        super().__init__()
        self.config = config

    def forward(self, hidden: Tensor, sublayer: Callable[[Tensor], Tensor]) -> Tensor:
        update = sublayer(hidden)
        if not isinstance(update, Tensor) or update.shape != hidden.shape:
            raise ValueError(
                "residual sublayer must return a tensor with the input shape"
            )
        return hidden + update


class StandardResidual(ResidualStreams):
    def __init__(self, config: StandardResidualSpec):
        super().__init__()
        if type(config) is not StandardResidualSpec:
            raise TypeError("StandardResidual requires StandardResidualSpec")
        self.config = config

    def build_connection(
        self,
        *,
        hidden_size: int,
        layer_index: int,
        sublayer: Literal["attention", "mlp"],
        device=None,
        dtype=None,
    ) -> ResidualConnection:
        positive_int("hidden_size", hidden_size)
        if type(layer_index) is not int or layer_index < 0:
            raise ValueError("layer_index must be a nonnegative integer")
        if sublayer not in {"attention", "mlp"}:
            raise ValueError("sublayer must be attention or mlp")
        return StandardResidualConnection(self.config)

    def expand(self, hidden: Tensor) -> Tensor:
        return hidden

    def reduce(self, hidden: Tensor) -> Tensor:
        return hidden


register_component_builder("residual", StandardResidualSpec, StandardResidual)
