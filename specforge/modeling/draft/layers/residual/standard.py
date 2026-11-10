"""The ordinary single-stream residual layout needs no boundary transforms."""

from collections.abc import Callable

from torch import Tensor

from specforge.config.draft._validation import positive_int
from specforge.config.draft.residual import StandardResidualSpec

from ..registry import register_component_builder
from .base import ResidualConnection, ResidualStreams


class StandardResidualConnection(ResidualConnection):
    def forward(self, hidden: Tensor, sublayer: Callable[[Tensor], Tensor]) -> Tensor:
        return hidden + sublayer(hidden)


class StandardResidual(ResidualStreams):
    def __init__(self, config: StandardResidualSpec):
        super().__init__()
        if type(config) is not StandardResidualSpec:
            raise TypeError("StandardResidual requires StandardResidualSpec")
        self.config = config

    def build_connection(
        self, *, hidden_size: int, device=None, dtype=None
    ) -> ResidualConnection:
        positive_int("hidden_size", hidden_size)
        return StandardResidualConnection()

    def expand(self, hidden: Tensor) -> Tensor:
        return hidden

    def reduce(self, hidden: Tensor) -> Tensor:
        return hidden


register_component_builder("residual", StandardResidualSpec, StandardResidual)
