"""The ordinary single-stream residual layout needs no boundary transforms."""

from torch import Tensor

from specforge.config.draft.residual import StandardResidualSpec

from ..registry import register_component_builder
from .base import ResidualStreams


class StandardResidual(ResidualStreams):
    def __init__(self, config: StandardResidualSpec):
        super().__init__()
        if type(config) is not StandardResidualSpec:
            raise TypeError("StandardResidual requires StandardResidualSpec")
        self.config = config

    def expand(self, hidden: Tensor) -> Tensor:
        return hidden

    def reduce(self, hidden: Tensor) -> Tensor:
        return hidden


register_component_builder("residual", StandardResidualSpec, StandardResidual)
