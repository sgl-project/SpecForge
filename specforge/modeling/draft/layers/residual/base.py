"""Model-boundary expansion and reduction of residual streams."""

from abc import ABC, abstractmethod

from torch import Tensor, nn


class ResidualStreams(nn.Module, ABC):
    @abstractmethod
    def expand(self, hidden: Tensor) -> Tensor:
        """Enter the residual layout consumed by decoder layers."""

    @abstractmethod
    def reduce(self, hidden: Tensor) -> Tensor:
        """Return one hidden stream before final normalization."""
