"""Residual boundaries and independently owned sublayer connections."""

from abc import ABC, abstractmethod
from collections.abc import Callable

from torch import Tensor, nn


class ResidualConnection(nn.Module, ABC):
    @abstractmethod
    def forward(self, hidden: Tensor, sublayer: Callable[[Tensor], Tensor]) -> Tensor:
        """Read residual streams, call the sublayer once, then merge its update.

        The callable includes the sublayer's normalization and runtime inputs.
        The connection owns any read/write/mixing parameters and keeps per-call
        intermediates local, so checkpoint recomputation needs no mutable state.
        """


class ResidualStreams(nn.Module, ABC):
    @abstractmethod
    def expand(self, hidden: Tensor) -> Tensor:
        """Enter the residual layout consumed by decoder layers."""

    @abstractmethod
    def reduce(self, hidden: Tensor) -> Tensor:
        """Return one hidden stream before final normalization."""

    @abstractmethod
    def build_connection(
        self, *, hidden_size: int, device=None, dtype=None
    ) -> ResidualConnection:
        """Create a fresh connection from this residual's configuration.

        The decoder registers one connection per attention/FFN sublayer; the
        boundary module neither stores nor shares them. Parameterized connections
        implement reset_draft_parameters like other draft components.
        """
