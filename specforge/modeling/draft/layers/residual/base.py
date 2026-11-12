"""Residual boundaries and independently owned sublayer connections."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Literal

from torch import Tensor, nn


class ResidualConnection(nn.Module, ABC):
    @abstractmethod
    def forward(self, hidden: Tensor, sublayer: Callable[[Tensor], Tensor]) -> Tensor:
        """Read residual streams, call the sublayer once, then merge its update.

        The callable includes the sublayer's normalization and runtime inputs,
        and returns an update with the same shape as the stream it receives.
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
        self,
        *,
        hidden_size: int,
        layer_index: int,
        sublayer: Literal["attention", "mlp"],
        device=None,
        dtype=None,
    ) -> ResidualConnection:
        """Create a fresh connection from this residual's configuration.

        Assembly creates one connection per attention/FFN sublayer and passes it
        to the decoder, which registers it. Each connection keeps this boundary's
        config node. layer_index is zero-based; sublayer identifies its role.
        The boundary neither stores nor shares connections. Parameterized connections
        implement reset_draft_parameters like other draft components.
        """
