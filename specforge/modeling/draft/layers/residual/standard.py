"""The existing pre-norm additive residual, without extra parameters."""

import torch
from torch import nn

from .base import Residual


class StandardResidual(Residual[torch.Tensor]):
    """Norm stays registered on the decoder to preserve checkpoint names."""

    def read(
        self, state: torch.Tensor, norm: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return norm(state), state

    def write(self, update: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return state + update
