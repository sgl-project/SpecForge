"""Acceptance-rate prediction from prepared draft features."""

from __future__ import annotations

import torch
from torch import nn


class AcceptRatePredictor(nn.Module):
    """Predict target/draft distribution acceptance probability per draft step."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features).squeeze(-1)
