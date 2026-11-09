"""Additive objective values shared with loss reductions and the trainer."""

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class ObjectiveResult:
    numerator: Tensor
    denominator: Tensor
    ratios: dict[str, tuple[Tensor, Tensor]] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    sums: dict = field(default_factory=dict)

    @property
    def loss(self):
        return self.numerator / self.denominator.clamp_min(
            torch.finfo(self.denominator.dtype).tiny
        )
