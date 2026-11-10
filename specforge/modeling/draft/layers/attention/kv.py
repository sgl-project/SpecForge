"""Projected training K/V with an attached autograd graph, not a mutable cache."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, eq=False)
class AttentionKV:
    """Keys are normalized/rotated; both tensors use [batch, heads, length, dim]."""

    key: Tensor
    value: Tensor

    def __post_init__(self):
        if (
            self.key.ndim != 4
            or self.value.ndim != 4
            or self.key.shape[:3] != self.value.shape[:3]
            or self.key.device != self.value.device
            or not self.key.is_floating_point()
            or not self.value.is_floating_point()
        ):
            raise ValueError("K/V must have matching batch, heads, length and device")

    def prepend(self, prefix: "AttentionKV | None") -> "AttentionKV":
        if prefix is None:
            return self
        for old, new in ((prefix.key, self.key), (prefix.value, self.value)):
            if (
                old.shape[:2] != new.shape[:2]
                or old.shape[-1] != new.shape[-1]
                or old.device != new.device
                or old.dtype != new.dtype
            ):
                raise ValueError(
                    "K/V prefix must match projected dimensions, device and dtype"
                )
        return AttentionKV(
            torch.cat((prefix.key, self.key), dim=2),
            torch.cat((prefix.value, self.value), dim=2),
        )
