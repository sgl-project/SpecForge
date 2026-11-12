"""Convolution inside a decoder layer, independent of an algorithm name."""

from dataclasses import dataclass
from typing import ClassVar

from .._validation import positive_int
from .base import ConvSpec


@dataclass(frozen=True, kw_only=True)
class GroupedCausalConvSpec(ConvSpec):
    type: ClassVar[str] = "grouped_causal"

    def capabilities(self) -> frozenset[str]:
        return frozenset({"block_sublayer_pair"})

    kernel_size: int
    group_size: int

    def __post_init__(self):
        positive_int("kernel_size", self.kernel_size)
        positive_int("group_size", self.group_size)
