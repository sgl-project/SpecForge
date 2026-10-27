"""Residual stream state is explicit and independent of the attention cache."""

from abc import ABC, abstractmethod
from typing import Generic, NamedTuple, TypeVar

import torch
from torch import nn


class HyperConnectionState(NamedTuple):
    streams: torch.Tensor  # [batch, sequence, streams, hidden]
    pre_mix: torch.Tensor | None = None  # single-pass mHC's next read


ResidualStream = torch.Tensor | HyperConnectionState
WriteState = TypeVar("WriteState")


class Residual(nn.Module, ABC, Generic[WriteState]):
    @abstractmethod
    def read(
        self, state: ResidualStream, norm: nn.Module
    ) -> tuple[torch.Tensor, WriteState]:
        """Read a sublayer input and return the state needed for its write."""
        raise NotImplementedError

    @abstractmethod
    def write(self, update: torch.Tensor, state: WriteState) -> ResidualStream:
        raise NotImplementedError


def expand_streams(
    hidden: torch.Tensor, count: int, *, single_pass: bool
) -> HyperConnectionState:
    streams = hidden.unsqueeze(-2).expand(*hidden.shape[:-1], count, hidden.shape[-1])
    pre_mix = None
    if single_pass:
        pre_mix = hidden.new_zeros(*hidden.shape[:-1], count, dtype=torch.float32)
        pre_mix[..., 0] = 1
    return HyperConnectionState(streams, pre_mix)


def collapse_streams(state: HyperConnectionState) -> torch.Tensor:
    """Read the last single-pass mHC mix, including at the model head."""
    if state.pre_mix is None:
        raise ValueError("single-pass mHC requires the incoming pre_mix")
    return (
        (state.streams.float() * state.pre_mix.unsqueeze(-1))
        .sum(-2)
        .to(state.streams.dtype)
    )
