"""Residual stream boundary. Multi-stream layouts require explicit model support."""

from abc import ABC, abstractmethod

from torch import Tensor, nn
from transformers import PretrainedConfig

from specforge.config.draft.components import component_config


class ResidualStream(nn.Module, ABC):
    """Transform the stream into a branch input and combine its output.

    A future GR/mHC implementation may use a different stream layout. Callers
    must implement the associated input/output transitions before enabling it.
    """

    @abstractmethod
    def branch_input(self, stream: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def combine(self, stream: Tensor, branch_output: Tensor) -> Tensor:
        raise NotImplementedError


class StandardResidual(ResidualStream):
    def branch_input(self, stream: Tensor) -> Tensor:
        return stream

    def combine(self, stream: Tensor, branch_output: Tensor) -> Tensor:
        return stream + branch_output


def build_residual(config: PretrainedConfig) -> ResidualStream:
    choice = component_config(config, "residual_config", "standard")
    if choice.type != "standard" or choice.variant is not None:
        raise ValueError(f"Residual stream {choice!r} is not implemented")
    return StandardResidual()
