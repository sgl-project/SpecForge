"""Residual stream boundary. Multi-stream layouts require explicit model support."""

from abc import ABC, abstractmethod

from torch import Tensor, nn


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


def build_residual(spec):
    from specforge.config.draft.component_registry import build_component

    return build_component("residual", spec, "single_stream")


def build_standard(*, spec):
    return StandardResidual()
