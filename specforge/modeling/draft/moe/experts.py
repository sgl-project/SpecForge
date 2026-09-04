# coding=utf-8
"""Routed experts contract: the expert weights and how tokens reach them.

An implementation owns the parameters of all ``n_routed_experts`` experts and
turns a :class:`RoutingResult` into the combined routed output. Weight layout
is the implementation's choice (per-expert modules, stacked ``[E, out, in]``
tensors, ...); it must register a :mod:`.state_dict` converter if its native
layout differs from the official checkpoint naming
(``experts.{i}.w{1,2,3}.weight``). ``MoEConfig.dispatch`` is the
implementation's execution knob (e.g. sorted-segment loop vs grouped GEMM).
"""

from __future__ import annotations

import abc
from typing import Type

import torch
from torch import nn

from ._registry import Registry
from .config import MoEConfig
from .router import RoutingResult


class RoutedExperts(nn.Module, abc.ABC):
    #: The training backend keeps a fully frozen instance replicated (outside
    #: FSDP sharding): no weight all-gathers or gradient reduce-scatters.
    fsdp_replicate_when_frozen = True

    def __init__(self, cfg: MoEConfig, hidden_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hidden_size
        self.n_experts = cfg.n_routed_experts
        self.intermediate_size = cfg.moe_intermediate_size

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, routing: RoutingResult) -> torch.Tensor:
        """``x`` is ``[T, hidden]``; return the combined routed output ``[T, hidden]``
        in ``x.dtype``."""

    @abc.abstractmethod
    def reset_parameters(self, std: float) -> None:
        """Initialize expert weights with the draft's ``initializer_range`` so an
        MoE FFN starts from the same distribution as the dense MLP it replaces."""


EXPERTS_BACKENDS: Registry[Type[RoutedExperts]] = Registry("MoE experts backend")


def register_experts_backend(name: str):
    return EXPERTS_BACKENDS.register(name)


def build_routed_experts(cfg: MoEConfig, hidden_size: int) -> RoutedExperts:
    return EXPERTS_BACKENDS.get(cfg.experts_backend)(cfg, hidden_size)
