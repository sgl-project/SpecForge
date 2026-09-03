# coding=utf-8
"""Shared expert contract: an always-on FFN added to the routed output.

Families differ in the gate on it (DeepSeek: none; Qwen: a per-token sigmoid
gate), which is ``MoEConfig.shared_expert_gate``, and in its width
(``shared_expert_intermediate_size``). Implementations register by name
(``MoEConfig.shared_expert``). Checkpoint naming follows the official
``shared_experts.w{1,2,3}.weight`` layout unless a converter says otherwise.
"""

from __future__ import annotations

import abc
from typing import Type

import torch
from torch import nn

from ._registry import Registry
from .config import MoEConfig


class SharedExpert(nn.Module, abc.ABC):
    def __init__(self, cfg: MoEConfig, hidden_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hidden_size
        self.intermediate_size = cfg.shared_expert_intermediate_size

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is ``[T, hidden]``; return ``[T, hidden]`` in ``x.dtype``."""

    def reset_parameters(self, std: float) -> None:
        """Hook for bare Parameters; ``nn.Linear`` children are covered by HF init."""


SHARED_EXPERTS: Registry[Type[SharedExpert]] = Registry("MoE shared expert")


def register_shared_expert(name: str):
    return SHARED_EXPERTS.register(name)


def build_shared_expert(cfg: MoEConfig, hidden_size: int) -> SharedExpert:
    return SHARED_EXPERTS.get(cfg.shared_expert)(cfg, hidden_size)
