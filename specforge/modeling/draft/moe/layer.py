# coding=utf-8
"""``MoELayer``, the dense/MoE switch ``build_ffn`` and the model-level hooks."""

from __future__ import annotations

from typing import Callable, Dict, Iterator, Optional, Tuple

import torch
from torch import nn

from .balance import build_balance_controller
from .base import MetricValue, TrainingState
from .config import MoEConfig, requests_routed_experts, resolve_moe_config
from .experts import build_routed_experts, build_shared_expert
from .router import build_router


class MoELayer(nn.Module):
    """Routed FFN ``cast(experts(x, gate(x)) + shared_experts(x))``, summed in fp32."""

    def __init__(self, cfg: MoEConfig, hidden_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hidden_size
        # gate / experts / shared_experts are the module-side naming contract.
        balance = build_balance_controller(cfg, cfg.n_routed_experts)
        self.gate = build_router(cfg, hidden_size, balance)
        self.experts = build_routed_experts(cfg, hidden_size)
        self.shared_experts = (
            build_shared_expert(cfg, hidden_size) if cfg.n_shared_experts else None
        )
        # Detached per-expert counts of the last training forward, for metrics.
        self.last_counts: Optional[torch.Tensor] = None

    @property
    def balance(self):
        return self.gate.balance

    def components(self) -> Tuple[TrainingState, ...]:
        """The parts that carry training state, in a fixed order."""
        parts = [self.gate, self.balance, self.experts]
        if self.shared_experts is not None:
            parts.append(self.shared_experts)
        return tuple(parts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.hidden_size)
        tokens_per_sample = int(shape[-2]) if len(shape) >= 3 else None
        routing = self.gate(flat, tokens_per_sample=tokens_per_sample)
        if self.training:
            self.last_counts = routing.counts.detach()
            for component in self.components():
                component.observe(routing)
        y = self.experts(flat, routing)  # fresh fp32 accumulator
        if self.shared_experts is not None:
            y.add_(self.shared_experts(flat))
        return y.to(x.dtype).view(shape)

    def apply_pending_update(self) -> None:
        for component in self.components():
            component.apply_pending_update()

    def aux_loss(self) -> Optional[torch.Tensor]:
        """Sum of the components' already-scaled losses, or ``None``."""
        total: Optional[torch.Tensor] = None
        for component in self.components():
            loss = component.aux_loss()
            if loss is not None:
                total = loss if total is None else total + loss
        return total

    def metrics(self) -> Dict[str, MetricValue]:
        out: Dict[str, MetricValue] = {}
        counts = self.last_counts
        if counts is not None and counts.numel():
            load = counts.float()
            mean = load.mean().clamp_min(1e-9)
            out["load_max_ratio"] = load.max() / mean
            out["load_min_ratio"] = load.min() / mean
            out["experts_unused_frac"] = (load == 0).float().mean()
        for component in self.components():
            out.update(component.metrics())
        return out

    def reset_parameters(self, std: float) -> None:
        """Initialize bare Parameters the HF ``_init_weights`` pass cannot see."""
        for component in self.components():
            component.reset_parameters(std)


def build_ffn(config, dense: Callable[[object], nn.Module]) -> nn.Module:
    """The dense/MoE switch: ``dense(config)`` unless the config names a ``moe_preset``."""
    moe_cfg = resolve_moe_config(config)
    if moe_cfg is not None:
        return MoELayer(moe_cfg, int(config.hidden_size))
    if requests_routed_experts(config):
        raise ValueError(
            "the draft config sets n_routed_experts but no moe_preset; name the "
            "target family's preset (see specforge.modeling.draft.moe)"
        )
    return dense(config)


# -- model-level hooks over any module tree -------------------------------------


def iter_moe_layers(module: nn.Module) -> Iterator[MoELayer]:
    for sub in module.modules():
        if isinstance(sub, MoELayer):
            yield sub


def apply_pending_balance_updates(module: nn.Module) -> None:
    """Apply stashed routing statistics; call first in the training forward."""
    for layer in iter_moe_layers(module):
        layer.apply_pending_update()


def collect_moe_aux_loss(module: nn.Module) -> Optional[torch.Tensor]:
    """Sum of the layers' already-scaled auxiliary losses, or ``None``."""
    total: Optional[torch.Tensor] = None
    for layer in iter_moe_layers(module):
        loss = layer.aux_loss()
        if loss is not None:
            total = loss if total is None else total + loss
    return total


def collect_moe_metrics(
    module: nn.Module, prefix: str = "moe/"
) -> Dict[str, MetricValue]:
    """Layer-averaged scalar metrics under ``prefix``; ``{}`` without MoE layers."""
    sums: Dict[str, MetricValue] = {}
    n = 0
    for layer in iter_moe_layers(module):
        n += 1
        for key, value in layer.metrics().items():
            sums[key] = value if key not in sums else sums[key] + value
    return {f"{prefix}{key}": value / n for key, value in sums.items()}
