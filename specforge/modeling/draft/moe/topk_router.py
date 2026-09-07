# coding=utf-8
"""Top-k router with pluggable score functions and optional group-limited routing."""

from __future__ import annotations

import math
import re
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .balance import BalanceController, MetricValue
from .config import MoEConfig
from .router import (
    Router,
    RoutingResult,
    get_score_function,
    register_router,
    register_score_function,
)


@register_score_function("softmax")
def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return logits.softmax(dim=-1)


@register_score_function("sigmoid")
def _sigmoid(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


@register_score_function("sqrtsoftplus")
def _sqrtsoftplus(logits: torch.Tensor) -> torch.Tensor:
    """DeepSeek-V4 scoring: ``sqrt(softplus(logits))``."""
    return F.softplus(logits).sqrt()


def group_limited_mask(
    selection: torch.Tensor, n_group: int, topk_group: int
) -> torch.Tensor:
    """Keep only the ``topk_group`` groups with the highest top-2 score sums
    (DeepSeek ``noaux_tc`` group scoring); other groups become ``-inf``."""
    tokens, n_experts = selection.shape
    grouped = selection.view(tokens, n_group, n_experts // n_group)
    group_scores = grouped.topk(min(2, grouped.shape[-1]), dim=-1).values.sum(-1)
    keep = group_scores.topk(topk_group, dim=-1).indices
    mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, keep, True)
    return grouped.masked_fill(~mask.unsqueeze(-1), float("-inf")).view(
        tokens, n_experts
    )


@torch.no_grad()
def routing_diagnostics(
    x: torch.Tensor, logits: torch.Tensor, indices: torch.Tensor, counts: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Cheap, sync-free diagnostics of how token-dependent routing is.

    - ``router_input_cos``: mean pairwise cosine of the router inputs (1.0 when
      every token feeds the gate the same direction).
    - ``logit_common_frac``: fraction of the logit energy (after removing each
      token's mean over experts, which softmax/top-k ignore) explained by the
      token-mean logit vector; 1.0 when all tokens share identical logits.
    - ``logit_common_std`` / ``logit_token_std``: spread across experts of that
      common vector vs the RMS token-specific deviation from it, in logit units
      (the scale a jitter must reach to change a token's top-k).
    - ``route_entropy_frac``: entropy of the expert histogram over ``log(E)``
      (1.0 = uniform; ``log(k)/log(E)`` when every token picks the same ``k``).
    - ``top1_mode_frac``: fraction of tokens whose first choice is the modal one.
    """
    tokens, n_experts = logits.shape
    xn = F.normalize(x.float(), dim=-1)
    m_sq = xn.mean(dim=0).square().sum()
    if tokens > 1:
        input_cos = (tokens * m_sq - 1.0) / (tokens - 1)
    else:
        input_cos = torch.ones_like(m_sq)
    centered = logits - logits.mean(dim=-1, keepdim=True)
    common = centered.mean(dim=0)
    total = centered.square().sum().clamp_min(1e-20)
    common_frac = tokens * common.square().sum() / total
    token_std = (centered - common).square().mean().sqrt()
    common_std = common.std() if n_experts > 1 else torch.zeros_like(m_sq)
    hist = counts.float() / counts.sum().clamp_min(1)
    entropy = -(hist * torch.log(hist.clamp_min(1e-20))).sum()
    entropy_frac = entropy / math.log(max(n_experts, 2))
    top1 = indices[:, 0]
    top1_counts = torch.zeros(
        n_experts, dtype=torch.long, device=indices.device
    ).scatter_add_(0, top1, torch.ones_like(top1))
    top1_mode = top1_counts.max().float() / max(tokens, 1)
    return {
        "router_input_cos": input_cos,
        "logit_common_frac": common_frac,
        "logit_common_std": common_std,
        "logit_token_std": token_std,
        "route_entropy_frac": entropy_frac,
        "top1_mode_frac": top1_mode,
    }


@register_router("topk")
class TopKRouter(Router):
    """``scores = f(x W^T)``; pick top-k on balance-adjusted scores; combine
    with the raw scores (optionally renormalized, then scaled).

    Training-only regularizers (``MoEConfig.router_*``): Gaussian logit jitter,
    the ST-MoE z-loss, and input centering: ``"ema"`` subtracts the
    ``gate.input_mean`` EMA buffer (updated from :meth:`apply_pending_update`
    so a checkpoint recompute routes identically); ``"batch"`` subtracts the
    exact micro-batch mean in training and the EMA in eval. Diagnostics from the last training forward are in
    :meth:`metrics` (see :func:`routing_diagnostics`)."""

    def __init__(
        self, cfg: MoEConfig, hidden_size: int, balance: BalanceController
    ) -> None:
        super().__init__(cfg, hidden_size, balance)
        self.weight = nn.Parameter(torch.empty(self.n_experts, hidden_size))
        self.score_fn = get_score_function(cfg.scoring_func)
        self.noise_std = float(cfg.router_noise_std)
        self.z_loss_coeff = float(cfg.router_z_loss_coeff)
        self.init_std = float(cfg.router_init_std)
        self.center = cfg.router_center
        self.center_momentum = float(cfg.router_center_momentum)
        if self.center in ("ema", "batch"):
            self.register_buffer(
                "input_mean", torch.zeros(hidden_size, dtype=torch.float32)
            )
            self.register_buffer("input_mean_steps", torch.zeros((), dtype=torch.long))
        else:
            self.input_mean = None
            self.input_mean_steps = None
        self._pending_input_mean: Optional[torch.Tensor] = None
        self._lag_logit_std: Optional[torch.Tensor] = None
        self._z_loss: Optional[torch.Tensor] = None
        self._diagnostics: Dict[str, torch.Tensor] = {}

    def _apply(self, fn, recurse=True):
        module = super()._apply(fn, recurse)
        # Keep the centering statistics fp32 through module-wide dtype casts.
        if module.input_mean is not None and module.input_mean.dtype != torch.float32:
            module.input_mean.data = module.input_mean.data.float()
        return module

    def reset_parameters(self, std: float) -> None:
        nn.init.normal_(
            self.weight, mean=0.0, std=self.init_std if self.init_std > 0 else std
        )

    def forward(self, x: torch.Tensor) -> RoutingResult:
        # Routing math in fp32 regardless of the model dtype.
        x32 = x.float()
        # Overwrite, never accumulate: stale training state must not survive
        # into the next forward (or an eval forward).
        self._z_loss = None
        self._pending_input_mean = None
        self._lag_logit_std = None
        if self.input_mean is not None:
            if self.training:
                batch_mean = x32.mean(dim=0)
                # Stash only; the EMA moves in apply_pending_update.
                self._pending_input_mean = batch_mean.detach()
                with torch.no_grad():
                    # Common-mode logit error eval routing would see from the
                    # EMA lag, on the scale of logit_token_std.
                    lag = F.linear(batch_mean.detach() - self.input_mean, self.weight.float())
                    self._lag_logit_std = lag.std()
                if self.center == "batch":
                    x32 = x32 - batch_mean
                else:
                    x32 = x32 - self.input_mean
            else:
                x32 = x32 - self.input_mean
        logits = F.linear(x32, self.weight.float())
        if self.training:
            if self.z_loss_coeff > 0 and logits.requires_grad:
                self._z_loss = self.z_loss_coeff * torch.logsumexp(
                    logits, dim=-1
                ).square().mean()
            if self.noise_std > 0:
                logits = logits + self.noise_std * torch.randn_like(logits)
        scores = self.score_fn(logits)
        selection = self.balance.adjust_selection_scores(scores)
        if self.cfg.group_limited:
            selection = group_limited_mask(
                selection, self.cfg.n_group, self.cfg.topk_group
            )
        indices = selection.topk(self.topk, dim=-1).indices
        weights = scores.gather(1, indices)
        if self.cfg.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.cfg.routed_scaling_factor
        flat = indices.flatten()
        # scatter_add instead of bincount: CUDA bincount hides a device sync.
        counts = torch.zeros(
            self.n_experts, dtype=torch.long, device=x.device
        ).scatter_add_(0, flat, torch.ones_like(flat))
        if self.training:
            self._diagnostics = routing_diagnostics(x, logits.detach(), indices, counts)
        return RoutingResult(
            weights=weights, indices=indices, counts=counts, scores=scores
        )

    def apply_pending_update(self) -> None:
        import torch.distributed as dist

        batch_mean = self._pending_input_mean
        self._pending_input_mean = None
        if batch_mean is None or self.input_mean is None:
            return
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_mean)
            batch_mean = batch_mean / dist.get_world_size()
        with torch.no_grad():
            if int(self.input_mean_steps) == 0:
                self.input_mean.copy_(batch_mean)
            else:
                self.input_mean.mul_(self.center_momentum).add_(
                    batch_mean, alpha=1.0 - self.center_momentum
                )
            self.input_mean_steps += 1

    def aux_loss(self) -> Optional[torch.Tensor]:
        return self._z_loss

    def metrics(self) -> Dict[str, MetricValue]:
        out: Dict[str, MetricValue] = dict(self._diagnostics)
        if self._z_loss is not None:
            out["z_loss"] = self._z_loss.detach()
        if self.input_mean is not None:
            out["input_mean_norm"] = self.input_mean.norm()
        if self._lag_logit_std is not None:
            out["logit_lag_std"] = self._lag_logit_std
        return out


_INPUT_MEAN_KEY = re.compile(r"^(?P<base>(?:.*\.)?)gate\.input_mean$")


def fold_router_centering(state: dict) -> dict:
    """Export-time fold of EMA router centering into a gate bias.

    ``W (x - mu) = W x - W mu``: the checkpoint's ``gate.input_mean`` (and its
    step counter) become ``gate.bias = -W @ mu`` so a serving engine that
    reads a router bias reproduces the trained routing without knowing about
    centering. No-op on dicts without the buffer.
    """
    out = dict(state)
    for key in list(state):
        m = _INPUT_MEAN_KEY.match(key)
        if m is None:
            continue
        base = m["base"]
        weight = state[f"{base}gate.weight"]
        mean = out.pop(key)
        out.pop(f"{base}gate.input_mean_steps", None)
        bias = -(weight.float() @ mean.float())
        out[f"{base}gate.bias"] = bias.to(weight.dtype)
    return out
