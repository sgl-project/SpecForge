"""Construction options shared by draft components, independent of algorithms."""

from dataclasses import dataclass
from typing import Callable, Literal

from torch import nn
from transformers import PretrainedConfig


@dataclass(frozen=True)
class LayerKernels:
    """Inject norm and FFN implementations without changing parameter paths."""

    make_rms_norm: Callable[[int, float], nn.Module]
    make_mlp: Callable[[PretrainedConfig], nn.Module]


@dataclass(frozen=True)
class AttentionConfig:
    """Resolved structure and visibility defaults for one attention layer.

    Causality and the window intersect with the explicit allow-mask. Algorithms
    supply token positions when sequence storage differs from logical order.
    """

    # QSA/DSA reserve structure names; their execution is not implemented yet.
    # SWA is a visibility rule, orthogonal to the projection/selection structure.
    kind: Literal["gqa", "mha", "mla", "qsa", "dsa"] = "gqa"
    is_causal: bool = False
    sliding_window: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("gqa", "mha", "mla", "qsa", "dsa"):
            raise ValueError(f"Unsupported attention kind: {self.kind!r}")
        if not isinstance(self.is_causal, bool):
            raise ValueError("is_causal must be a boolean")
        if self.sliding_window is not None and (
            isinstance(self.sliding_window, bool)
            or not isinstance(self.sliding_window, int)
            or self.sliding_window <= 0
        ):
            raise ValueError("sliding_window must be a positive integer or None")


@dataclass(frozen=True)
class ResidualConfig:
    """Opt-in residual architecture; GR matches Qwen's per-branch normalization."""

    kind: Literal["standard", "gr", "mhc"] = "standard"
    num_streams: int = 4
    gate_rank: int = 16
    sinkhorn_iters: int = 20
    sinkhorn_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.kind not in ("standard", "gr", "mhc"):
            raise ValueError(f"Unsupported residual kind: {self.kind!r}")
        for name in ("num_streams", "gate_rank", "sinkhorn_iters"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.sinkhorn_eps < 1:
            raise ValueError("sinkhorn_eps must be between 0 and 1")
