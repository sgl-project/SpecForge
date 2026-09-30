"""Module factories used by the DFlash draft backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3MLP, Qwen3RMSNorm

#: Activations Liger's SwiGLU MLP implements.
LIGER_HIDDEN_ACTS = ("silu", "swish")
#: Liger kernels are Triton programs. SpecForge validates them on CUDA and ROCm;
#: Liger also ships Ascend and XPU backends that SpecForge has not validated.
_LIGER_DEFAULT_PLATFORMS = ("cuda", "rocm")
_LIGER_UNVALIDATED_PLATFORMS = ("npu", "xpu")
_PLATFORM_NAMES = {"cuda": "CUDA", "rocm": "ROCm", "npu": "Ascend NPU", "xpu": "XPU"}
_NO_DEPS_STACKS = {
    "rocm": "ROCm PyTorch/Triton",
    "npu": "torch_npu/triton-ascend",
    "xpu": "XPU PyTorch/Triton",
}


@dataclass(frozen=True)
class DFlashKernels:
    """Stable construction boundary between DFlash and kernel providers."""

    make_rms_norm: Callable[[int, float], nn.Module]
    make_mlp: Callable[[Qwen3Config], nn.Module]


def _make_qwen3_rms_norm(hidden_size: int, eps: float) -> nn.Module:
    return Qwen3RMSNorm(hidden_size, eps=eps)


def _make_qwen3_mlp(config: Qwen3Config) -> nn.Module:
    return Qwen3MLP(config)


DEFAULT_DFLASH_KERNELS = DFlashKernels(
    make_rms_norm=_make_qwen3_rms_norm,
    make_mlp=_make_qwen3_mlp,
)


@dataclass(frozen=True)
class LigerKernelChoice:
    """Resolved ``model.use_liger_kernel`` for one trainer platform."""

    enabled: bool
    reason: str
    platform: str

    @property
    def unvalidated(self) -> bool:
        """Liger was requested on a Triton backend SpecForge never validated."""
        return self.enabled and self.platform in _LIGER_UNVALIDATED_PLATFORMS

    def describe(self) -> str:
        return f"Liger kernels: {'on' if self.enabled else 'off'} ({self.reason})"


def current_liger_platform() -> str:
    """Return this process's accelerator; PyTorch reports ROCm as ``cuda``."""
    from specforge.utils import get_device_type

    device_type = get_device_type()
    if device_type == "cuda" and getattr(torch.version, "hip", None) is not None:
        return "rocm"
    return device_type


def resolve_liger_kernel_choice(
    requested: Optional[bool],
    *,
    platform: str,
    hidden_act: str,
) -> LigerKernelChoice:
    """Resolve the tri-state flag; an explicit ``true`` never degrades silently.

    ``None`` enables Liger only where SpecForge validated it: a CUDA or ROCm
    trainer with a silu/swish draft. ``True`` raises when Liger cannot run.
    """

    if requested is False:
        return LigerKernelChoice(False, "model.use_liger_kernel=false", platform)
    name = _PLATFORM_NAMES.get(platform, platform)
    if requested is None:
        if platform not in _LIGER_DEFAULT_PLATFORMS:
            return LigerKernelChoice(
                False, f"auto: trainer device {platform!r} is not CUDA/ROCm", platform
            )
        if hidden_act not in LIGER_HIDDEN_ACTS:
            return LigerKernelChoice(
                False,
                f"auto: draft hidden_act {hidden_act!r} is not silu/swish",
                platform,
            )
        return LigerKernelChoice(True, f"auto: DFlash on {name}", platform)

    if platform not in _LIGER_DEFAULT_PLATFORMS + _LIGER_UNVALIDATED_PLATFORMS:
        raise ValueError(
            "model.use_liger_kernel=true requires a CUDA or ROCm GPU (Liger "
            f"kernels are Triton); this trainer's device is {platform!r}. Remove "
            "the key or set it to false to use the native Qwen3 RMSNorm/SwiGLU."
        )
    if hidden_act not in LIGER_HIDDEN_ACTS:
        raise ValueError(
            "model.use_liger_kernel=true requires the draft hidden_act to be "
            f"'silu' or 'swish' (Liger SwiGLU); got {hidden_act!r}. Remove the "
            "key or set it to false to use the native Qwen3 MLP."
        )
    if platform in _LIGER_UNVALIDATED_PLATFORMS:
        return LigerKernelChoice(
            True, f"explicit, unvalidated in SpecForge on {name}", platform
        )
    return LigerKernelChoice(True, "explicit", platform)


def _liger_install_hint(platform: str) -> str:
    stack = _NO_DEPS_STACKS.get(platform)
    if stack is None:
        return 'pip install "specforge[liger]", or pip install liger-kernel'
    return f"pip install --no-deps liger-kernel into the existing {stack} stack"


def load_liger_dflash_kernels(
    choice: Optional[LigerKernelChoice] = None,
) -> DFlashKernels:
    """Load Liger lazily and adapt its constructors to the DFlash boundary."""

    choice = choice or LigerKernelChoice(True, "explicit", "cuda")
    try:
        # Liger's kernels are Triton programs; name a missing Triton directly.
        import triton  # noqa: F401
        from liger_kernel.transformers import LigerRMSNorm, LigerSwiGLUMLP
    except ImportError as exc:
        raise ImportError(
            f"model.use_liger_kernel resolved to true ({choice.reason}) but Liger "
            f"could not be imported: {exc}. Install it "
            f"({_liger_install_hint(choice.platform)}) or set "
            "model.use_liger_kernel: false to use the native Qwen3 RMSNorm/SwiGLU."
        ) from exc

    def make_rms_norm(hidden_size: int, eps: float) -> nn.Module:
        return LigerRMSNorm(hidden_size, eps=eps)

    def make_mlp(config: Qwen3Config) -> nn.Module:
        return LigerSwiGLUMLP(config)

    return DFlashKernels(
        make_rms_norm=make_rms_norm,
        make_mlp=make_mlp,
    )
