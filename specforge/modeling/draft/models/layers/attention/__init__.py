"""Attention construction boundaries.

DFlash's context-then-draft layout and EAGLE's feature/cache contract are
separate interfaces. New sparse or stateful structures must declare which
contract they implement instead of being coerced into a dense Q/K/V API.
"""

from torch import nn
from transformers import Qwen3Config

from specforge.modeling.draft.config.dflash import resolve_dflash_attention_mode

from ..kernels import DFlashKernels


def build_dflash_attention(
    config: Qwen3Config, layer_idx: int, kernels: DFlashKernels
) -> nn.Module:
    """Construct a full attention module for DFlash's context/query contract.

    DSA/QSA must implement selection and cache semantics in their own module,
    not inherit the dense projection-only scaffold. Stateful structures such
    as KDA also need an explicit caller/cache contract before registration.
    """
    from .gqa import Qwen3DFlashAttention
    from .mla import Qwen3DFlashMLAAttention

    mode = resolve_dflash_attention_mode(config)
    constructors = {
        "gqa": Qwen3DFlashAttention,
        "mha": Qwen3DFlashAttention,
        "mla": Qwen3DFlashMLAAttention,
    }
    try:
        constructor = constructors[mode]
    except KeyError:
        raise ValueError(f"DFlash attention {mode!r} is not implemented") from None
    return constructor(config=config, layer_idx=layer_idx, kernels=kernels)
