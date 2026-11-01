"""Execution adapters for explicit Q/K/V. No projection or indexer weights."""

from typing import TYPE_CHECKING, Callable, Optional

from torch.nn.attention.flex_attention import BlockMask

if TYPE_CHECKING:
    from ..base import DenseAttention

import torch
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    eager_attention_forward as eager_attention_with_sinks,
)
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
)

from .flex_attention import compile_friendly_flex_attention
from .flex_attention_backend import flex_attention_backend
from .visibility import prepare_tensor_mask, resolve_visibility


def dense_attention(
    module: "DenseAttention",
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | BlockMask | None,
    *,
    query_positions: torch.Tensor | None = None,
    key_positions: torch.Tensor | None = None,
    mask_encodes_visibility: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    implementation = module.config._attn_implementation
    # A per-head sink logit, HF's ``sinks`` (gpt-oss, DeepSeek-V4/V4.1); only backends
    # that expose the softmax normalizer can apply it: fused flex on GPU, eager on CPU.
    sink = getattr(module, "sinks", None)
    if sink is not None and implementation not in ("eager", "flex_attention"):
        raise ValueError(
            "Attention sinks need flex_attention (fast, GPU) or eager (CPU reference)"
        )
    causal = kwargs.pop("is_causal", module.is_causal)
    if not isinstance(causal, bool):
        raise ValueError("is_causal must be a boolean")
    valid_queries = None
    native_sdpa_causal = (
        implementation == "sdpa"
        and causal
        and module.sliding_window is None
        and attention_mask is None
        and query_positions is None
        and key_positions is None
        and (q.shape[-2] == k.shape[-2] or q.shape[-2] == 1)
    )
    resolved_by_algorithm = mask_encodes_visibility and attention_mask is not None
    if implementation in ("eager", "sdpa", "flex_attention"):
        if not native_sdpa_causal and not resolved_by_algorithm:
            attention_mask = resolve_visibility(
                attention_mask,
                q,
                k,
                causal=causal,
                window=module.sliding_window,
                query_positions=query_positions,
                key_positions=key_positions,
                flex=implementation == "flex_attention",
            )
    elif implementation.startswith("flash_attention"):
        # Flash's native right-aligned causal/window support is suitable for
        # contiguous sequences. Arbitrary logical layouts need Eager/SDPA/Flex.
        if query_positions is not None or key_positions is not None:
            raise ValueError(
                "Explicit visibility positions require Eager, SDPA or FlexAttention"
            )
        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError("FlashAttention only supports a 2D padding mask")
            valid_queries = attention_mask[:, None, -q.shape[-2] :, None].bool()
            valid_queries = valid_queries & attention_mask.any(-1)[:, None, None, None]
    else:
        raise ValueError(f"Unsupported dense attention backend: {implementation}")
    if implementation == "flex_attention":
        kernel_options = dict(kwargs.pop("kernel_options", None) or {})
        backend = flex_attention_backend()
        if backend is not None:
            kernel_options["BACKEND"] = backend

        # Flex already returns zeros for queries (and heads) without visible
        # keys; validity is only needed to keep an o_proj bias from reviving
        # them, so the default bias-free path skips the extra LSE output.
        need_validity = (
            attention_mask is not None
            and getattr(module.o_proj, "bias", None) is not None
        )
        if sink is not None and q.device.type != "cuda":
            # Inductor cannot emit the logsumexp for CPU flex; eager carries sinks there.
            raise ValueError(
                "Attention sinks under flex_attention need CUDA; use eager"
            )
        return_lse = need_validity or sink is not None
        result = compile_friendly_flex_attention(
            q,
            k,
            v,
            block_mask=attention_mask,
            enable_gqa=True,
            scale=module.scaling,
            kernel_options=kernel_options or None,
            return_lse=return_lse,
        )
        if return_lse:
            attn_output, lse = result
            if need_validity:
                valid_queries = (~torch.isneginf(lse)).unsqueeze(-1)
            if sink is not None:
                # softmax over [keys, sink] = softmax over keys x sigmoid(lse - sink);
                # flex's lse is the natural log, and -inf (no keys) keeps zeros.
                gate = torch.sigmoid(lse - sink.float().view(1, -1, 1))
                attn_output = attn_output * gate.unsqueeze(-1).to(attn_output.dtype)
        else:
            attn_output = result
        attn_output = attn_output.transpose(1, 2)
        attn_weights = None
    else:
        attn_fn: Callable = eager_attention_forward
        if sink is not None:
            attn_fn = eager_attention_with_sinks
        if implementation in ("eager", "sdpa"):
            attention_mask, valid_queries = prepare_tensor_mask(
                attention_mask, q.dtype, eager=implementation == "eager"
            )
        if implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[implementation]
        attn_output, attn_weights = attn_fn(
            module,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not module.training else module.attention_dropout,
            scaling=module.scaling,
            sliding_window=module.sliding_window,
            is_causal=causal,
            **kwargs,
        )
        if valid_queries is not None and attn_weights is not None:
            attn_weights = attn_weights.masked_fill(~valid_queries, 0)
    if valid_queries is not None:
        # Zero each head before o_proj; otherwise an empty head leaks forbidden
        # values even when another head has visible keys for the same query.
        attn_output = attn_output.masked_fill(~valid_queries.transpose(1, 2), 0)
    return attn_output, attn_weights, valid_queries
