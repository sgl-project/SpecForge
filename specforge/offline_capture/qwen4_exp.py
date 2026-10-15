"""Qwen3.8-Flash-Next (``qwen4_exp``) support for the HF offline capture backend.

Three pieces, each checked against the Transformers reference it replaces
(``tests/test_offline_capture/test_hf_qwen4_exp.py``):

* ``hyper_connection_taps`` -- where a DSpark tap reads the target. The
  residual stream is ``hc_count`` streams wide, so "the output of layer i" is
  ``hc_count`` vectors. The tap folds them with the next layer's own gated read,
  ``layers[i + 1].attn_hyper_connection(streams)[0]``: the normalized input
  layer ``i + 1``'s attention actually sees. vLLM-Ascend must fold the same way.
* ``fast_indexer_forward`` -- the QSA block selection without the reference's
  per-query Python loop, which runs ``seq_len`` host-synchronizing iterations
  per sparse layer and makes a 4k-token capture impractical.
* ``sorted_experts_forward`` -- the routed experts with one host readback per
  layer instead of one ``torch.where`` per hit expert.
"""

from __future__ import annotations

import math
import types
from typing import Callable, Dict, List, Sequence

import torch
import torch.nn.functional as F

MODEL_TYPES = frozenset({"qwen4_exp", "qwen4_exp_text"})


def is_qwen4_exp(config) -> bool:
    return getattr(config, "model_type", None) in MODEL_TYPES


def text_model(model):
    """The ``Qwen4ExpTextModel`` inside a causal-LM or conditional-generation model."""

    inner = getattr(model, "model", model)
    return getattr(inner, "language_model", inner)


def hyper_connection_taps(
    model,
    layer_ids: Sequence[int],
    sink: Dict[int, torch.Tensor],
) -> List[torch.utils.hooks.RemovableHandle]:
    """Record each tap's folded state into ``sink`` during the next forward.

    ``layer_ids`` follow SpecForge's convention: id ``i`` is the output of layer
    ``i``, which vLLM numbers as aux layer ``i + 1``.
    """

    layers = text_model(model).layers
    handles = []
    for layer_id in layer_ids:
        if not 0 <= layer_id < len(layers) - 1:
            raise ValueError(
                f"tap {layer_id} has no following layer to fold its "
                f"{len(layers)}-layer hyper-connection streams with"
            )
        reader = layers[layer_id + 1]
        if getattr(reader, "ple", None) is not None:
            # PLE joins the streams before that read, so the fold would carry
            # layer i + 1's n-gram injection rather than layer i's output.
            raise ValueError(
                f"tap {layer_id} is read by PLE layer {layer_id + 1}; choose "
                "a tap whose next layer has no PLE"
            )

        def record(_module, _args, output, *, key=layer_id):
            sink[key] = output[0]

        handles.append(reader.attn_hyper_connection.register_forward_hook(record))
    return handles


def _is_plain_causal(visible: torch.Tensor) -> bool:
    """True when every row's visibility is exactly ``tril`` over a square mask."""

    batch, _, query_len, key_len = visible.shape
    if query_len != key_len:
        return False
    causal = torch.ones(
        query_len, key_len, dtype=torch.bool, device=visible.device
    ).tril_()
    return bool((visible[:, 0] == causal).all())


def fast_indexer_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,
    attention_mask: torch.Tensor,
    past_key_values=None,
):
    """Vectorized ``Qwen4ExpTextQSAIndexer.forward`` for a cache-free prefill.

    The reference walks every query: it splits the visible tokens into
    ``compress_ratio``-token blocks, scores each complete block by
    ``sum_h relu(q_h . rope(k_norm(mean_block(k))))`` and keeps the top
    ``budget // compress_ratio`` blocks plus the incomplete tail. Without
    padding or a cache, query ``q`` sees tokens ``0..q``, so its blocks are the
    fixed blocks ``[4b, 4b + 4)`` with ``b < (q + 1) // 4``: every query shares
    one set of pooled keys and the per-query loop becomes one score matrix.
    Any other mask falls back to the reference.
    """

    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    visible = (
        attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    )
    if past_key_values is not None or not _is_plain_causal(visible):
        return self._specforge_reference_forward(
            hidden_states, position_embeddings, attention_mask, past_key_values
        )

    batch, seq_len, _ = hidden_states.shape
    ratio = self.compress_ratio
    cos, sin = position_embeddings
    cos, sin = cos[:, -seq_len:, :], sin[:, -seq_len:, :]

    q, raw_keys = torch.split(
        self.index_qk_proj(hidden_states),
        [
            self.index_n_heads * self.index_head_dim,
            self.index_kv_heads * self.index_head_dim,
        ],
        dim=-1,
    )
    hidden_shape = (batch, seq_len, -1, self.index_head_dim)
    q = self.q_layernorm(q.reshape(hidden_shape))
    q = apply_rotary_pos_emb(q, cos=cos, sin=sin, unsqueeze_dim=2)
    raw_keys = raw_keys.reshape(hidden_shape).squeeze(2)

    num_blocks = seq_len // ratio
    selected = torch.zeros(
        batch, seq_len, seq_len, dtype=torch.bool, device=hidden_states.device
    )
    positions = torch.arange(seq_len, device=hidden_states.device)
    # Complete blocks visible to each query, and its incomplete tail.
    complete = torch.div(positions + 1, ratio, rounding_mode="floor")
    tail_start = complete * ratio
    tail = (positions[None, :] >= tail_start[:, None]) & (
        positions[None, :] <= positions[:, None]
    )
    selected |= tail.unsqueeze(0)

    if num_blocks > 0:
        groups = raw_keys[:, : num_blocks * ratio].reshape(
            batch, num_blocks, ratio, self.index_head_dim
        )
        pooled = groups.float().mean(dim=2).to(raw_keys.dtype)
        pooled = self.k_layernorm(pooled)
        starts = positions[: num_blocks * ratio : ratio]
        block_keys = apply_rotary_pos_emb(
            pooled.unsqueeze(2),
            cos=cos.index_select(1, starts),
            sin=sin.index_select(1, starts),
            unsqueeze_dim=2,
        ).squeeze(2)
        # [B, S, H, D] x [B, D, N] -> [B, S, H, N], the reference's per-query
        # [H, D] x [D, N] product computed for every query at once.
        scores = torch.matmul(
            q.float(), block_keys.float().transpose(-1, -2).unsqueeze(1)
        )
        scores = torch.relu(scores).sum(dim=2) / math.sqrt(self.index_head_dim)
        blocks = torch.arange(num_blocks, device=hidden_states.device)
        valid = blocks[None, :] < complete[:, None]
        scores = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
        keep = min(self.block_topk, num_blocks)
        chosen = torch.zeros(
            batch, seq_len, num_blocks, dtype=torch.bool, device=scores.device
        )
        chosen.scatter_(-1, scores.topk(keep, dim=-1).indices, True)
        chosen &= valid.unsqueeze(0)
        selected[..., : num_blocks * ratio] |= chosen.repeat_interleave(ratio, dim=-1)

    selected = selected.unsqueeze(1)
    if attention_mask.is_floating_point():
        min_dtype = torch.finfo(attention_mask.dtype).min
        return torch.where(selected, attention_mask.new_zeros(()), min_dtype)
    return selected


def sorted_experts_forward(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """``Qwen4ExpTextExperts.forward`` with one host readback per layer.

    The reference finds each hit expert's tokens with its own ``torch.where``,
    a device-to-host sync per expert -- up to 512 per layer. Sorting the
    assignments once and reading back only the group boundaries gives every
    expert the same rows. The accumulation order is unchanged: experts in
    ascending order, and an expert never sees a token twice.
    """

    top_k = top_k_index.shape[-1]
    flat = top_k_index.reshape(-1)
    # Ascend has no AiCore ArgSort for integer keys; expert ids are exact in
    # float32, so sorting the float view gives the same permutation on AiCore.
    order = torch.argsort(flat.float(), stable=True)
    bounds = [0] + torch.cumsum(
        torch.bincount(flat, minlength=self.num_experts), dim=0
    ).tolist()
    tokens = torch.div(order, top_k, rounding_mode="floor")
    slots = order % top_k
    output = torch.zeros_like(hidden_states)
    for expert in range(self.num_experts):
        start, end = bounds[expert], bounds[expert + 1]
        if start == end:
            continue
        token = tokens[start:end]
        gate, up = F.linear(hidden_states[token], self.gate_up_proj[expert]).chunk(
            2, dim=-1
        )
        expert_out = F.linear(self.act_fn(gate) * up, self.down_proj[expert])
        expert_out = expert_out * top_k_weights[token, slots[start:end], None]
        output.index_add_(0, token, expert_out.to(output.dtype))
    return output


def _replace_forward(module, forward: Callable) -> None:
    """Swap a module's forward, below an Accelerate device hook if there is one."""

    bound = types.MethodType(forward, module)
    if hasattr(module, "_old_forward"):
        module._specforge_reference_forward = module._old_forward
        module._old_forward = bound
    else:
        module._specforge_reference_forward = module.forward
        module.forward = bound


def install_fast_paths(model) -> Dict[str, int]:
    """Replace the indexer and expert forwards; returns how many of each."""

    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextExperts,
        Qwen4ExpTextQSAIndexer,
    )

    counts = {"indexer": 0, "experts": 0}
    for module in model.modules():
        if isinstance(module, Qwen4ExpTextQSAIndexer):
            _replace_forward(module, fast_indexer_forward)
            counts["indexer"] += 1
        elif isinstance(module, Qwen4ExpTextExperts):
            _replace_forward(module, sorted_experts_forward)
            counts["experts"] += 1
    return counts


__all__ = [
    "MODEL_TYPES",
    "fast_indexer_forward",
    "hyper_connection_taps",
    "install_fast_paths",
    "is_qwen4_exp",
    "sorted_experts_forward",
    "text_model",
]
