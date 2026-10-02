"""H-Spec hybrid Mamba-attention draft architecture.

This SpecForge integration keeps the DSpark block-parallel training and
Markov objective while replacing the target-context attention path.  Target
KVs are supplied directly rather than materialized from hidden states.  The
Mamba module uses a reference recurrence so its correctness can be tested
before introducing a fused parallel-scan kernel; ``hspec_config``
``mamba_backend`` stays ``'reference'`` until kernel parity is available.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
)

from .dflash import apply_rotary_pos_emb
from .dflash_kernels import DEFAULT_DFLASH_KERNELS, DFlashKernels
from .dspark import DSparkDraftModel
from .flex_attention_backend import flex_attention_backend
from .registry import register_draft

_VALID_HSPEC_ATTENTION = {"eager", "sdpa", "flex_attention"}


class RMSNormGated(nn.Module):
    """Gated RMSNorm used by the Mamba mixer (fp32 statistics).

    Mirrors the serving-side ``_RMSNormGated``: the gate is applied before the
    norm, statistics are computed in float32, and the weight multiplies the
    normalized output.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(
        self, x: torch.Tensor, gate: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        if gate is not None:
            x = x * F.silu(gate.float())
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(input_dtype)


class HSpecMamba2Reference(nn.Module):
    """Small Mamba-2 reference recurrence for correctness-first integration.

    The module intentionally accepts an initial state and follows the same
    head/state decomposition as the paper: one scalar transition per Mamba head
    is broadcast across head dimensions, while B/C are grouped.  It is not the
    production CUDA/Triton kernel and should not be benchmarked as one.

    Architecture matches the serving-side ``HSpecMambaMixer`` so checkpoints
    stay interchangeable: biased causal convolution (evaluated with Conv1d's
    own zero padding, no extra manual shift), gated RMSNorm after the scan,
    and a softplus dt without a clamp floor.
    """

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        method = config.hspec_config
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(method["mamba_num_heads"])
        self.head_dim = int(method["mamba_head_dim"])
        self.state_size = int(method["mamba_state_size"])
        self.n_groups = int(method["mamba_groups"])
        self.conv_kernel_size = int(method["mamba_conv_kernel"])
        self.intermediate_size = self.num_heads * self.head_dim
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.state_size
        self.in_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size + self.conv_dim + self.num_heads,
            bias=False,
        )
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            bias=True,
        )
        self.dt_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, self.num_heads + 1, dtype=torch.float32))
        )
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.norm = RMSNormGated(self.intermediate_size)
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self._state_shape = (
            self.num_heads,
            self.head_dim,
            self.state_size,
        )

    @property
    def state_shape(self) -> tuple[int, int, int]:
        return self._state_shape

    def forward(
        self,
        hidden_states: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = hidden_states.shape
        projected = self.in_proj(hidden_states)
        gate, bc, dt = torch.split(
            projected,
            [self.intermediate_size, self.conv_dim, self.num_heads],
            dim=-1,
        )
        bc = bc.transpose(1, 2)
        # Evaluate the conv in float32 (weights included) so the recurrence
        # matches the serving-side explicit fp32 tap-sum numerically.
        bc = F.conv1d(
            bc.float(),
            self.conv1d.weight.float(),
            self.conv1d.bias.float(),
            padding=self.conv_kernel_size - 1,
            groups=self.conv_dim,
        )[
            ..., :sequence_length
        ].transpose(1, 2).to(hidden_states.dtype)
        bc = F.silu(bc)
        state_value, state_b, state_c = torch.split(
            bc,
            [
                self.intermediate_size,
                self.n_groups * self.state_size,
                self.n_groups * self.state_size,
            ],
            dim=-1,
        )
        state_value = state_value.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        state_b = state_b.view(
            batch_size, sequence_length, self.n_groups, self.state_size
        )
        state_c = state_c.view(
            batch_size, sequence_length, self.n_groups, self.state_size
        )
        dt = F.softplus(dt + self.dt_bias)
        decay = torch.exp(-dt[..., None] * torch.exp(self.A_log)[..., None])
        group_heads = self.num_heads // self.n_groups
        state_b = state_b.repeat_interleave(group_heads, dim=2)
        state_c = state_c.repeat_interleave(group_heads, dim=2)
        if initial_state is None:
            state = torch.zeros(
                batch_size,
                self.num_heads,
                self.head_dim,
                self.state_size,
                device=hidden_states.device,
                dtype=torch.float32,
            )
        else:
            state = initial_state.float()
        outputs = []
        for step in range(sequence_length):
            state = state * decay[:, step][..., None] + (
                state_b[:, step].unsqueeze(2)
                * state_value[:, step].to(torch.float32)[..., None]
                * dt[:, step].to(torch.float32)[:, :, None, None]
            )
            output = (state * state_c[:, step].to(torch.float32).unsqueeze(2)).sum(-1)
            outputs.append(
                output.reshape(batch_size, self.intermediate_size)
                + self.D.to(output.dtype).repeat_interleave(self.head_dim)[
                    None, :
                ]
                * state_value[:, step].reshape(batch_size, self.intermediate_size)
            )
        scan_output = torch.stack(outputs, dim=1).to(hidden_states.dtype)
        scan_output = self.norm(scan_output, gate)
        final_state = state.to(hidden_states.dtype)
        return self.out_proj(scan_output), final_state



def merge_attention_states(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    block_output: torch.Tensor,
    block_lse: torch.Tensor,
) -> torch.Tensor:
    """Merge prefix and dense-block attention states using exact log-sum-exp."""
    prefix_lse = prefix_lse.float()
    block_lse = block_lse.float()
    merged_lse = torch.logaddexp(prefix_lse, block_lse)
    # Rows where both branches are fully masked carry -inf LSEs; select the
    # zero weights explicitly so exp(-inf - -inf) (a NaN) never leaks through.
    merged_valid = torch.isfinite(merged_lse)
    prefix_weight = torch.where(
        merged_valid,
        torch.exp(prefix_lse - merged_lse),
        torch.zeros_like(merged_lse),
    ).unsqueeze(-1)
    block_weight = torch.where(
        merged_valid,
        torch.exp(block_lse - merged_lse),
        torch.zeros_like(merged_lse),
    ).unsqueeze(-1)
    return (
        prefix_output.float() * prefix_weight
        + block_output.float() * block_weight
    ).to(prefix_output.dtype)

class HSpecTargetKVAttention(nn.Module):
    """Draft attention with externally supplied target KVs."""

    def __init__(self, config: Qwen3Config, layer_idx: int, kernels: DFlashKernels):
        super().__init__()
        if config._attn_implementation not in _VALID_HSPEC_ATTENTION:
            raise ValueError(
                "H-Spec attention_backend must be one of "
                f"{sorted(_VALID_HSPEC_ATTENTION)}, got "
                f"{config._attn_implementation!r}"
            )
        self.config = config
        self.layer_idx = layer_idx
        if config._attn_implementation == "flex_attention":
            if config.attention_dropout != 0.0:
                raise ValueError(
                    "flex_attention does not support attention_dropout; "
                    f"got {config.attention_dropout}"
                )
        self.head_dim = int(
            getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        )
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        if self.num_heads % self.num_key_value_heads:
            raise ValueError("H-Spec requires KV heads to divide attention heads")
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            self.hidden_size_of(config),
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size_of(config),
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size_of(config),
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size_of(config),
            bias=config.attention_bias,
        )
        self.q_norm = kernels.make_rms_norm(self.head_dim, config.rms_norm_eps)
        self.k_norm = kernels.make_rms_norm(self.head_dim, config.rms_norm_eps)

    @staticmethod
    def hidden_size_of(config: Qwen3Config) -> int:
        return int(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_key: torch.Tensor,
        target_value: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        prefix_mask: Optional[torch.Tensor] = None,
        block_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, query_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch_size, query_length, self.num_heads, self.head_dim
        )
        query = self.q_norm(query).transpose(1, 2)
        draft_key = self.k_proj(hidden_states).view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        )
        draft_value = self.v_proj(hidden_states).view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        )
        draft_key = self.k_norm(draft_key).transpose(1, 2)
        draft_value = draft_value.transpose(1, 2)
        cos, sin = position_embeddings
        query, draft_key = apply_rotary_pos_emb(
            query, draft_key, cos[:, -query_length:], sin[:, -query_length:]
        )
        target_key, target_value = self._normalize_target_kv(target_key, target_value)
        merge_mode = self._merge_mode()
        if (
            self.config._attn_implementation != "flex_attention"
            and merge_mode == "split"
        ):
            return self._split_source_attention(
                query=query,
                draft_key=draft_key,
                draft_value=draft_value,
                target_key=target_key,
                target_value=target_value,
                prefix_mask=prefix_mask,
                block_mask=block_mask,
            )
        return self._cat_attention(
            query=query,
            draft_key=draft_key,
            draft_value=draft_value,
            target_key=target_key,
            target_value=target_value,
            prefix_mask=prefix_mask,
            block_mask=block_mask,
        )

    @staticmethod
    def _merge_mode() -> str:
        mode = os.environ.get("SPECFORGE_HSPEC_ATTENTION_MERGE", "split")
        if mode not in ("split", "cat"):
            raise ValueError(
                "SPECFORGE_HSPEC_ATTENTION_MERGE must be 'split' or 'cat', "
                f"got {mode!r}"
            )
        return mode

    def _normalize_target_kv(
        self, target_key: torch.Tensor, target_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize target K/V to ``[batch, kv_heads, seq, head_dim]``.

        Accepted layouts:
        - ``[seq, kv_heads * head_dim]`` (shared across batch)
        - ``[batch, seq, kv_heads * head_dim]`` (offline trainer per-layer slice)
        - ``[batch, kv_heads, seq, head_dim]`` (paged-pool gather)
        - ``[batch, seq, kv_heads, head_dim]``
        """

        kv_heads = self.num_key_value_heads
        head_dim = self.head_dim
        kv_width = kv_heads * head_dim

        def normalize(tensor: torch.Tensor, name: str) -> torch.Tensor:
            if tensor.dim() == 2:
                if tensor.shape[-1] != kv_width:
                    raise ValueError(
                        f"{name} last dimension must be {kv_width}, "
                        f"got {tensor.shape[-1]}"
                    )
                tensor = tensor.unsqueeze(0)
            if tensor.dim() == 3:
                if tensor.shape[-1] != kv_width:
                    raise ValueError(
                        f"{name} last dimension must be {kv_width}, "
                        f"got {tensor.shape[-1]}"
                    )
                return tensor.view(
                    tensor.shape[0], tensor.shape[1], kv_heads, head_dim
                ).transpose(1, 2)
            if tensor.dim() == 4:
                if tensor.shape[1] == kv_heads and tensor.shape[-1] == head_dim:
                    return tensor
                if tensor.shape[2] == kv_heads and tensor.shape[-1] == head_dim:
                    return tensor.transpose(1, 2)
            raise ValueError(
                f"{name} must be [seq, width], [batch, seq, width], "
                "[batch, kv_heads, seq, head_dim], or "
                f"[batch, seq, kv_heads, head_dim], got {tuple(tensor.shape)}"
            )

        return normalize(target_key, "target K"), normalize(target_value, "target V")

    def _dense_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor],
        causal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dense attention returning the output and per-query log-sum-exp.

        Scores and the LSE are accumulated in float32 so that splitting one
        attention into prefix + block branches merges back exactly.  Rows with
        no valid key return zero output and ``-inf`` LSE instead of NaN, which
        keeps empty prefixes and padding-only rows safe in the merge.
        """

        if key.shape[1] != self.num_heads:
            key = key.repeat_interleave(self.num_key_value_groups, dim=1)
            value = value.repeat_interleave(self.num_key_value_groups, dim=1)
        scores = torch.matmul(query.float(), key.float().transpose(2, 3))
        scores = scores * self.scaling
        if causal:
            query_length, key_length = scores.shape[2], scores.shape[3]
            causal_mask = torch.ones(
                query_length,
                key_length,
                device=scores.device,
                dtype=torch.bool,
            ).tril()
            scores = scores.masked_fill(~causal_mask, float("-inf"))
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, :, :], float("-inf"))
        row_valid = torch.isfinite(scores).any(dim=-1)
        safe_scores = torch.where(
            row_valid.unsqueeze(-1), scores, torch.zeros_like(scores)
        )
        lse = torch.logsumexp(safe_scores, dim=-1)
        lse = torch.where(
            row_valid, lse, torch.full_like(lse, float("-inf"))
        )
        probabilities = torch.exp(safe_scores - lse.unsqueeze(-1))
        probabilities = torch.where(
            row_valid.unsqueeze(-1), probabilities, torch.zeros_like(probabilities)
        )
        output = torch.matmul(probabilities.to(value.dtype), value)
        return output.to(query.dtype), lse

    def _split_source_attention(
        self,
        query: torch.Tensor,
        draft_key: torch.Tensor,
        draft_value: torch.Tensor,
        target_key: torch.Tensor,
        target_value: torch.Tensor,
        prefix_mask: Optional[torch.Tensor],
        block_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Borrowed target prefix + transient local block via exact LSE merge.

        The target K/V are post-RoPE and are never concatenated with the draft
        block, so no ACL/NCHW format mixing can occur on Ascend NPU and no
        drafter KV cache needs to exist.

        ``prefix_mask`` follows the DFlash context semantics: either
        ``[batch, prefix]`` key validity or ``[batch, query, prefix]`` per-query
        visibility (each anchor block only sees target tokens before its
        anchor).  ``block_mask`` is ``[batch, query, query]``; the DFlash draft
        objective is bidirectional within each block, so an explicit mask is
        passed whenever block boundaries are known and ``causal`` is only the
        boundary-free fallback.
        """

        batch_size, _, query_length, _ = query.shape
        prefix_length = target_key.shape[2]
        block_output, block_lse = self._dense_attention(
            query=query,
            key=draft_key,
            value=draft_value,
            mask=block_mask,
            causal=block_mask is None,
        )
        if prefix_length == 0:
            merged = block_output
        else:
            if prefix_mask is None:
                prefix_mask = torch.ones(
                    (batch_size, prefix_length),
                    device=query.device,
                    dtype=torch.bool,
                )
            if prefix_mask.dim() == 2:
                prefix_mask = prefix_mask[:, None, :].expand(
                    batch_size, query_length, prefix_length
                )
            prefix_output, prefix_lse = self._dense_attention(
                query=query,
                key=target_key,
                value=target_value,
                mask=prefix_mask,
                causal=False,
            )
            merged = merge_attention_states(
                prefix_output, prefix_lse, block_output, block_lse
            )
        return self.o_proj(
            merged.transpose(1, 2).reshape(batch_size, query_length, -1)
        )

    def _cat_attention(
        self,
        query: torch.Tensor,
        draft_key: torch.Tensor,
        draft_value: torch.Tensor,
        target_key: torch.Tensor,
        target_value: torch.Tensor,
        prefix_mask: Optional[torch.Tensor],
        block_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single fused attention over ``cat(target_prefix, draft_block)``."""

        batch_size, _, query_length, _ = query.shape
        target_key = target_key.to(memory_format=torch.contiguous_format)
        target_value = target_value.to(memory_format=torch.contiguous_format)
        key = torch.cat([target_key.to(draft_key.dtype), draft_key], dim=2)
        value = torch.cat([target_value.to(draft_value.dtype), draft_value], dim=2)
        prefix_length = key.shape[2] - query_length
        if prefix_mask is None:
            prefix_mask = torch.ones(
                (batch_size, prefix_length),
                device=query.device,
                dtype=torch.bool,
            )
        if block_mask is not None:
            draft_block = block_mask
        else:
            draft_block = torch.ones(
                query_length,
                query_length,
                device=query.device,
                dtype=torch.bool,
            ).tril()
        if prefix_mask.dim() == 2:
            prefix_mask = prefix_mask[:, None, :].expand(
                batch_size, query_length, -1
            )
        key_mask = torch.cat(
            [
                prefix_mask,
                draft_block.expand(batch_size, -1, -1),
            ],
            dim=-1,
        )
        if self.config._attn_implementation == "eager":
            from transformers.models.qwen3.modeling_qwen3 import repeat_kv

            attention_mask = torch.zeros_like(key_mask, dtype=query.dtype)
            attention_mask.masked_fill_(~key_mask, torch.finfo(query.dtype).min)
            expanded_key = repeat_kv(key, self.num_key_value_groups)
            expanded_value = repeat_kv(value, self.num_key_value_groups)
            scores = torch.matmul(query, expanded_key.transpose(2, 3)) * self.scaling
            scores = scores + attention_mask.unsqueeze(1)
            probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(
                query.dtype
            )
            output = torch.matmul(probabilities, expanded_value)
        elif self.config._attn_implementation == "sdpa":
            expanded_key = key.repeat_interleave(self.num_key_value_groups, dim=1)
            expanded_value = value.repeat_interleave(self.num_key_value_groups, dim=1)
            additive = torch.zeros_like(key_mask, dtype=query.dtype)
            additive.masked_fill_(~key_mask, torch.finfo(query.dtype).min)
            output = F.scaled_dot_product_attention(
                query,
                expanded_key,
                expanded_value,
                attn_mask=additive.unsqueeze(1),
                scale=self.scaling,
            )
        else:
            from transformers.integrations.flex_attention import (
                compile_friendly_flex_attention,
            )

            from torch.nn.attention.flex_attention import (
                create_block_mask,
            )

            key_length = key.shape[2]
            full_key_mask = key_mask[:, None]

            def mask_mod(batch, _head, query_index, key_index):
                return full_key_mask[batch, 0, query_index, key_index]

            block_mask = create_block_mask(
                mask_mod,
                B=batch_size,
                H=None,
                Q_LEN=query_length,
                KV_LEN=key_length,
                device=query.device,
            )
            kernel_options = {}
            backend = flex_attention_backend()
            if backend is not None:
                kernel_options["BACKEND"] = backend
            output = compile_friendly_flex_attention(
                query,
                key,
                value,
                block_mask=block_mask,
                enable_gqa=True,
                scale=self.scaling,
                kernel_options=kernel_options or None,
            )
        return self.o_proj(output.transpose(1, 2).reshape(
            batch_size, query_length, -1
        ))

class HSpecDecoderLayer(nn.Module):
    """Mamba -> target-KV attention -> MLP hybrid layer."""

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        kernels: DFlashKernels,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = int(config.hidden_size)
        self.mamba = HSpecMamba2Reference(config)
        self.self_attn = HSpecTargetKVAttention(config, layer_idx, kernels)
        self.mlp = kernels.make_mlp(config)
        self.input_layernorm = kernels.make_rms_norm(
            self.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = kernels.make_rms_norm(
            self.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        target_key: torch.Tensor,
        target_value: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        prefix_mask: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        block_mask: Optional[torch.Tensor] = None,
        block_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        batch_size, sequence_length, _ = normalized.shape
        if block_size is not None and sequence_length % block_size == 0:
            # Draft blocks are independent decode steps in serving: each block
            # runs its own recurrence seeded from its anchor latent, so the
            # flattened multi-anchor training sequence must not chain state
            # across blocks.
            per_block = normalized.reshape(
                batch_size * (sequence_length // block_size), block_size, -1
            )
            mamba_output, final_state = self.mamba(per_block, initial_state)
            mamba_output = mamba_output.reshape(
                batch_size, sequence_length, -1
            )
        else:
            mamba_output, final_state = self.mamba(normalized, initial_state)
        hidden_states = residual + mamba_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            target_key=target_key,
            target_value=target_value,
            position_embeddings=position_embeddings,
            prefix_mask=prefix_mask,
            block_mask=block_mask,
        )
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(hidden_states)
        return hidden_states, final_state


@register_draft
class HSpecDraftModel(DSparkDraftModel):
    """Hybrid Mamba-attention parallel drafter (SpecForge first integration)."""

    _no_split_modules = ["HSpecDecoderLayer"]

    def __init__(self, config) -> None:
        method = getattr(config, "hspec_config", None)
        if not isinstance(method, dict):
            raise ValueError("HSpecDraftModel requires config.hspec_config")
        if method.get("mamba_backend", "reference") != "reference":
            raise ValueError(
                "current SpecForge H-Spec skeleton supports "
                "hspec_config.mamba_backend='reference' only"
            )
        super().__init__(config)
        # The base class builds a uniform decoder stack that H-Spec never
        # runs (the hybrid/partial stacks below replace it); keeping it
        # would add dead parameters that break DDP with
        # find_unused_parameters=False.
        del self.layers
        self.num_hybrid_layers = int(method.get("num_hybrid_layers", 3))
        self.num_partial_layers = int(method.get("num_partial_layers", 1))
        if "block_pattern" in method:
            block_pattern = method["block_pattern"]
        else:
            block_pattern = []
            for _ in range(self.num_hybrid_layers):
                block_pattern.extend(["mamba", "attention", "mlp"])
            for _ in range(self.num_partial_layers):
                block_pattern.extend(["mamba", "mlp"])
        if not isinstance(block_pattern, list) or not block_pattern:
            raise ValueError("hspec_config.block_pattern must be a non-empty list")
        if len(block_pattern) != config.num_hidden_layers:
            raise ValueError(
                "hspec_config.block_pattern length must equal "
                f"num_hidden_layers={config.num_hidden_layers}, "
                f"got {len(block_pattern)}"
            )
        if any(
            token not in {"mamba", "attention", "mlp"} for token in block_pattern
        ):
            raise ValueError(
                "hspec_config.block_pattern supports only 'mamba', 'attention', "
                f"'mlp'; got {block_pattern}"
            )
        self.block_pattern = tuple(block_pattern)
        self.debug_only = bool(method.get("debug_only", False))
        if self.debug_only:
            warnings.warn(
                "hspec_config.block_pattern is marked debug_only=True; this "
                "topology is for NPU smoke/debug runs only and must not train "
                "a final checkpoint",
                stacklevel=2,
            )
        self.hybrid_layers = nn.ModuleList(
            [
                HSpecDecoderLayer(config, layer_idx, DEFAULT_DFLASH_KERNELS)
                for layer_idx, token in enumerate(self.block_pattern)
                if token == "mamba"
                and layer_idx + 1 < len(self.block_pattern)
                and self.block_pattern[layer_idx + 1] == "attention"
            ]
        )
        self.partial_layers = nn.ModuleList(
            [
                HSpecMamba2Reference(config)
                for layer_idx, token in enumerate(self.block_pattern)
                if token == "mamba"
                and (
                    layer_idx + 1 == len(self.block_pattern)
                    or self.block_pattern[layer_idx + 1] != "attention"
                )
            ]
        )
        self.partial_norms = nn.ModuleList(
            [
                DEFAULT_DFLASH_KERNELS.make_rms_norm(
                    config.hidden_size, config.rms_norm_eps
                )
                for _ in self.partial_layers
            ]
        )
        self.target_kv_layer_ids = tuple(
            int(layer_id) for layer_id in method.get("target_kv_layer_ids", [])
        )
        if len(self.target_kv_layer_ids) != len(self.hybrid_layers):
            raise ValueError(
                "H-Spec requires one target KV layer per mamba+attention block"
            )
        self.attn_kv_layer_ids = self.target_kv_layer_ids
        self.latent_fusion_layer_ids = tuple(
            int(layer_id)
            for layer_id in method.get(
                "latent_fusion_layer_ids", self.target_layer_ids
            )
        )
        self.target_kv_width = (
            config.num_key_value_heads
            * int(config.head_dim or config.hidden_size // config.num_attention_heads)
        )
        self.tp_shard_count = 1
        # Per-mamba-layer seed projections, matching the serving-side
        # ``seed_projs``: each mamba layer seeds its recurrence from the
        # anchor token's fused latent instead of chaining the previous
        # mamba layer's final state.
        self.mamba_seed_mode = str(method.get("mamba_seed_mode", "per_layer"))
        if self.mamba_seed_mode not in ("per_layer", "shared"):
            raise ValueError(
                "hspec_config.mamba_seed_mode must be 'per_layer' or 'shared', "
                f"got {self.mamba_seed_mode!r}"
            )
        self.mamba_layers = (
            [layer.mamba for layer in self.hybrid_layers]
            + list(self.partial_layers)
        )
        num_seed_projs = (
            len(self.mamba_layers)
            if self.mamba_seed_mode == "per_layer"
            else 1
        )
        if not self.mamba_layers:
            raise ValueError("H-Spec block_pattern must contain mamba layers")
        seed_out = int(
            torch.tensor(self.mamba_layers[0].state_shape).prod()
        )
        self.seed_projs = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, seed_out, bias=False)
                for _ in range(num_seed_projs)
            ]
        )

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)
        if isinstance(module, HSpecMamba2Reference):
            nn.init.zeros_(module.dt_bias)
            nn.init.ones_(module.D)

    def _build_block_seed_states(
        self,
        target_hidden: torch.Tensor,
        anchor_positions: Optional[torch.Tensor],
        num_blocks: int,
    ) -> list[torch.Tensor]:
        """Seed each mamba layer from its block anchor's fused target latent.

        Serving seeds every draft block from the anchor token's latent; the
        training analog gathers ``target_hidden`` at each anchor position and
        projects it per layer (``hidden_norm(fc(latent))`` then
        ``seed_projs[i]``).
        """

        batch_size, sequence_length, _ = target_hidden.shape
        if anchor_positions is None:
            raise ValueError(
                "H-Spec forward requires anchor_positions to seed the mamba "
                "blocks from their anchor latents"
            )
        if anchor_positions.shape != (batch_size, num_blocks):
            raise ValueError(
                "anchor_positions must have shape "
                f"({batch_size}, {num_blocks}), got "
                f"{tuple(anchor_positions.shape)}"
            )
        if (
            int(anchor_positions.min()) < 0
            or int(anchor_positions.max()) >= sequence_length
        ):
            raise ValueError("anchor_positions fall outside target_hidden")
        gather_index = (
            anchor_positions.to(target_hidden.device)
            .reshape(batch_size, num_blocks, 1)
            .expand(batch_size, num_blocks, target_hidden.shape[-1])
        )
        anchor_latent = target_hidden.gather(1, gather_index)
        latent = self.fc(anchor_latent)
        if getattr(self, "fc_norm", None) is not None:
            chunks = torch.chunk(latent, len(self.fc_norm), dim=-1)
            latent = torch.cat(
                [
                    norm(chunk)
                    for norm, chunk in zip(self.fc_norm, chunks, strict=True)
                ],
                dim=-1,
            )
        z = self.hidden_norm(latent)
        z = z.reshape(batch_size * num_blocks, -1)
        state_shape = self.mamba_layers[0].state_shape
        seed_states = []
        for mamba_index in range(len(self.mamba_layers)):
            proj = self.seed_projs[
                mamba_index if self.mamba_seed_mode == "per_layer" else 0
            ]
            seed_states.append(proj(z).reshape(batch_size * num_blocks, *state_shape))
        return seed_states

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[object] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        target_keys: Optional[torch.Tensor] = None,
        target_values: Optional[torch.Tensor] = None,
        selected_target_k: Optional[torch.Tensor] = None,
        selected_target_v: Optional[torch.Tensor] = None,
        prefix_masks: Optional[torch.Tensor] = None,
        target_last_hidden_states: Optional[torch.Tensor] = None,
        anchor_positions: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        del kwargs
        if noise_embedding is None or target_hidden is None:
            raise ValueError("H-Spec forward requires noise and target hidden states")
        if target_keys is None and selected_target_k is not None:
            # Accepted layouts: [prefix, kv_width] (single sample) or
            # [batch, prefix, kv_width].
            if selected_target_k.dim() == 2:
                if selected_target_k.shape[-1] != self.target_kv_width:
                    raise ValueError(
                        "selected_target_k last dimension must be "
                        f"{self.target_kv_width}, got "
                        f"{selected_target_k.shape[-1]}"
                    )
                target_keys = selected_target_k.unsqueeze(0)
            elif selected_target_k.dim() == 3:
                target_keys = selected_target_k
            else:
                raise ValueError(
                    "selected_target_k must be [prefix, kv_width] or "
                    f"[batch, prefix, kv_width], got {tuple(selected_target_k.shape)}"
                )
        if target_values is None and selected_target_v is not None:
            if selected_target_v.dim() == 2:
                if selected_target_v.shape[-1] != self.target_kv_width:
                    raise ValueError(
                        "selected_target_v last dimension must be "
                        f"{self.target_kv_width}, got "
                        f"{selected_target_v.shape[-1]}"
                    )
                target_values = selected_target_v.unsqueeze(0)
            elif selected_target_v.dim() == 3:
                target_values = selected_target_v
            else:
                raise ValueError(
                    "selected_target_v must be [prefix, kv_width] or "
                    f"[batch, prefix, kv_width], got {tuple(selected_target_v.shape)}"
                )
        if target_hidden.dim() == 2:
            target_hidden = target_hidden.unsqueeze(0)
        if target_keys is None or target_values is None:
            raise ValueError("H-Spec forward requires selected target K and V tensors")
        if target_last_hidden_states is None:
            raise ValueError("H-Spec forward requires target_last_hidden_states")
        batch_size, query_length, _ = noise_embedding.shape
        prefix_length = target_keys.shape[1]
        num_blocks = query_length // self.block_size
        if query_length % self.block_size:
            raise ValueError(
                "noise_embedding sequence length must be a multiple of "
                f"block_size={self.block_size}, got {query_length}"
            )
        # Consume the DFlash attention mask: its first ``prefix_length``
        # columns carry per-query context visibility (each anchor block only
        # sees target tokens before its anchor) and the remainder is the
        # same-block bidirectional draft mask.  Without it the draft block
        # falls back to a boundary-free causal mask and the prefix to an
        # all-valid mask, which reintroduces the future-token leak.
        per_query_prefix = None
        block_mask = None
        flat_attention_mask = attention_mask
        if isinstance(flat_attention_mask, dict):
            flat_attention_mask = flat_attention_mask.get("full_attention")
        if (
            isinstance(flat_attention_mask, torch.Tensor)
            and flat_attention_mask.dim() == 4
            and flat_attention_mask.shape[1] == 1
        ):
            # create_dflash_sdpa_mask emits [batch, 1, query, prefix + query]
            flat_attention_mask = flat_attention_mask.squeeze(1)
        if (
            isinstance(flat_attention_mask, torch.Tensor)
            and flat_attention_mask.dim() == 3
            and tuple(flat_attention_mask.shape) == (
                batch_size,
                query_length,
                prefix_length + query_length,
            )
        ):
            per_query_prefix = flat_attention_mask[..., :prefix_length]
            block_mask = flat_attention_mask[..., prefix_length:]
        elif flat_attention_mask is not None:
            # An uninterpretable mask must not silently degrade to an
            # all-visible prefix: that would leak future target K/V into the
            # draft objective.
            raise ValueError(
                "H-Spec forward requires a dense [batch, query, "
                "prefix + query] DFlash attention mask (or None), got shape "
                f"{tuple(flat_attention_mask.shape)}"
            )
        else:
            block_mask = (
                torch.arange(query_length, device=noise_embedding.device)[None, :, None]
                // self.block_size
                == torch.arange(query_length, device=noise_embedding.device)[
                    None, None, :
                ]
                // self.block_size
            ).expand(batch_size, -1, -1)
        if prefix_masks is None:
            prefix_masks = torch.ones(
                (batch_size, prefix_length),
                device=target_keys.device,
                dtype=torch.bool,
            )
        seed_states = self._build_block_seed_states(
            target_hidden, anchor_positions, num_blocks
        )
        cos, sin = self.rotary_emb(noise_embedding, position_ids)
        hidden_states = noise_embedding
        target_kv_width = self.target_kv_width
        mamba_index = 0
        for layer_index, hybrid_layer in enumerate(self.hybrid_layers):
            start = layer_index * target_kv_width
            end = start + target_kv_width
            layer_key = target_keys[..., start:end]
            layer_value = target_values[..., start:end]
            hidden_states, _ = hybrid_layer(
                hidden_states,
                target_key=layer_key,
                target_value=layer_value,
                initial_state=seed_states[mamba_index],
                prefix_mask=(
                    per_query_prefix
                    if per_query_prefix is not None
                    else prefix_masks
                ),
                position_embeddings=(cos, sin),
                block_mask=block_mask,
                block_size=self.block_size,
            )
            mamba_index += 1
        for partial_norm, partial_layer in zip(self.partial_norms, self.partial_layers):
            batch_size, sequence_length, _ = hidden_states.shape
            normalized = partial_norm(hidden_states).reshape(
                batch_size * num_blocks, self.block_size, -1
            )
            partial_output, _ = partial_layer(
                normalized,
                initial_state=seed_states[mamba_index],
            )
            hidden_states = partial_output.reshape(
                batch_size, sequence_length, -1
            )
            mamba_index += 1
        return self.norm(hidden_states)


__all__ = ["HSpecDraftModel", "HSpecDecoderLayer", "HSpecMamba2Reference"]
