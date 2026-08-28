"""Kimi Delta Attention for DFlash-family draft models.

KDA is recurrent rather than KV-cache attention. During block-parallel draft
training, every proposal block is therefore an independent sequence: recurrent
state resets at each block boundary and cannot bypass the DFlash mask. Target
context is injected by another attention layer in the hybrid stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import FlashAttentionKwargs, Qwen3Config
from typing_extensions import Unpack

from .dflash import Qwen3DFlashAttentionBase
from .dflash_kernels import DFlashKernels

_CUDA_MAX_GRID_DIM_Z = 65_535
_KDA_BACKENDS = {"fla", "reference"}


@dataclass(frozen=True)
class KDAConfig:
    """Validated KDA dimensions and execution policy."""

    hidden_size: int
    head_dim: int
    num_heads: int
    block_size: int
    short_conv_kernel_size: int
    use_full_rank_gate: bool
    gate_lower_bound: Optional[float]
    backend: str

    @property
    def projection_size(self) -> int:
        return self.num_heads * self.head_dim

    @classmethod
    def from_config(cls, config: Qwen3Config) -> KDAConfig:
        linear_config = dict(getattr(config, "linear_attn_config", None) or {})
        required = ("head_dim", "num_heads", "short_conv_kernel_size")
        missing = [name for name in required if linear_config.get(name) is None]
        if missing:
            raise ValueError(
                f"KDA linear_attn_config is missing required fields: {missing}"
            )

        dimensions = {
            "hidden_size": int(config.hidden_size),
            "head_dim": int(linear_config["head_dim"]),
            "num_heads": int(linear_config["num_heads"]),
            "block_size": int(config.block_size),
            "short_conv_kernel_size": int(linear_config["short_conv_kernel_size"]),
        }
        for name, value in dimensions.items():
            if value <= 0:
                raise ValueError(f"KDA {name} must be positive, got {value}")

        use_full_rank_gate = linear_config.get("use_full_rank_gate", False)
        if not isinstance(use_full_rank_gate, bool):
            raise ValueError(
                "KDA linear_attn_config.use_full_rank_gate must be a boolean, "
                f"got {use_full_rank_gate!r}"
            )

        lower_bound = linear_config.get("gate_lower_bound")
        if lower_bound is not None:
            lower_bound = float(lower_bound)
            if not math.isfinite(lower_bound) or lower_bound >= 0:
                raise ValueError(
                    "KDA linear_attn_config.gate_lower_bound must be a finite "
                    f"negative number or null, got {lower_bound!r}"
                )

        backend = str(linear_config.get("backend", "fla")).lower()
        if backend not in _KDA_BACKENDS:
            raise ValueError(
                "KDA linear_attn_config.backend must be one of "
                f"{sorted(_KDA_BACKENDS)}, got {backend!r}"
            )
        return cls(
            **dimensions,
            use_full_rank_gate=use_full_rank_gate,
            gate_lower_bound=lower_bound,
            backend=backend,
        )


def validate_dflash_kda_config(config: Qwen3Config) -> None:
    """Validate KDA fields without importing the optional FLA backend."""

    KDAConfig.from_config(config)


class KDAShortConvolution(nn.Module):
    """Causal depthwise convolution with Kimi checkpoint-compatible weights."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(channels, kernel_size))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        channels_first = inputs.transpose(1, 2)
        channels_first = F.pad(channels_first, (self.kernel_size - 1, 0))
        convolved = F.conv1d(
            channels_first,
            self.weight.unsqueeze(1),
            bias=None,
            groups=self.weight.shape[0],
        )
        return F.silu(convolved.transpose(1, 2))


class KDAGatedRMSNorm(nn.Module):
    """RMSNorm followed by KDA's sigmoid output gate."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, inputs: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        variance = inputs.float().square().mean(dim=-1, keepdim=True)
        normalized = inputs * torch.rsqrt(variance + self.eps).to(inputs.dtype)
        return normalized * self.weight.to(inputs.dtype) * torch.sigmoid(gate)


def reference_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: Optional[float],
) -> torch.Tensor:
    """Differentiable KDA recurrence used as the correctness oracle."""

    q = F.normalize(q.float(), dim=-1).to(q.dtype)
    k = F.normalize(k.float(), dim=-1).to(k.dtype)
    beta = torch.sigmoid(beta.float()).to(q.dtype)

    gate_input = raw_gate.float() + dt_bias.view(1, 1, *raw_gate.shape[-2:])
    decay_scale = A_log.float().exp().view(1, 1, -1, 1)
    if lower_bound is None:
        log_decay = -decay_scale * F.softplus(gate_input)
    else:
        log_decay = float(lower_bound) * torch.sigmoid(decay_scale * gate_input)

    state = torch.zeros(
        q.shape[0],
        q.shape[2],
        q.shape[3],
        v.shape[3],
        dtype=torch.float32,
        device=q.device,
    )
    outputs = []
    score_scale = q.shape[-1] ** -0.5
    for step in range(q.shape[1]):
        state = state * log_decay[:, step].exp().unsqueeze(-1)
        step_key = k[:, step].float()
        step_value = v[:, step].float()
        prediction = torch.einsum("bhd,bhdv->bhv", step_key, state)
        delta = (step_value - prediction) * beta[:, step].float().unsqueeze(-1)
        state = state + torch.einsum("bhd,bhv->bhdv", step_key, delta)
        output = torch.einsum("bhd,bhdv->bhv", q[:, step].float(), state)
        outputs.append((output * score_scale).to(q.dtype))
    return torch.stack(outputs, dim=1)


def _load_fla_chunk_kda() -> Callable[..., tuple[torch.Tensor, object]]:
    try:
        from fla.ops.kda import chunk_kda
    except ImportError as exc:
        raise ImportError(
            "KDA training with backend='fla' requires fla-core==0.5.1; "
            "install SpecForge with the 'kda' extra"
        ) from exc
    return chunk_kda


def _pad_batch_to(tensor: torch.Tensor, size: int) -> torch.Tensor:
    padding = size - int(tensor.shape[0])
    if padding <= 0:
        return tensor
    return torch.cat(
        (tensor, tensor.new_zeros((padding, *tensor.shape[1:]))),
        dim=0,
    )


def fla_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: Optional[float],
) -> torch.Tensor:
    """Run FLA KDA while respecting the CUDA grid-z launch limit."""

    chunk_kda = _load_fla_chunk_kda()
    max_blocks_per_launch = max(1, _CUDA_MAX_GRID_DIM_Z // int(q.shape[2]))
    outputs = []
    for start in range(0, int(q.shape[0]), max_blocks_per_launch):
        end = min(start + max_blocks_per_launch, int(q.shape[0]))
        batch_size = end - start
        # TileLang specializes KDA on the independent-sequence count. DSpark's
        # valid anchor width changes per sample, so launching the exact width
        # would compile hundreds of kernels during a training run. Power-of-two
        # buckets bound the specialization set without imposing a fixed anchor
        # count or leaking KDA policy into the shared DFlash training path.
        padded_size = min(
            1 << (batch_size - 1).bit_length(),
            max_blocks_per_launch,
        )
        output, _ = chunk_kda(
            q=_pad_batch_to(q[start:end], padded_size),
            k=_pad_batch_to(k[start:end], padded_size),
            v=_pad_batch_to(v[start:end], padded_size),
            g=_pad_batch_to(raw_gate[start:end], padded_size),
            beta=_pad_batch_to(beta[start:end], padded_size),
            # FSDP1 requires each decoder block to have one parameter dtype.
            # SpecForge therefore shards these as BF16 (with FP32 optimizer
            # masters) and materializes the tiny FP32 kernel inputs here.
            A_log=A_log.float(),
            dt_bias=dt_bias.float(),
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=lower_bound is not None,
            lower_bound=lower_bound,
        )
        outputs.append(output[:batch_size])
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


class Qwen3DFlashKDAAttention(Qwen3DFlashAttentionBase):
    """KDA applied independently to every block-parallel proposal."""

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        kernels: DFlashKernels,
    ) -> None:
        super().__init__(config, layer_idx, kernels)
        self.kda_config = KDAConfig.from_config(config)
        spec = self.kda_config

        self.head_dim = spec.head_dim
        self.num_heads = spec.num_heads
        self.block_size = spec.block_size
        self.backend = spec.backend
        self.lower_bound = spec.gate_lower_bound
        projection_size = spec.projection_size

        self.q_proj = nn.Linear(spec.hidden_size, projection_size, bias=False)
        self.k_proj = nn.Linear(spec.hidden_size, projection_size, bias=False)
        self.v_proj = nn.Linear(spec.hidden_size, projection_size, bias=False)
        self.q_conv1d = KDAShortConvolution(
            projection_size, spec.short_conv_kernel_size
        )
        self.k_conv1d = KDAShortConvolution(
            projection_size, spec.short_conv_kernel_size
        )
        self.v_conv1d = KDAShortConvolution(
            projection_size, spec.short_conv_kernel_size
        )

        self.A_log = nn.Parameter(
            torch.log(torch.empty(spec.num_heads, dtype=torch.float32).uniform_(1, 16))
        )
        self.f_a_proj = nn.Linear(spec.hidden_size, spec.head_dim, bias=False)
        self.f_b_proj = nn.Linear(spec.head_dim, projection_size, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(projection_size, dtype=torch.float32))
        self.b_proj = nn.Linear(spec.hidden_size, spec.num_heads, bias=False)
        if spec.use_full_rank_gate:
            self.g_proj = nn.Linear(spec.hidden_size, projection_size, bias=False)
        else:
            self.g_a_proj = nn.Linear(spec.hidden_size, spec.head_dim, bias=False)
            self.g_b_proj = nn.Linear(spec.head_dim, projection_size, bias=False)
        self.o_norm = KDAGatedRMSNorm(spec.head_dim, eps=float(config.rms_norm_eps))
        self.o_proj = nn.Linear(projection_size, spec.hidden_size, bias=False)

    def _independent_blocks(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, int, int]:
        batch_size, query_len, hidden_size = hidden_states.shape
        if query_len % self.block_size:
            raise ValueError(
                "KDA draft query length must be divisible by block_size; "
                f"got query_len={query_len}, block_size={self.block_size}"
            )
        num_blocks = query_len // self.block_size
        blocks = hidden_states.reshape(
            batch_size * num_blocks, self.block_size, hidden_size
        )
        return blocks, batch_size, query_len

    def _output_gate(self, blocks: torch.Tensor) -> torch.Tensor:
        if self.kda_config.use_full_rank_gate:
            return self.g_proj(blocks)
        return self.g_b_proj(self.g_a_proj(blocks))

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Dense/MLA siblings may use the shared cache during SpecForge decode.
        # KDA deliberately leaves it untouched because every proposal block is
        # an independent recurrent sequence.
        del (
            target_hidden,
            position_embeddings,
            attention_mask,
            past_key_values,
            cache_position,
            kwargs,
        )

        blocks, batch_size, query_len = self._independent_blocks(hidden_states)
        block_count = blocks.shape[0]
        projection_shape = (
            block_count,
            self.block_size,
            self.num_heads,
            self.head_dim,
        )
        q = self.q_conv1d(self.q_proj(blocks)).view(projection_shape)
        k = self.k_conv1d(self.k_proj(blocks)).view(projection_shape)
        v = self.v_conv1d(self.v_proj(blocks)).view(projection_shape)
        raw_gate = self.f_b_proj(self.f_a_proj(blocks)).view(projection_shape)
        beta = self.b_proj(blocks).view(block_count, self.block_size, self.num_heads)

        kda_fn = reference_kda if self.backend == "reference" else fla_kda
        attention_output = kda_fn(
            q,
            k,
            v,
            raw_gate,
            beta,
            self.A_log,
            self.dt_bias,
            self.lower_bound,
        )
        output_gate = self._output_gate(blocks).view(projection_shape)
        attention_output = self.o_norm(attention_output, output_gate)
        attention_output = self.o_proj(attention_output.flatten(-2))
        attention_output = attention_output.reshape(
            batch_size, query_len, self.kda_config.hidden_size
        )
        return attention_output, None


__all__ = [
    "KDAConfig",
    "Qwen3DFlashKDAAttention",
    "fla_kda",
    "reference_kda",
    "validate_dflash_kda_config",
]
