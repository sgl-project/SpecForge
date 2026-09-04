"""Kimi Delta Attention for DFlash-family draft models.

KDA is recurrent rather than KV-cache attention, so the layer has to decide
where the recurrent state of a proposal block starts. ``linear_attn_config``
selects one of two policies through ``context_state``:

* ``"reset"`` (default): every proposal block is an independent sequence. The
  state starts from zero at each block boundary, the DFlash mask cannot be
  bypassed by a linear-attention scan, and target context reaches the layer
  only through the hybrid stack's GQA/MHA/MLA layers.
* ``"scan"``: the recurrence first consumes the target context, so a block
  anchored at position ``t`` starts from the state that has read positions
  strictly before ``t`` (the same visibility as the DFlash attention mask) and
  its short convolution sees the last context rows. Block-parallel training
  produces one state per anchor with a two-level segment scan; SpecForge
  generation keeps a per-request running state that is advanced with every
  newly verified context slice.
"""

from __future__ import annotations

import itertools
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
_KDA_CONTEXT_STATES = ("reset", "scan")
# FLA processes chunks of 64 tokens; variable-length launches are padded to it.
_FLA_CHUNK_SIZE = 64


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
    context_state: str = "reset"

    @property
    def projection_size(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def scans_context(self) -> bool:
        return self.context_state == "scan"

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

        context_state = str(linear_config.get("context_state", "reset")).lower()
        if context_state not in _KDA_CONTEXT_STATES:
            raise ValueError(
                "KDA linear_attn_config.context_state must be one of "
                f"{list(_KDA_CONTEXT_STATES)}, got {context_state!r}"
            )
        return cls(
            **dimensions,
            use_full_rank_gate=use_full_rank_gate,
            gate_lower_bound=lower_bound,
            backend=backend,
            context_state=context_state,
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
    *,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    """Differentiable KDA recurrence used as the correctness oracle.

    Mirrors the FLA ``chunk_kda`` contract: ``initial_state`` (``[N, H, K, V]``)
    seeds the recurrence, ``output_final_state`` also returns the state after
    the last step, and ``cu_seqlens`` selects the variable-length layout (a
    batch of one with sequences flattened along time and one initial/final
    state per sequence).
    """

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                "variable-length KDA expects a flattened batch of size 1, "
                f"got {q.shape[0]}"
            )
        bounds = [int(bound) for bound in cu_seqlens.tolist()]
        outputs, final_states = [], []
        for index, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            state = None if initial_state is None else initial_state[index : index + 1]
            output, final_state = reference_kda(
                q[:, start:end],
                k[:, start:end],
                v[:, start:end],
                raw_gate[:, start:end],
                beta[:, start:end],
                A_log,
                dt_bias,
                lower_bound,
                initial_state=state,
                output_final_state=True,
            )
            outputs.append(output)
            final_states.append(final_state)
        output = torch.cat(outputs, dim=1)
        if output_final_state:
            return output, torch.cat(final_states, dim=0)
        return output

    q = F.normalize(q.float(), dim=-1).to(q.dtype)
    k = F.normalize(k.float(), dim=-1).to(k.dtype)
    beta = torch.sigmoid(beta.float()).to(q.dtype)

    gate_input = raw_gate.float() + dt_bias.view(1, 1, *raw_gate.shape[-2:])
    decay_scale = A_log.float().exp().view(1, 1, -1, 1)
    if lower_bound is None:
        log_decay = -decay_scale * F.softplus(gate_input)
    else:
        log_decay = float(lower_bound) * torch.sigmoid(decay_scale * gate_input)

    if initial_state is None:
        state = torch.zeros(
            q.shape[0],
            q.shape[2],
            q.shape[3],
            v.shape[3],
            dtype=torch.float32,
            device=q.device,
        )
    else:
        state = initial_state.to(device=q.device, dtype=torch.float32)
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
    if outputs:
        output = torch.stack(outputs, dim=1)
    else:
        output = q.new_zeros((q.shape[0], 0, q.shape[2], v.shape[3]))
    if output_final_state:
        return output, state
    return output


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


def _pad_time_to(tensor: torch.Tensor, size: int) -> torch.Tensor:
    padding = size - int(tensor.shape[1])
    if padding <= 0:
        return tensor
    return torch.cat(
        (tensor, tensor.new_zeros((tensor.shape[0], padding, *tensor.shape[2:]))),
        dim=1,
    )


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def _fla_kernel_kwargs(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: Optional[float],
) -> dict:
    return {
        # FSDP1 requires each decoder block to have one parameter dtype.
        # SpecForge therefore shards these as BF16 (with FP32 optimizer
        # masters) and materializes the tiny FP32 kernel inputs here.
        "A_log": A_log.float(),
        "dt_bias": dt_bias.float(),
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "safe_gate": lower_bound is not None,
        "lower_bound": lower_bound,
    }


def _fla_kda_varlen(
    chunk_kda: Callable[..., tuple[torch.Tensor, object]],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cu_seqlens: torch.Tensor,
    kernel_kwargs: dict,
):
    """One variable-length FLA launch over flattened context segments.

    Anchor segments have arbitrary lengths and counts, and FLA/TileLang
    specialize kernels on launch shapes. The flattened batch is therefore
    padded with dummy sequences (one token each, the last one absorbing the
    remainder) to a power-of-two sequence count and a chunk-aligned token
    count. Dummy sequences never share state with real ones; their outputs and
    final states are sliced away.
    """

    if q.shape[0] != 1:
        raise ValueError(
            "variable-length KDA expects a flattened batch of size 1, "
            f"got {q.shape[0]}"
        )
    bounds = cu_seqlens.to(device=q.device, dtype=torch.long)
    num_sequences = int(bounds.numel()) - 1
    total = int(q.shape[1])
    padded_sequences = 1 << max(num_sequences - 1, 0).bit_length()
    extra_sequences = padded_sequences - num_sequences
    padded_total = _round_up(total + extra_sequences, _FLA_CHUNK_SIZE)
    if padded_total > total and extra_sequences == 0:
        # The length remainder needs a dummy sequence to live in.
        padded_sequences *= 2
        extra_sequences = padded_sequences - num_sequences
        padded_total = _round_up(total + extra_sequences, _FLA_CHUNK_SIZE)
    if extra_sequences:
        dummy_ends = total + torch.arange(
            1, extra_sequences + 1, device=bounds.device, dtype=torch.long
        )
        dummy_ends[-1] = padded_total
        bounds = torch.cat((bounds, dummy_ends))
    state = initial_state
    if state is not None and extra_sequences:
        state = _pad_batch_to(state, padded_sequences)
    output, final_state = chunk_kda(
        q=_pad_time_to(q, padded_total),
        k=_pad_time_to(k, padded_total),
        v=_pad_time_to(v, padded_total),
        g=_pad_time_to(raw_gate, padded_total),
        beta=_pad_time_to(beta, padded_total),
        initial_state=state,
        output_final_state=output_final_state,
        cu_seqlens=bounds,
        **kernel_kwargs,
    )
    output = output[:, :total]
    if output_final_state:
        return output, final_state[:num_sequences]
    return output


def fla_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: Optional[float],
    *,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    """Run FLA KDA while respecting the CUDA grid-z launch limit.

    Same optional state contract as :func:`reference_kda`.
    """

    chunk_kda = _load_fla_chunk_kda()
    kernel_kwargs = _fla_kernel_kwargs(A_log, dt_bias, lower_bound)
    if cu_seqlens is not None:
        return _fla_kda_varlen(
            chunk_kda,
            q,
            k,
            v,
            raw_gate,
            beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            kernel_kwargs=kernel_kwargs,
        )

    max_blocks_per_launch = max(1, _CUDA_MAX_GRID_DIM_Z // int(q.shape[2]))
    outputs, final_states = [], []
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
        launch_kwargs = {
            "q": _pad_batch_to(q[start:end], padded_size),
            "k": _pad_batch_to(k[start:end], padded_size),
            "v": _pad_batch_to(v[start:end], padded_size),
            "g": _pad_batch_to(raw_gate[start:end], padded_size),
            "beta": _pad_batch_to(beta[start:end], padded_size),
        }
        if initial_state is not None:
            launch_kwargs["initial_state"] = _pad_batch_to(
                initial_state[start:end], padded_size
            )
        output, final_state = chunk_kda(
            **launch_kwargs,
            output_final_state=output_final_state,
            **kernel_kwargs,
        )
        outputs.append(output[:batch_size])
        if output_final_state:
            final_states.append(final_state[:batch_size])
    output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
    if output_final_state:
        final_state = (
            final_states[0] if len(final_states) == 1 else torch.cat(final_states)
        )
        return output, final_state
    return output


def _scan_segments(
    kda_fn: Callable,
    tensors: tuple[torch.Tensor, ...],
    segments: list[tuple[int, int]],
    initial_states: list[torch.Tensor],
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: Optional[float],
) -> torch.Tensor:
    """Advance several disjoint context segments in one variable-length launch."""

    q, k, v, raw_gate, beta = tensors
    lengths = [end - start for start, end in segments]
    cu_seqlens = torch.tensor(
        [0, *itertools.accumulate(lengths)], device=k.device, dtype=torch.long
    )

    def gather(tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat([tensor[:, start:end] for start, end in segments], dim=1)

    _, final_states = kda_fn(
        gather(q),
        gather(k),
        gather(v),
        gather(raw_gate),
        gather(beta),
        A_log,
        dt_bias,
        lower_bound,
        initial_state=torch.cat(initial_states, dim=0),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    return final_states


def _scan_row_states(
    kda_fn: Callable,
    tensors: tuple[torch.Tensor, ...],
    anchors: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: Optional[float],
    group_size: Optional[int],
) -> torch.Tensor:
    """States before every anchor of one context row, ``[N, H, K, V]``."""

    _, k, v, _, _ = tensors
    unique_anchors, inverse = torch.unique(anchors, sorted=True, return_inverse=True)
    boundaries = [0, *unique_anchors.tolist()]
    num_segments = len(boundaries) - 1
    group = group_size or max(1, math.isqrt(num_segments))
    num_groups = -(-num_segments // group)
    zero_state = k.new_zeros(
        (1, k.shape[2], k.shape[3], v.shape[3]), dtype=torch.float32
    )

    # Level 1: sequential over groups, one launch per group span.
    group_inputs, group_outputs = [], []
    state = zero_state
    for index in range(num_groups):
        group_inputs.append(state)
        start = boundaries[index * group]
        end = boundaries[min((index + 1) * group, num_segments)]
        if end > start:
            state = _scan_segments(
                kda_fn, tensors, [(start, end)], [state], A_log, dt_bias, lower_bound
            )
        group_outputs.append(state)

    # Level 2: the same in-group offset of every group shares one launch. The
    # last segment of a group ends at the group boundary and reuses level 1.
    anchor_states: list[Optional[torch.Tensor]] = [None] * num_segments
    current = list(group_inputs)
    for offset in range(group):
        launch: list[tuple[int, int, int]] = []
        for index in range(num_groups):
            segment = index * group + offset
            last = min((index + 1) * group, num_segments) - 1
            if segment > last:
                continue
            if segment == last:
                anchor_states[segment] = group_outputs[index]
                continue
            start, end = boundaries[segment], boundaries[segment + 1]
            if end == start:
                anchor_states[segment] = current[index]
                continue
            launch.append((index, start, end))
        if launch:
            final_states = _scan_segments(
                kda_fn,
                tensors,
                [(start, end) for _, start, end in launch],
                [current[index] for index, _, _ in launch],
                A_log,
                dt_bias,
                lower_bound,
            )
            for position, (index, _, _) in enumerate(launch):
                current[index] = final_states[position : position + 1]
                anchor_states[index * group + offset] = current[index]
    stacked = torch.cat(anchor_states, dim=0)
    return stacked[inverse]


def scan_kda_context_states(
    kda_fn: Callable,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: Optional[float],
    anchor_positions: torch.Tensor,
    *,
    group_size: Optional[int] = None,
) -> torch.Tensor:
    """Recurrent state after the context positions strictly before each anchor.

    ``k``/``v`` are the convolved context projections ``[B, S, H, D]``,
    ``raw_gate`` ``[B, S, H, D]`` and ``beta`` ``[B, S, H]`` the context gates,
    and ``anchor_positions`` ``[B, N]`` the anchor of every proposal block
    (values are clamped to ``[0, S]``). Returns fp32 states ``[B, N, H, K, V]``.

    Rows are independent. Within a row the unique sorted anchors cut the
    context into segments, and a two-level scan (sequential over groups of
    ``group_size`` segments, then one variable-length launch per in-group
    offset across all groups) reaches every anchor with about ``2 * sqrt(N)``
    launches instead of ``N`` sequential ones while staying exact.
    """

    batch_size, context_len = k.shape[:2]
    if anchor_positions.shape[0] != batch_size:
        raise ValueError(
            "anchor_positions must carry one row per context row, got "
            f"{tuple(anchor_positions.shape)} for {batch_size} rows"
        )
    anchors = anchor_positions.to(torch.long).clamp(0, context_len)
    # The recurrent state does not depend on queries; zeros keep the launch
    # shape uniform without paying for a context query projection.
    queries = torch.zeros_like(k)
    rows = []
    for row in range(batch_size):
        tensors = tuple(
            tensor[row : row + 1] for tensor in (queries, k, v, raw_gate, beta)
        )
        rows.append(
            _scan_row_states(
                kda_fn, tensors, anchors[row], A_log, dt_bias, lower_bound, group_size
            )
        )
    return torch.stack(rows, dim=0)


def _gather_left_rows(
    context: torch.Tensor,
    anchors: torch.Tensor,
    window: int,
) -> torch.Tensor:
    """The ``window`` context rows before each anchor, zero where none exist."""

    batch_size, _, hidden_size = context.shape
    num_blocks = anchors.shape[1]
    if window == 0:
        return context.new_zeros((batch_size, num_blocks, 0, hidden_size))
    offsets = torch.arange(window, 0, -1, device=anchors.device)
    positions = anchors.unsqueeze(-1) - offsets
    valid = positions >= 0
    flat = positions.clamp(min=0).reshape(batch_size, -1)
    rows = context.gather(1, flat.unsqueeze(-1).expand(-1, -1, hidden_size))
    rows = rows.reshape(batch_size, num_blocks, window, hidden_size)
    return rows * valid.unsqueeze(-1).to(rows.dtype)


class Qwen3DFlashKDAAttention(Qwen3DFlashAttentionBase):
    """KDA over proposal blocks with block-local or context-scanned state."""

    # The decoder layer forwards training anchors only to layers that ask.
    uses_anchor_positions = True

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
        self.context_state = spec.context_state
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

        # Per-request running context for the "scan" policy during generation.
        # Plain attributes: never checkpointed, reset per generated sequence.
        self._running_state: Optional[torch.Tensor] = None
        self._running_tail: Optional[torch.Tensor] = None

    @property
    def scans_context(self) -> bool:
        return self.kda_config.scans_context

    @property
    def _kda_fn(self) -> Callable:
        return reference_kda if self.backend == "reference" else fla_kda

    def reset_state(self) -> None:
        """Forget the running context state kept across generation steps."""

        self._running_state = None
        self._running_tail = None

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

    def _gates(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw decay gate ``[..., H, D]`` and raw beta ``[..., H]`` for rows."""

        leading = rows.shape[:-1]
        raw_gate = self.f_b_proj(self.f_a_proj(rows)).view(
            *leading, self.num_heads, self.head_dim
        )
        beta = self.b_proj(rows).view(*leading, self.num_heads)
        return raw_gate, beta

    def _convolve_with_left_rows(
        self,
        conv: KDAShortConvolution,
        proj: nn.Linear,
        left_rows: torch.Tensor,
        rows: torch.Tensor,
    ) -> torch.Tensor:
        """Convolve ``rows`` as if ``left_rows`` immediately preceded them."""

        joined = torch.cat((proj(left_rows), proj(rows)), dim=1)
        return conv(joined)[:, left_rows.shape[1] :]

    def _scan_rows(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Advance one state per row over that row's context slice."""

        batch_size, length = k.shape[:2]
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * length, length, device=k.device, dtype=torch.long
        )
        _, state = self._kda_fn(
            torch.zeros_like(k).reshape(1, batch_size * length, *k.shape[2:]),
            k.reshape(1, batch_size * length, *k.shape[2:]),
            v.reshape(1, batch_size * length, *v.shape[2:]),
            raw_gate.reshape(1, batch_size * length, *raw_gate.shape[2:]),
            beta.reshape(1, batch_size * length, *beta.shape[2:]),
            self.A_log,
            self.dt_bias,
            self.lower_bound,
            initial_state=state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )
        return state

    def _advance_running_state(
        self, target_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Consume a newly verified context slice during generation."""

        batch_size, length, hidden_size = target_hidden.shape
        window = self.kda_config.short_conv_kernel_size - 1
        tail, state = self._running_tail, self._running_state
        if tail is None or state is None or tail.shape[0] != batch_size:
            tail = target_hidden.new_zeros((batch_size, window, hidden_size))
            state = target_hidden.new_zeros(
                (batch_size, self.num_heads, self.head_dim, self.head_dim),
                dtype=torch.float32,
            )
        if length > 0:
            shape = (batch_size, length, self.num_heads, self.head_dim)
            k = self._convolve_with_left_rows(
                self.k_conv1d, self.k_proj, tail, target_hidden
            ).view(shape)
            v = self._convolve_with_left_rows(
                self.v_conv1d, self.v_proj, tail, target_hidden
            ).view(shape)
            raw_gate, beta = self._gates(target_hidden)
            state = self._scan_rows(k, v, raw_gate, beta, state)
            if window:
                tail = torch.cat((tail, target_hidden), dim=1)[:, -window:]
        self._running_state = state.detach()
        self._running_tail = tail.detach()
        return state, tail

    def _anchor_context(
        self,
        target_hidden: torch.Tensor,
        anchor_positions: Optional[torch.Tensor],
        batch_size: int,
        num_blocks: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-block initial states and left conv rows for block-parallel use."""

        context_len = target_hidden.shape[1]
        window = self.kda_config.short_conv_kernel_size - 1
        if anchor_positions is None:
            anchors = target_hidden.new_full(
                (batch_size, num_blocks), context_len, dtype=torch.long
            )
        else:
            anchors = anchor_positions.to(device=target_hidden.device, dtype=torch.long)
            if tuple(anchors.shape) != (batch_size, num_blocks):
                raise ValueError(
                    "anchor_positions must have shape (batch, num_blocks)="
                    f"{(batch_size, num_blocks)}, got {tuple(anchors.shape)}"
                )
            anchors = anchors.clamp(0, context_len)
        shape = (batch_size, context_len, self.num_heads, self.head_dim)
        k = self.k_conv1d(self.k_proj(target_hidden)).view(shape)
        v = self.v_conv1d(self.v_proj(target_hidden)).view(shape)
        raw_gate, beta = self._gates(target_hidden)
        states = scan_kda_context_states(
            self._kda_fn,
            k,
            v,
            raw_gate,
            beta,
            self.A_log,
            self.dt_bias,
            self.lower_bound,
            anchors,
        )
        left_rows = _gather_left_rows(target_hidden, anchors, window)
        return states, left_rows

    def _recur(
        self,
        blocks: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        batch_size: int,
        query_len: int,
    ) -> tuple[torch.Tensor, None]:
        block_count = blocks.shape[0]
        shape = (block_count, self.block_size, self.num_heads, self.head_dim)
        raw_gate, beta = self._gates(blocks)
        attention_output = self._kda_fn(
            q.view(shape),
            k.view(shape),
            v.view(shape),
            raw_gate,
            beta,
            self.A_log,
            self.dt_bias,
            self.lower_bound,
            initial_state=initial_state,
        )
        output_gate = self._output_gate(blocks).view(shape)
        attention_output = self.o_norm(attention_output, output_gate)
        attention_output = self.o_proj(attention_output.flatten(-2))
        attention_output = attention_output.reshape(
            batch_size, query_len, self.kda_config.hidden_size
        )
        return attention_output, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        anchor_positions: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Dense/MLA siblings may use the shared cache during SpecForge decode.
        # KDA leaves it untouched: "reset" needs no context at all and "scan"
        # keeps its own running state on the module.
        del position_embeddings, attention_mask, cache_position, kwargs

        blocks, batch_size, query_len = self._independent_blocks(hidden_states)
        if not self.scans_context:
            del target_hidden, past_key_values, anchor_positions
            q = self.q_conv1d(self.q_proj(blocks))
            k = self.k_conv1d(self.k_proj(blocks))
            v = self.v_conv1d(self.v_proj(blocks))
            return self._recur(blocks, q, k, v, None, batch_size, query_len)

        block_count = blocks.shape[0]
        num_blocks = block_count // batch_size
        if past_key_values is not None:
            # Generation: the slice holds the context verified since the last
            # call; every block of this step starts from the advanced state.
            state, tail = self._advance_running_state(target_hidden)
            states = state.unsqueeze(1).expand(-1, num_blocks, -1, -1, -1)
            left_rows = tail.unsqueeze(1).expand(-1, num_blocks, -1, -1)
        else:
            states, left_rows = self._anchor_context(
                target_hidden, anchor_positions, batch_size, num_blocks
            )
        initial_state = states.reshape(
            block_count, self.num_heads, self.head_dim, self.head_dim
        )
        left_rows = left_rows.reshape(block_count, -1, left_rows.shape[-1])
        q = self._convolve_with_left_rows(self.q_conv1d, self.q_proj, left_rows, blocks)
        k = self._convolve_with_left_rows(self.k_conv1d, self.k_proj, left_rows, blocks)
        v = self._convolve_with_left_rows(self.v_conv1d, self.v_proj, left_rows, blocks)
        return self._recur(blocks, q, k, v, initial_state, batch_size, query_len)


__all__ = [
    "KDAConfig",
    "Qwen3DFlashKDAAttention",
    "fla_kda",
    "reference_kda",
    "scan_kda_context_states",
    "validate_dflash_kda_config",
]
