"""Kimi-K3 MLA and KDA backbones for DSpark draft training.

The target model alternates Kimi Delta Attention (KDA) with Multi-Latent
Attention (MLA).  These draft variants keep DSpark's projector, Markov head,
confidence head, and DFlash objective while replacing only the five decoder
layers:

* ``KimiK3DSpark5MLADraftModel`` uses five MLA layers.
* ``KimiK3DSpark4KDA1MLADraftModel`` uses KDA, KDA, MLA, KDA, KDA.

The hybrid intentionally uses MLA as its only target-context injection point.
Each KDA layer resets at every proposal block, so anchors never share recurrent
state and the DFlash training mask cannot be bypassed by a linear-attention
scan.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)
from typing_extensions import Tuple, Unpack

from .dflash import build_target_layer_ids, normalize_draft_head_checkpoint_keys
from .dspark import DSparkDraftModel
from .registry import register_draft


def _rotate_half(x: torch.Tensor, *, interleaved: bool) -> torch.Tensor:
    if interleaved:
        paired = x.float().reshape(*x.shape[:-1], -1, 2)
        first, second = paired.unbind(dim=-1)
        return torch.stack((-second, first), dim=-1).flatten(-2).to(x.dtype)
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    *,
    interleaved: bool,
) -> torch.Tensor:
    freqs = torch.einsum("bs,d->bsd", positions.float(), inv_freq.float())
    angles = (
        torch.repeat_interleave(freqs, 2, dim=-1)
        if interleaved
        else torch.cat((freqs, freqs), dim=-1)
    )
    cos = angles.cos().to(dtype=x.dtype).unsqueeze(1)
    sin = angles.sin().to(dtype=x.dtype).unsqueeze(1)
    return x * cos + _rotate_half(x, interleaved=interleaved) * sin


class KimiK3DraftMLAAttention(nn.Module):
    """K3 MLA in compressed-latent (absorbed) form."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        del layer_idx
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.q_lora_rank = int(config.q_lora_rank)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.v_head_dim = int(config.v_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = self.qk_head_dim**-0.5
        self.rope_interleave = bool(getattr(config, "rope_interleave", True))
        self.use_output_gate = bool(getattr(config, "mla_use_output_gate", False))

        if self.qk_rope_head_dim % 2:
            raise ValueError("qk_rope_head_dim must be even")

        bias = bool(getattr(config, "attention_bias", False))
        eps = float(config.rms_norm_eps)
        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=bias)
        self.q_a_layernorm = Qwen3RMSNorm(self.q_lora_rank, eps=eps)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=bias,
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=bias,
        )
        self.kv_a_layernorm = Qwen3RMSNorm(self.kv_lora_rank, eps=eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=bias,
        )
        if self.use_output_gate:
            self.g_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * self.v_head_dim,
                bias=False,
            )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=bias,
        )

        rope_parameters = getattr(config, "rope_parameters", None) or {}
        rope_theta = float(
            rope_parameters.get(
                "rope_theta",
                getattr(config, "rope_theta", 10000.0),
            )
        )
        inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(0, self.qk_rope_head_dim, 2, dtype=torch.float32)
                / self.qk_rope_head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _project_kv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.kv_a_proj_with_mqa(x)
        kv_latent, k_rope = latent.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        return self.kv_a_layernorm(kv_latent), k_rope

    @torch.compiler.disable
    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        del kwargs
        if past_key_values is not None:
            raise NotImplementedError(
                "Kimi-K3 draft MLA training does not use the HF cache path"
            )

        batch, query_len = hidden_states.shape[:2]
        context_len = target_hidden.shape[1]
        q_lora = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_lora).view(
            batch, query_len, self.num_heads, self.qk_head_dim
        )
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        context_latent, context_rope = self._project_kv(target_hidden)
        noise_latent, noise_rope = self._project_kv(hidden_states)
        kv_latent = torch.cat((context_latent, noise_latent), dim=1)
        k_rope = torch.cat((context_rope, noise_rope), dim=1)

        q_positions = position_ids[:, -query_len:]
        q_rope = _apply_rope(
            q_rope.transpose(1, 2),
            q_positions,
            self.inv_freq,
            interleaved=self.rope_interleave,
        ).transpose(1, 2)
        k_rope = _apply_rope(
            k_rope.unsqueeze(1),
            position_ids[:, : context_len + query_len],
            self.inv_freq,
            interleaved=self.rope_interleave,
        )

        kv_b = self.kv_b_proj.weight.view(
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        w_kc, w_vc = kv_b.split([self.qk_nope_head_dim, self.v_head_dim], dim=1)
        q_absorbed = torch.einsum("bqhd,hdk->bqhk", q_nope, w_kc)
        q_attn = torch.cat((q_absorbed, q_rope), dim=-1).transpose(1, 2)
        k_attn = torch.cat((kv_latent.unsqueeze(1), k_rope), dim=-1)
        v_attn = kv_latent.unsqueeze(1)

        if isinstance(attention_mask, BlockMask):
            latent_out = flex_attention(
                q_attn,
                k_attn,
                v_attn,
                block_mask=attention_mask,
                scale=self.scaling,
                enable_gqa=True,
            )
        else:
            latent_out = F.scaled_dot_product_attention(
                q_attn,
                k_attn,
                v_attn,
                attn_mask=attention_mask,
                dropout_p=0.0,
                scale=self.scaling,
                enable_gqa=True,
            )

        attn_out = torch.einsum("bhqk,hvk->bqhv", latent_out, w_vc)
        attn_out = attn_out.reshape(batch, query_len, -1)
        if self.use_output_gate:
            attn_out = attn_out * torch.sigmoid(self.g_proj(hidden_states))
        return self.o_proj(attn_out), None


class KimiK3DraftMLADecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        hidden_size = int(config.hidden_size)
        eps = float(config.rms_norm_eps)
        self.self_attn = KimiK3DraftMLAAttention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = Qwen3RMSNorm(hidden_size, eps=eps)

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_value,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return (residual + self.mlp(hidden_states),)


class KimiK3ShortConvolution(nn.Module):
    """Causal depthwise convolution with target-compatible parameter layout."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.weight = nn.Parameter(torch.empty(channels, self.kernel_size))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = F.conv1d(
            x,
            self.weight.unsqueeze(1),
            bias=None,
            groups=self.weight.shape[0],
        )
        return F.silu(x.transpose(1, 2))


class KimiK3GatedRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps).to(x.dtype)
        return normalized * self.weight.to(x.dtype) * torch.sigmoid(gate)


def _reference_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: Optional[float],
) -> torch.Tensor:
    """Small differentiable recurrence used by CPU tests, not production."""

    q = F.normalize(q.float(), dim=-1).to(q.dtype)
    k = F.normalize(k.float(), dim=-1).to(k.dtype)
    beta = torch.sigmoid(beta.float()).to(q.dtype)
    gate_input = raw_gate.float() + dt_bias.view(1, 1, *raw_gate.shape[-2:])
    scale = A_log.float().exp().view(1, 1, -1, 1)
    if lower_bound is None:
        log_decay = -scale * F.softplus(gate_input)
    else:
        log_decay = float(lower_bound) * torch.sigmoid(scale * gate_input)
    log_decay = log_decay.to(q.dtype)

    state = q.new_zeros(
        q.shape[0], q.shape[2], q.shape[3], v.shape[3], dtype=torch.float32
    )
    outputs = []
    score_scale = q.shape[-1] ** -0.5
    for step in range(q.shape[1]):
        decay = log_decay[:, step].float().exp().unsqueeze(-1)
        state = state * decay
        key = k[:, step].float()
        value = v[:, step].float()
        prediction = torch.einsum("bhd,bhdv->bhv", key, state)
        delta = (value - prediction) * beta[:, step].float().unsqueeze(-1)
        state = state + torch.einsum("bhd,bhv->bhdv", key, delta)
        outputs.append(
            torch.einsum("bhd,bhdv->bhv", q[:, step].float(), state)
            .mul(score_scale)
            .to(q.dtype)
        )
    return torch.stack(outputs, dim=1)


def _fla_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: Optional[float],
) -> torch.Tensor:
    try:
        from fla.ops.kda import chunk_kda
    except ImportError as exc:
        raise ImportError(
            "Kimi-K3 KDA training requires fla-core==0.5.1; install "
            "SpecForge with the 'kda' extra"
        ) from exc

    output, _ = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=raw_gate,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=lower_bound is not None,
        lower_bound=lower_bound,
    )
    return output


class KimiK3DraftKDAAttention(nn.Module):
    """K3 KDA applied independently to every DSpark proposal block."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        del layer_idx
        linear_config = dict(getattr(config, "linear_attn_config", None) or {})
        self.hidden_size = int(config.hidden_size)
        self.head_dim = int(linear_config["head_dim"])
        self.num_heads = int(linear_config["num_heads"])
        self.block_size = int(config.block_size)
        self.conv_size = int(linear_config["short_conv_kernel_size"])
        self.use_full_rank_gate = bool(linear_config.get("use_full_rank_gate", False))
        self.lower_bound = linear_config.get("gate_lower_bound")
        self.backend = str(linear_config.get("backend", "fla")).lower()
        if self.backend not in {"fla", "reference"}:
            raise ValueError(
                "linear_attn_config.backend must be 'fla' or 'reference', "
                f"got {self.backend!r}"
            )

        projection_size = self.num_heads * self.head_dim
        self.q_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.q_conv1d = KimiK3ShortConvolution(projection_size, self.conv_size)
        self.k_conv1d = KimiK3ShortConvolution(projection_size, self.conv_size)
        self.v_conv1d = KimiK3ShortConvolution(projection_size, self.conv_size)

        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16))
        )
        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(projection_size, dtype=torch.float32))
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        if self.use_full_rank_gate:
            self.g_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        else:
            self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)
        self.o_norm = KimiK3GatedRMSNorm(self.head_dim, eps=float(config.rms_norm_eps))
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

    def _blocks(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        batch, query_len, hidden = x.shape
        if query_len % self.block_size:
            raise ValueError(
                "KDA draft query length must be divisible by block_size; "
                f"got {query_len} and {self.block_size}"
            )
        num_blocks = query_len // self.block_size
        return (
            x.reshape(batch * num_blocks, self.block_size, hidden),
            batch,
            query_len,
        )

    @torch.compiler.disable
    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        del target_hidden, attention_mask, kwargs
        if past_key_values is not None:
            raise NotImplementedError(
                "Kimi-K3 draft KDA training resets state at each proposal block"
            )

        blocks, batch, query_len = self._blocks(hidden_states)
        q = self.q_conv1d(self.q_proj(blocks))
        k = self.k_conv1d(self.k_proj(blocks))
        v = self.v_conv1d(self.v_proj(blocks))
        shape = (*q.shape[:2], self.num_heads, self.head_dim)
        q, k, v = (tensor.view(shape) for tensor in (q, k, v))
        raw_gate = self.f_b_proj(self.f_a_proj(blocks)).view(shape)
        beta = self.b_proj(blocks).float()

        kernel: Callable[..., torch.Tensor]
        kernel = _reference_kda if self.backend == "reference" else _fla_kda
        output = kernel(
            q,
            k,
            v,
            raw_gate,
            beta,
            self.A_log,
            self.dt_bias,
            self.lower_bound,
        )
        if self.use_full_rank_gate:
            output_gate = self.g_proj(blocks).view(shape)
        else:
            output_gate = self.g_b_proj(self.g_a_proj(blocks)).view(shape)
        output = self.o_norm(output, output_gate)
        output = output.reshape(*blocks.shape[:2], -1)
        output = self.o_proj(output).reshape(batch, query_len, self.hidden_size)
        return output, None


class KimiK3DraftKDADecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        hidden_size = int(config.hidden_size)
        eps = float(config.rms_norm_eps)
        self.self_attn = KimiK3DraftKDAAttention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = Qwen3RMSNorm(hidden_size, eps=eps)

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:
        del position_ids
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            past_key_values=past_key_value,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return (residual + self.mlp(hidden_states),)


class _KimiK3DSparkDraftBase(DSparkDraftModel):
    expected_projector_type = "dspark"

    def _initialize_kimi_backbone(self, config, layers: nn.ModuleList) -> None:
        dflash_config = dict(getattr(config, "dflash_config", None) or {})
        projector_type = dflash_config.get("projector_type")
        if projector_type is None:
            dflash_config["projector_type"] = self.expected_projector_type
        elif projector_type != self.expected_projector_type:
            raise ValueError(
                "Kimi-K3 DSpark drafts require dflash_config.projector_type='dspark'"
            )
        config.dflash_config = dflash_config

        # Avoid constructing and immediately discarding the large GQA DSpark
        # backbone.  This is the shared DFlash initialization with custom layers.
        Qwen3PreTrainedModel.__init__(self, config)
        self.config = config
        self.layers = layers
        self.target_layer_ids = dflash_config.get(
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = int(config.block_size)
        self.mask_token_id = dflash_config.get("mask_token_id")
        self.projector_type = dflash_config.get("projector_type")
        self.pure_draft_prefix_len = dflash_config.get("pure_draft_prefix_len", 0)
        self.shift_label = dflash_config.get("shift_label", False)
        self._init_draft_head(config, dflash_config)
        self.register_load_state_dict_pre_hook(normalize_draft_head_checkpoint_keys)
        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if use_cache or past_key_values is not None:
            raise NotImplementedError(
                "Kimi-K3 DSpark training backbones do not use the HF cache path"
            )
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )[0]
        return self.norm(hidden_states)


def _validate_mla_config(config) -> None:
    required = (
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
    )
    missing = [name for name in required if getattr(config, name, None) is None]
    if missing:
        raise ValueError(f"Kimi-K3 draft MLA config is missing: {missing}")
    if not bool(getattr(config, "mla_use_nope", False)):
        raise ValueError("Kimi-K3 draft MLA requires mla_use_nope=true")
    if not bool(getattr(config, "mla_use_output_gate", False)):
        raise ValueError("Kimi-K3 draft MLA requires mla_use_output_gate=true")


@register_draft
class KimiK3DSpark5MLADraftModel(_KimiK3DSparkDraftBase):
    """Five-layer K3 MLA DSpark draft."""

    _no_split_modules = ["KimiK3DraftMLADecoderLayer"]

    def __init__(self, config) -> None:
        _validate_mla_config(config)
        if int(config.num_hidden_layers) != 5:
            raise ValueError("KimiK3DSpark5MLADraftModel requires exactly 5 layers")
        layers = nn.ModuleList(
            KimiK3DraftMLADecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        )
        self._initialize_kimi_backbone(config, layers)


@register_draft
class KimiK3DSpark4KDA1MLADraftModel(_KimiK3DSparkDraftBase):
    """K3 draft with KDA, KDA, MLA, KDA, KDA layers."""

    _no_split_modules = [
        "KimiK3DraftKDADecoderLayer",
        "KimiK3DraftMLADecoderLayer",
    ]

    def __init__(self, config) -> None:
        _validate_mla_config(config)
        if int(config.num_hidden_layers) != 5:
            raise ValueError("KimiK3DSpark4KDA1MLADraftModel requires exactly 5 layers")
        layer_pattern = list(
            getattr(config, "draft_layer_types", None)
            or ["kda", "kda", "mla", "kda", "kda"]
        )
        expected = ["kda", "kda", "mla", "kda", "kda"]
        if layer_pattern != expected:
            raise ValueError(
                "Kimi-K3 4KDA+1MLA layer pattern must be "
                f"{expected}, got {layer_pattern}"
            )
        factories = {
            "kda": KimiK3DraftKDADecoderLayer,
            "mla": KimiK3DraftMLADecoderLayer,
        }
        layers = nn.ModuleList(
            factories[layer_type](config, layer_idx)
            for layer_idx, layer_type in enumerate(layer_pattern)
        )
        self._initialize_kimi_backbone(config, layers)


__all__ = [
    "KimiK3DSpark5MLADraftModel",
    "KimiK3DSpark4KDA1MLADraftModel",
    "KimiK3DraftMLAAttention",
    "KimiK3DraftKDAAttention",
]
