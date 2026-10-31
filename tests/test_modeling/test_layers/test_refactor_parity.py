"""Pre-refactor DFlash-family attention parity (bit-exact, CPU FP32).

The ``LEGACY`` section is copied verbatim from specforge/modeling/draft/dflash.py
and dflash2.py at e3ab28d6, before attention moved to
``specforge.modeling.draft.layers``. Do not edit it to make a test pass; a
difference means the refactor changed numerics.

One intended difference exists since visibility was unified across backends:
SDPA now zeroes queries with no visible key after ``o_proj`` (as eager always
did), so a dropped draft block no longer emits the projection bias. Training
excludes those rows from the loss, and so do these checks; everything else
stays bit-exact.
"""

import copy
import math
import unittest
from types import MethodType
from typing import Callable, Optional

import torch
from torch import nn
from torch.testing import assert_close
from transformers import DynamicCache, Qwen3Config
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Tuple, Unpack

from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.dflash2 import DFlash2DraftModel, Qwen3DFlash2DecoderLayer
from specforge.modeling.draft.dflash_kernels import (
    DEFAULT_DFLASH_KERNELS,
    DFlashKernels,
)
from specforge.modeling.draft.dflash_mask import (
    create_dflash_sdpa_mask,
    resolve_dflash_is_causal,
)
from specforge.modeling.draft.domino import DominoDraftModel
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.modeling.draft.layers.attention.backends.flex_attention import (
    compile_friendly_flex_attention,
)
from specforge.modeling.draft.layers.attention.backends.flex_attention_backend import (
    flex_attention_backend,
)

SLIDING_ATTENTION = "sliding_attention"

# ---- LEGACY (verbatim from e3ab28d6) -------------------------------------


def _require_bool_config(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"DFlash {field} must be a boolean, got {value!r}")
    return value


def _resolve_mla_rope_interleaved(config: Qwen3Config) -> bool:
    """Rotation convention from the standard MLA ``rope_interleave`` field."""

    return _require_bool_config(
        getattr(config, "rope_interleave", True),
        "config.rope_interleave",
    )


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Rotate consecutive pairs, the DeepSeek-style MLA RoPE convention."""

    paired = x.reshape(*x.shape[:-1], -1, 2)
    first, second = paired.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


def apply_mla_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    interleaved: bool,
) -> torch.Tensor:
    if interleaved:
        half = cos.shape[-1] // 2
        cos = cos[..., :half].repeat_interleave(2, dim=-1)
        sin = sin[..., :half].repeat_interleave(2, dim=-1)
        rotated = _rotate_half_interleaved(x)
    else:
        rotated = rotate_half(x)
    return x * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)


def _prepare_dflash_eager_mask(
    attention_mask: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Convert a boolean allow-mask to eager's additive representation."""

    if attention_mask is None or attention_mask.dtype != torch.bool:
        return attention_mask, None

    valid_queries = attention_mask.any(dim=-1, keepdim=True)
    # A finite minimum keeps eager softmax stable. Fully masked query rows are
    # explicitly zeroed after attention so they cannot average forbidden values.
    additive_mask = torch.zeros_like(attention_mask, dtype=dtype)
    additive_mask.masked_fill_(~attention_mask, torch.finfo(dtype).min)
    return additive_mask, valid_queries


class Qwen3DFlashAttentionBase(nn.Module):
    """Shared scaffold for the DFlash-family attention modes.

    Subclasses own only the projection parameterization: ``_init_projections``
    builds the weights and must define ``scaling``, ``num_key_value_groups``,
    and ``o_proj`` (the shared forward relies on them); ``_compute_qkv``
    returns rotated ``(q, k, v)`` in ``(batch, heads, seq, dim)`` layout with
    keys ordered context-then-draft. Everything the modes must agree on —
    KV-cache updates, backend dispatch, fully-masked-query zeroing, and the
    output projection — lives here so it is maintained in exactly one place.
    """

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        kernels: DFlashKernels,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        if config._attn_implementation == "flex_attention":
            assert (
                config.attention_dropout == 0.0
            ), "DFlash FlexAttention requires attention_dropout=0.0"
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == SLIDING_ATTENTION
            else None
        )
        # Read by mask-free backends (flash_attention_2) and SDPA without a mask.
        self.is_causal = resolve_dflash_is_causal(
            getattr(config, "is_causal", None), config.layer_types[layer_idx]
        )
        self._init_projections(config, kernels)
        for attribute in ("scaling", "num_key_value_groups", "o_proj"):
            assert hasattr(
                self, attribute
            ), f"_init_projections must define {attribute}"

    def _init_projections(self, config: Qwen3Config, kernels: DFlashKernels) -> None:
        raise NotImplementedError

    def _compute_qkv(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

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
        bsz, q_len = hidden_states.shape[:-1]
        q, k, v = self._compute_qkv(hidden_states, target_hidden, position_embeddings)
        if past_key_values is not None:
            cos, sin = position_embeddings
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        valid_queries = None
        if self.config._attn_implementation == "flex_attention":
            kernel_options = dict(kwargs.pop("kernel_options", None) or {})
            backend = flex_attention_backend()
            if backend is not None:
                kernel_options["BACKEND"] = backend

            attn_output = compile_friendly_flex_attention(
                q,
                k,
                v,
                block_mask=attention_mask,
                enable_gqa=True,
                scale=self.scaling,
                kernel_options=kernel_options or None,
            ).transpose(1, 2)
            attn_weights = None
        else:
            attn_fn: Callable = eager_attention_forward
            if self.config._attn_implementation == "eager":
                attention_mask, valid_queries = _prepare_dflash_eager_mask(
                    attention_mask,
                    q.dtype,
                )
            else:
                attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
            attn_output, attn_weights = attn_fn(
                self,
                q,
                k,
                v,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )
            if valid_queries is not None and attn_weights is not None:
                attn_weights = attn_weights.masked_fill(~valid_queries, 0)
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        if valid_queries is not None:
            attn_output = attn_output.masked_fill(
                ~valid_queries.any(dim=1),
                0,
            )
        return attn_output, attn_weights


class Qwen3DFlashAttention(Qwen3DFlashAttentionBase):
    """GQA/MHA projections over the family's context-then-draft KV layout."""

    def _init_projections(self, config: Qwen3Config, kernels: DFlashKernels) -> None:
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = kernels.make_rms_norm(self.head_dim, config.rms_norm_eps)
        self.k_norm = kernels.make_rms_norm(self.head_dim, config.rms_norm_eps)

    def _compute_qkv(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v


class Qwen3DFlashMLAAttention(Qwen3DFlashAttentionBase):
    """Multi-head Latent Attention projections for DFlash-family drafts.

    Standard MLA parameterization: an optional low-rank Q path, a shared
    compressed KV latent, and partial RoPE (interleaved or NeoX, from the
    standard ``rope_interleave`` field). K/V are expanded per head for
    training so the mode runs through the same masks and attention backends
    as :class:`Qwen3DFlashAttention`.
    """

    def _init_projections(self, config: Qwen3Config, kernels: DFlashKernels) -> None:
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_groups = 1
        self.q_lora_rank = (
            None
            if getattr(config, "q_lora_rank", None) is None
            else int(config.q_lora_rank)
        )
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.v_head_dim = int(config.v_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.head_dim = self.qk_head_dim
        self.scaling = self.qk_head_dim**-0.5
        # DeepSeek YaRN applies mscale_all_dim to the full QK logits; the
        # rotary attention factor separately scales only the partial-RoPE slice.
        rope_parameters = config.rope_parameters
        if rope_parameters.get("rope_type", "default") != "default":
            mscale_all_dim = rope_parameters.get("mscale_all_dim", 0)
            if mscale_all_dim:
                factor = rope_parameters["factor"]
                mscale = (
                    1.0
                    if factor <= 1
                    else 0.1 * mscale_all_dim * math.log(factor) + 1.0
                )
                self.scaling *= mscale * mscale
        self.rope_interleaved = _resolve_mla_rope_interleaved(config)

        hidden_size = int(config.hidden_size)
        bias = bool(config.attention_bias)
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
            )
        else:
            self.q_a_proj = nn.Linear(hidden_size, self.q_lora_rank, bias=bias)
            self.q_a_layernorm = kernels.make_rms_norm(
                self.q_lora_rank,
                config.rms_norm_eps,
            )
            self.q_b_proj = nn.Linear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
            )
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=bias,
        )
        self.kv_a_layernorm = kernels.make_rms_norm(
            self.kv_lora_rank,
            config.rms_norm_eps,
        )
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            hidden_size,
            bias=bias,
        )

    def _project_q(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.q_lora_rank is None:
            return self.q_proj(hidden_states)
        return self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

    def _project_kv(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len = hidden_states.shape[:2]
        kv_compressed, k_rope = self.kv_a_proj_with_mqa(hidden_states).split(
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv = self.kv_b_proj(self.kv_a_layernorm(kv_compressed)).view(
            bsz,
            seq_len,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope, value = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        return k_nope, k_rope, value

    def _compute_qkv(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, q_len = hidden_states.shape[:2]
        query = self._project_q(hidden_states).view(
            bsz,
            q_len,
            self.num_heads,
            self.qk_head_dim,
        )
        q_nope, q_rope = query.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )
        ctx_k_nope, ctx_k_rope, ctx_value = self._project_kv(target_hidden)
        noise_k_nope, noise_k_rope, noise_value = self._project_kv(hidden_states)
        k_nope = torch.cat((ctx_k_nope, noise_k_nope), dim=1)
        k_rope = torch.cat((ctx_k_rope, noise_k_rope), dim=1)
        v = torch.cat((ctx_value, noise_value), dim=1)
        # The model-level rotary carries qk_rope_head_dim; queries take the
        # trailing positions exactly like apply_rotary_pos_emb.
        cos, sin = position_embeddings
        q_rope = apply_mla_rope(
            q_rope.transpose(1, 2),
            cos[:, -q_len:],
            sin[:, -q_len:],
            interleaved=self.rope_interleaved,
        )
        k_rope = apply_mla_rope(
            k_rope.unsqueeze(1),
            cos,
            sin,
            interleaved=self.rope_interleaved,
        ).expand(-1, self.num_heads, -1, -1)
        q = torch.cat((q_nope.transpose(1, 2), q_rope), dim=-1)
        k = torch.cat((k_nope.transpose(1, 2), k_rope), dim=-1)
        return q, k, v.transpose(1, 2)


_DFLASH_ATTENTION_CLASSES = {
    "gqa": Qwen3DFlashAttention,
    "mha": Qwen3DFlashAttention,
    "mla": Qwen3DFlashMLAAttention,
}


def _legacy_dflash_decoder_forward(
    self,
    target_hidden: Optional[torch.Tensor] = None,
    hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,  # necessary, but kept here for BC
    **kwargs: Unpack[FlashAttentionKwargs],
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(
        hidden_states=hidden_states,
        target_hidden=target_hidden,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )[0]
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states


def _legacy_dflash2_decoder_forward(
    self,
    target_hidden: Optional[torch.Tensor] = None,
    hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, attention_kernel = self.attention_conv.prepare(hidden_states)
    hidden_states = self.self_attn(
        hidden_states=hidden_states,
        target_hidden=target_hidden,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )[0]
    hidden_states = self.attention_conv.finish(
        hidden_states,
        attention_kernel,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states, mlp_kernel = self.mlp_conv.prepare(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = self.mlp_conv.finish(hidden_states, mlp_kernel)
    return residual + hidden_states


# ---- END LEGACY ------------------------------------------------------------


def _config(
    architecture="DFlashDraftModel",
    *,
    attention_mode="gqa",
    implementation="sdpa",
    layer_types=("full_attention", "full_attention"),
    is_causal=None,
):
    dflash_config = {
        "attention_mode": attention_mode,
        "target_layer_ids": [1, 2],
        "mask_token_id": 31,
        "block_size": 4,
    }
    if architecture == "DFlash2DraftModel":
        dflash_config.update(
            conv_group_size=4,
            conv_kernel_size=2,
            selector_rank=4,
            selector_top_k=3,
        )
    elif architecture == "DSparkDraftModel":
        dflash_config.update(
            projector_type="dspark", markov_rank=4, enable_confidence_head=True
        )
    elif architecture == "DominoDraftModel":
        dflash_config.update(projector_type="domino", emb_dim=6, gru_hidden_dim=5)
    config = Qwen3Config(
        architectures=[architecture],
        block_size=4,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=4 if attention_mode == "mla" else 2,
        num_hidden_layers=len(layer_types),
        num_target_layers=4,
        head_dim=4,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        max_position_embeddings=64,
        vocab_size=32,
        layer_types=list(layer_types),
        sliding_window=3 if "sliding_attention" in layer_types else None,
        use_sliding_window="sliding_attention" in layer_types,
        attention_bias=True,
        attention_dropout=0.0,
        dflash_config=dflash_config,
    )
    if is_causal is not None:
        config.is_causal = is_causal
    config._attn_implementation = implementation
    return config


_MODELS = {
    "DFlashDraftModel": DFlashDraftModel,
    "DFlash2DraftModel": DFlash2DraftModel,
    "DSparkDraftModel": DSparkDraftModel,
    "DominoDraftModel": DominoDraftModel,
}


def _with_legacy_layers(model):
    legacy = copy.deepcopy(model)
    attention_cls = _DFLASH_ATTENTION_CLASSES[model.attention_mode]
    for layer_idx, layer in enumerate(legacy.layers):
        attention = attention_cls(
            config=model.config, layer_idx=layer_idx, kernels=DEFAULT_DFLASH_KERNELS
        )
        attention.load_state_dict(layer.self_attn.state_dict())
        # Assigning an existing child keeps its position in the parameter order.
        layer.self_attn = attention
        forward = (
            _legacy_dflash2_decoder_forward
            if isinstance(layer, Qwen3DFlash2DecoderLayer)
            else _legacy_dflash_decoder_forward
        )
        layer.forward = MethodType(forward, layer)
    return legacy


def _training_inputs(model):
    torch.manual_seed(3)
    seq_len, block_size = 10, model.block_size
    anchors = torch.tensor([[2, 6], [4, 0]])
    keep = torch.tensor([[True, True], [True, False]])
    positions = torch.cat(
        (
            torch.arange(seq_len).expand(2, -1),
            (anchors[..., None] + torch.arange(block_size)).flatten(1),
        ),
        dim=1,
    )
    masks = {}
    for layer_type in set(model.layer_types):
        masks[layer_type] = create_dflash_sdpa_mask(
            anchors,
            keep,
            seq_len,
            block_size,
            "cpu",
            sliding_window=(
                model.sliding_window if layer_type == "sliding_attention" else None
            ),
            is_causal=getattr(model.config, "is_causal", None),
        )
    inputs = dict(
        position_ids=positions,
        attention_mask=masks,
        noise_embedding=torch.randn(2, anchors.shape[1] * block_size, 16),
        target_hidden=torch.randn(2, seq_len, model.fc.in_features),
    )
    return inputs, keep.repeat_interleave(block_size, dim=1)


class PreRefactorParityTest(unittest.TestCase):
    CASES = [
        dict(),
        dict(implementation="eager"),
        dict(layer_types=("sliding_attention", "full_attention")),
        dict(
            implementation="eager", layer_types=("sliding_attention", "full_attention")
        ),
        dict(is_causal=True),
        dict(attention_mode="mla"),
        dict(attention_mode="mla", implementation="eager"),
        dict(architecture="DFlash2DraftModel"),
        dict(architecture="DFlash2DraftModel", implementation="eager"),
        dict(architecture="DSparkDraftModel", attention_mode="mla"),
        dict(architecture="DominoDraftModel"),
    ]

    def test_outputs_gradients_and_optimizer_steps_match(self):
        for case in self.CASES:
            with self.subTest(**case):
                config = _config(**case)
                torch.manual_seed(0)
                model = _MODELS[config.architectures[0]](config).train()
                legacy = _with_legacy_layers(model)
                self.assertEqual(
                    [name for name, _ in model.named_parameters()],
                    [name for name, _ in legacy.named_parameters()],
                )
                optimizers = [
                    torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.1)
                    for m in (model, legacy)
                ]
                inputs, valid_rows = _training_inputs(model)
                for _ in range(2):
                    outputs = []
                    for m, optimizer in zip((model, legacy), optimizers):
                        optimizer.zero_grad()
                        output = m(**inputs)
                        output.square().mean(-1)[valid_rows].mean().backward()
                        optimizer.step()
                        outputs.append(output[valid_rows])
                    assert_close(*outputs, atol=0, rtol=0)
                    for (name, p), (_, q) in zip(
                        model.named_parameters(), legacy.named_parameters()
                    ):
                        assert_close(p, q, atol=0, rtol=0, msg=name)

    def test_cached_block_decoding_matches(self):
        for case in (
            dict(),
            dict(implementation="eager"),
            dict(layer_types=("sliding_attention", "full_attention")),
            dict(attention_mode="mla"),
        ):
            with self.subTest(**case):
                config = _config(**case)
                torch.manual_seed(0)
                model = DFlashDraftModel(config).eval()
                legacy = _with_legacy_layers(model)
                caches = (DynamicCache(), DynamicCache())
                with torch.no_grad():
                    for start, context in ((0, 6), (6, 3)):
                        noise = torch.randn(1, model.block_size, 16)
                        target = torch.randn(1, context, model.fc.in_features)
                        cached = caches[0].get_seq_length()
                        positions = torch.arange(
                            cached, cached + context + model.block_size
                        )[None]
                        outputs = [
                            m(
                                position_ids=positions,
                                noise_embedding=noise,
                                target_hidden=target,
                                past_key_values=cache,
                                use_cache=True,
                            )
                            for m, cache in zip((model, legacy), caches)
                        ]
                        assert_close(*outputs, atol=0, rtol=0)
                        for cache in caches:
                            cache.crop(cache.get_seq_length() - model.block_size)


if __name__ == "__main__":
    unittest.main()
