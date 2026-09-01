# coding=utf-8
"""DeepSeek-V4 DSpark draft model (official drafter architecture).

Training-side port of the DSpark speculative-decoding head bundled with
``deepseek-ai/DeepSeek-V4-Flash-0731``: three DeepSeek-V4 blocks (MLA-style
low-rank attention, 256-expert MoE, mHC hyper-connections) stored under the
``mtp.*`` checkpoint namespace. Module names deliberately mirror the official
checkpoint tensor names so a trained state dict round-trips into the official
bundled layout without renaming.

Reference implementation: the ``inference/model.py`` shipped with
``deepseek-ai/DeepSeek-V4-Flash-DSpark`` (classes ``DSparkBlock``,
``DSparkAttention``, ``DSparkMarkovHead``, ``DSparkConfidenceHead``) and the
tilelang kernels in its ``kernel.py`` (``hc_split_sinkhorn``, ``act_quant``,
``sparse_attn``).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from .dspark import VanillaMarkovHead
from .moe import MoEGate, SparseMoE
from .registry import register_draft


class DSparkV4DraftConfig(PretrainedConfig):
    model_type = "deepseek_v4_dspark_draft"

    def __init__(
        self,
        vocab_size: int = 129280,
        hidden_size: int = 4096,
        num_hidden_layers: int = 3,
        num_target_layers: int = 43,
        rms_norm_eps: float = 1e-6,
        # attention
        num_attention_heads: int = 64,
        num_key_value_heads: int = 1,
        head_dim: int = 512,
        qk_rope_head_dim: int = 64,
        q_lora_rank: int = 1024,
        o_lora_rank: int = 1024,
        o_groups: int = 8,
        sliding_window: int = 128,
        rope_theta: float = 10000.0,
        # moe
        n_routed_experts: int = 256,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 6,
        moe_intermediate_size: int = 2048,
        scoring_func: str = "sqrtsoftplus",
        routed_scaling_factor: float = 1.5,
        swiglu_limit: float = 10.0,
        # hyper-connections
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        # dspark
        block_size: int = 5,
        initializer_range: float = 0.02,
        dflash_config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_target_layers = num_target_layers
        self.rms_norm_eps = rms_norm_eps
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.sliding_window = sliding_window
        self.rope_theta = rope_theta
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.swiglu_limit = swiglu_limit
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.block_size = block_size
        self.initializer_range = initializer_range
        self.dflash_config = dict(dflash_config or {})
        super().__init__(**kwargs)


class DSparkV4RMSNorm(nn.Module):
    """RMSNorm with fp32 math regardless of parameter storage dtype."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * x).to(dtype)


def _freqs_cis(
    positions: torch.Tensor, rope_dim: int, theta: float
) -> torch.Tensor:
    """Complex rotary frequencies at absolute positions (no YaRN: the DSpark
    window layers disable it and use the base ``rope_theta``)."""
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, rope_dim, 2, device=positions.device, dtype=torch.float32)
            / rope_dim
        )
    )
    freqs = positions.float().unsqueeze(-1) * inv_freq
    return torch.polar(torch.ones_like(freqs), freqs)


def _apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False):
    """Interleaved-pair complex rotary embedding, matching the reference
    ``apply_rotary_emb`` (NOT the HF rotate-half convention). ``freqs_cis``
    must already broadcast against ``x`` with the last dim halved."""
    dtype = x.dtype
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    return torch.view_as_real(xc * freqs_cis).flatten(-2).to(dtype)


def _fake_quant_fp8_ue8m0(x: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    """Straight-through fused quant+dequant to FP8 E4M3 with power-of-2
    (ue8m0) per-group scales; mirrors ``act_quant(..., "ue8m0", inplace=True)``
    used on the non-rope KV dims at serving time (QAT parity)."""
    orig_dtype = x.dtype
    xf = x.float()
    if xf.shape[-1] % group_size != 0:
        group_size = xf.shape[-1]
    grouped = xf.unflatten(-1, (-1, group_size))
    amax = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (grouped / scale).clamp(-448.0, 448.0)
    q = q.to(torch.float8_e4m3fn).float() * scale
    quantized = q.flatten(-2).to(orig_dtype)
    return x + (quantized - x).detach()


def _hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
):
    """fp32 port of the reference ``hc_split_sinkhorn`` tilelang kernel."""
    hc = hc_mult
    scale = hc_scale.float()
    base = hc_base.float()
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * torch.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * scale[2] + base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


class DSparkV4Attention(nn.Module):
    """DSpark window attention: each draft block attends to the last
    ``sliding_window`` projected main-stream positions strictly before its
    anchor plus every token of its own block (bidirectionally), with a
    learned per-head sink logit in the softmax."""

    def __init__(self, config: DSparkV4DraftConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.n_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.window_size = config.sliding_window
        self.rope_theta = config.rope_theta
        self.eps = config.rms_norm_eps
        self.softmax_scale = self.head_dim**-0.5
        self.qscale_ref_compat = False

        dim = config.hidden_size
        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads))
        self.wq_a = nn.Linear(dim, config.q_lora_rank, bias=False)
        self.q_norm = DSparkV4RMSNorm(config.q_lora_rank, self.eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(dim, self.head_dim, bias=False)
        self.kv_norm = DSparkV4RMSNorm(self.head_dim, self.eps)
        self.wo_a = nn.Linear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, dim, bias=False)

    def _kv(self, x: torch.Tensor, freqs_cis: torch.Tensor, kv_fake_quant: bool):
        kv = self.kv_norm(self.wkv(x))
        rope = _apply_rope(kv[..., -self.rope_dim :], freqs_cis)
        nope = kv[..., : -self.rope_dim]
        if kv_fake_quant:
            nope = _fake_quant_fp8_ue8m0(nope)
        return torch.cat([nope, rope], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        main_x: torch.Tensor,
        anchor_positions: torch.Tensor,
        context_freqs_cis: torch.Tensor,
        block_freqs_cis: torch.Tensor,
        kv_fake_quant: bool = True,
    ) -> torch.Tensor:
        """x: [B, N, bs, D] block hidden; main_x: [B, S, D] projected main
        stream; anchor_positions: [B, N]; context_freqs_cis: [S, rope/2];
        block_freqs_cis: [B, N, bs, rope/2]."""
        bsz, n_blocks, block, _ = x.shape
        win = self.window_size

        # Main-stream KV at absolute context positions.
        main_kv = self._kv(main_x, context_freqs_cis.unsqueeze(0), kv_fake_quant)

        # Gather the fixed window [anchor-win, anchor-1] per block.
        window_offsets = torch.arange(-win, 0, device=x.device).view(1, 1, win)
        window_idx = anchor_positions.unsqueeze(-1) + window_offsets  # [B, N, win]
        window_valid = window_idx >= 0
        gather_idx = window_idx.clamp(min=0).reshape(bsz, -1, 1)
        window_kv = torch.gather(
            main_kv, 1, gather_idx.expand(-1, -1, self.head_dim)
        ).view(bsz, n_blocks, win, self.head_dim)

        # Block queries and KV at positions anchor+offset.
        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.unflatten(-1, (self.n_heads, self.head_dim))
        # Weightless per-head rms scale. Rounding differs between the two
        # official implementations; default to the sglang serving math
        # (fp32 mean, scale rounded to bf16 before the multiply). The
        # reference inference/model.py computes the whole thing in bf16 —
        # the parity gate flips this switch to compare against it.
        if self.qscale_ref_compat:
            q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        else:
            q = q * torch.rsqrt(
                q.float().square().mean(-1, keepdim=True) + self.eps
            ).to(q.dtype)
        q_rope = _apply_rope(q[..., -self.rope_dim :], block_freqs_cis.unsqueeze(-2))
        q = torch.cat([q[..., : -self.rope_dim], q_rope], dim=-1)
        block_kv = self._kv(x, block_freqs_cis, kv_fake_quant)

        # Scores over [window (win) | own block (bs)] + sink column, fp32.
        kv_all = torch.cat([window_kv, block_kv], dim=2)  # [B, N, win+bs, hd]
        scores = torch.einsum(
            "bnqhd,bnkd->bnhqk", q.float(), kv_all.float()
        ) * self.softmax_scale
        neg_inf = torch.finfo(scores.dtype).min
        window_mask = window_valid.view(bsz, n_blocks, 1, 1, win)
        scores[..., :win] = scores[..., :win].masked_fill(~window_mask, neg_inf)
        sink = self.attn_sink.float().view(1, 1, self.n_heads, 1, 1)
        sink = sink.expand(bsz, n_blocks, -1, block, 1)
        probs = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
        o = torch.einsum("bnhqk,bnkd->bnqhd", probs, kv_all.float())

        o_rope = _apply_rope(
            o[..., -self.rope_dim :], block_freqs_cis.unsqueeze(-2), inverse=True
        )
        o = torch.cat([o[..., : -self.rope_dim], o_rope], dim=-1).to(x.dtype)

        # Grouped low-rank output projection.
        o = o.reshape(bsz, n_blocks, block, self.n_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bnqgd,grd->bnqgr", o, wo_a.to(o.dtype))
        return self.wo_b(o.flatten(3))



class DSparkV4Stage(nn.Module):
    """One ``mtp.*`` stage: a DeepSeek-V4 block with hyper-connections."""

    def __init__(self, config: DSparkV4DraftConfig, stage_id: int):
        super().__init__()
        self.stage_id = stage_id
        self.norm_eps = config.rms_norm_eps
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        dim = config.hidden_size
        hc_dim = self.hc_mult * dim
        mix_hc = (2 + self.hc_mult) * self.hc_mult

        self.attn = DSparkV4Attention(config)
        self.ffn = SparseMoE(config)
        self.attn_norm = DSparkV4RMSNorm(dim, self.norm_eps)
        self.ffn_norm = DSparkV4RMSNorm(dim, self.norm_eps)
        self.hc_attn_fn = nn.Parameter(torch.zeros(mix_hc, hc_dim))
        self.hc_ffn_fn = nn.Parameter(torch.zeros(mix_hc, hc_dim))
        self.hc_attn_base = nn.Parameter(torch.zeros(mix_hc))
        self.hc_ffn_base = nn.Parameter(torch.zeros(mix_hc))
        self.hc_attn_scale = nn.Parameter(torch.ones(3))
        self.hc_ffn_scale = nn.Parameter(torch.ones(3))

        num_target_features = len(
            (config.dflash_config or {}).get("target_layer_ids", [])
        ) or 3
        self.is_first = stage_id == 0
        self.is_last = stage_id == config.num_hidden_layers - 1
        if self.is_first:
            self.main_proj = nn.Linear(dim * num_target_features, dim, bias=False)
            self.main_norm = DSparkV4RMSNorm(dim, self.norm_eps)
        if self.is_last:
            self.norm = DSparkV4RMSNorm(dim, self.norm_eps)
            self.hc_head_fn = nn.Parameter(torch.zeros(self.hc_mult, hc_dim))
            self.hc_head_base = nn.Parameter(torch.zeros(self.hc_mult))
            self.hc_head_scale = nn.Parameter(torch.ones(1))

    def _hc_mixes(self, x_flat: torch.Tensor, hc_fn: torch.Tensor) -> torch.Tensor:
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        return F.linear(x_flat, hc_fn.float()) * rsqrt

    def hc_pre(self, x, hc_fn, hc_scale, hc_base):
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(2).float()
        mixes = self._hc_mixes(x_flat, hc_fn)
        pre, post, comb = _hc_split_sinkhorn(
            mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
        )
        y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=2)
        return y.to(dtype), post, comb

    def hc_post(self, x, residual, post, comb):
        y = post.unsqueeze(-1) * x.float().unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.float().unsqueeze(-2), dim=2
        )
        return y.to(x.dtype)

    def hc_head(self, x, hc_fn, hc_scale, hc_base):
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(2).float()
        mixes = self._hc_mixes(x_flat, hc_fn)
        pre = torch.sigmoid(mixes * hc_scale.float() + hc_base.float()) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=2)
        return y.to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        main_input: torch.Tensor,
        anchor_positions: torch.Tensor,
        context_freqs_cis: torch.Tensor,
        block_freqs_cis: torch.Tensor,
        block_size: int,
        kv_fake_quant: bool,
    ):
        """Returns ``(x_hc, main_x, final_hidden, prenorm_hidden)``.

        Everything that touches this stage's parameters happens inside this
        forward so FSDP per-stage wrapping sees every use (stage 0 projects
        the captured features; the last stage applies hc_head + final norm).
        """
        main_x = (
            self.main_norm(self.main_proj(main_input))
            if self.is_first
            else main_input
        )
        bsz = x.shape[0]
        residual = x
        h, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        h = self.attn_norm(h)
        h = self.attn(
            h.view(bsz, -1, block_size, h.shape[-1]),
            main_x,
            anchor_positions,
            context_freqs_cis,
            block_freqs_cis,
            kv_fake_quant=kv_fake_quant,
        ).reshape(bsz, -1, h.shape[-1])
        x = self.hc_post(h, residual, post, comb)

        residual = x
        h, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        h = self.ffn_norm(h)
        h = self.ffn(h)
        x = self.hc_post(h, residual, post, comb)
        if not self.is_last:
            return x, main_x, None, None
        prenorm = self.hc_head(
            x, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
        )
        return x, main_x, self.norm(prenorm), prenorm


class DSparkV4ConfidenceHead(nn.Module):
    """Official DSpark confidence head: bias-free projection computed in fp32
    over the pre-final-norm hidden concatenated with the Markov embedding."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(features.float(), self.proj.weight.float()).squeeze(-1)


_ROOT_HEAD_PREFIXES = ("markov_head.", "confidence_head.")


def _rename_root_heads_on_load(
    module, state_dict, prefix, *args
):
    """Accept official ``mtp.<last>.*`` head names for the root-owned heads."""
    del args
    last = module.config.num_hidden_layers - 1
    marker = f"{prefix}mtp.{last}."
    for head in _ROOT_HEAD_PREFIXES:
        for key in [k for k in state_dict if k.startswith(marker + head)]:
            renamed = prefix + key[len(marker):]
            if renamed not in state_dict:
                state_dict[renamed] = state_dict[key]
            state_dict.pop(key)


@register_draft
class DSparkV4DraftModel(PreTrainedModel):
    config_class = DSparkV4DraftConfig
    _no_split_modules = ["DSparkV4Stage"]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config=None,
        output_loading_info=False,
        torch_dtype=None,
        **kwargs,
    ):
        """Load via ``Module.load_state_dict`` instead of HF's per-key loader.

        The checkpoint's per-expert names (``experts.{i}.w*.weight``) are
        state-dict-hook aliases of the stacked expert parameters, not real
        submodules, so transformers' key-path navigation cannot resolve
        them. Checkpoints for this model are local export/bundle dirs.
        """
        import json as _json
        import os as _os

        from safetensors.torch import load_file as _load_safetensors

        path = str(pretrained_model_name_or_path)
        if config is None:
            config = cls.config_class.from_pretrained(path)
        model = cls(config)
        if torch_dtype is not None and torch_dtype != "auto":
            model = model.to(torch_dtype)

        index_file = _os.path.join(path, "model.safetensors.index.json")
        if _os.path.isfile(index_file):
            with open(index_file) as stream:
                shard_names = sorted(set(_json.load(stream)["weight_map"].values()))
        elif _os.path.isfile(_os.path.join(path, "model.safetensors")):
            shard_names = ["model.safetensors"]
        else:
            raise OSError(f"no model.safetensors[.index.json] under {path!r}")
        state: Dict[str, torch.Tensor] = {}
        for name in shard_names:
            state.update(_load_safetensors(_os.path.join(path, name)))

        result = model.load_state_dict(state, strict=False)
        if output_loading_info:
            return model, {
                "missing_keys": list(result.missing_keys),
                "unexpected_keys": list(result.unexpected_keys),
                "mismatched_keys": [],
                "error_msgs": [],
            }
        return model

    def __init__(self, config: DSparkV4DraftConfig) -> None:
        # The trainer stamps training.attention_backend (``native``) onto the
        # config; transformers validates that string even though this model
        # implements its own attention. Coerce to a value HF accepts.
        if getattr(config, "_attn_implementation", None) not in (None, "eager"):
            config._attn_implementation = "eager"
        super().__init__(config)
        self.config = config
        dflash_config = dict(getattr(config, "dflash_config", None) or {})
        config.dflash_config = dflash_config

        self.block_size = int(config.block_size)
        self.target_layer_ids = list(
            dflash_config.get("target_layer_ids", [40, 41, 42])
        )
        self.mask_token_id = dflash_config.get("mask_token_id", None)
        self.projector_type = dflash_config.get("projector_type", "dspark_v4")
        self.kv_fake_quant = bool(dflash_config.get("kv_fake_quant", True))
        # Recompute each stage's forward during backward: trades ~+30% forward
        # compute for dropping most inter-stage activations. Worth it when the
        # allocator is running at the HBM ceiling (fragmentation-driven
        # expandable_segments mapping failures inflate step time ~2x).
        self.stage_gradient_checkpointing = bool(
            dflash_config.get("stage_gradient_checkpointing", False)
        )
        # The training wrapper consults this to build sliding masks for the
        # Qwen3 backbone; the V4 window semantics live inside the model.
        self.sliding_window = None
        self.native_block_attention = True

        self.mtp = nn.ModuleList(
            [
                DSparkV4Stage(config, stage_id)
                for stage_id in range(config.num_hidden_layers)
            ]
        )
        moe_bias_update_rate = float(dflash_config.get("moe_bias_update_rate", 1e-3))
        for stage in self.mtp:
            stage.ffn.bias_update_rate = moe_bias_update_rate

        # The markov/confidence heads are used by the training wrapper OUTSIDE
        # the stage forwards (in the chunked objective), so they live on the
        # root module — FSDP gathers root parameters for the whole root
        # forward, while per-stage units are gathered only during their own
        # forward. State dicts therefore carry them under their root names
        # (markov_head.*, confidence_head.*); the load hook additionally
        # accepts the official checkpoint naming (mtp.<last>.markov_head.*),
        # and the export bundler maps root names back to official names.
        markov_rank = int(dflash_config.get("markov_rank", 256))
        self.markov_head = VanillaMarkovHead(
            vocab_size=config.vocab_size, markov_rank=markov_rank
        )
        self.confidence_head = DSparkV4ConfidenceHead(
            config.hidden_size + markov_rank
        )
        self.register_load_state_dict_pre_hook(_rename_root_heads_on_load)

        self._confidence_hidden: Optional[torch.Tensor] = None
        self.post_init()

    # -- initialization ----------------------------------------------------
    def _init_weights(self, module):
        # transformers 5 loads checkpoints BEFORE running _init_weights and
        # relies on nn.init.* (patched to respect the per-param
        # `_is_hf_initialized` flag) so loaded tensors are not clobbered.
        # Guard the direct writes below the same way.
        std = self.config.initializer_range

        def pending(param) -> bool:
            return not getattr(param, "_is_hf_initialized", False)

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, DSparkV4RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, MoEGate):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if pending(module.bias):
                module.bias.data.zero_()
        elif isinstance(module, DSparkV4Attention):
            nn.init.zeros_(module.attn_sink)
        elif isinstance(module, DSparkV4Stage):
            hc = module.hc_mult
            for fn_name in ("hc_attn_fn", "hc_ffn_fn"):
                nn.init.zeros_(getattr(module, fn_name))
            for scale_name in ("hc_attn_scale", "hc_ffn_scale"):
                nn.init.ones_(getattr(module, scale_name))
            for base_name in ("hc_attn_base", "hc_ffn_base"):
                param = getattr(module, base_name)
                if not pending(param):
                    continue
                base = param.data
                base.zero_()
                # pre ~ uniform 1/hc, post ~ 1, comb ~ identity mixing.
                base[:hc] = -torch.log(torch.tensor(float(hc - 1)))
                comb = base[2 * hc :].view(hc, hc)
                comb.fill_(0.0)
                comb.fill_diagonal_(4.0)
            if hasattr(module, "hc_head_fn"):
                nn.init.zeros_(module.hc_head_fn)
                nn.init.zeros_(module.hc_head_base)
                nn.init.ones_(module.hc_head_scale)
        elif isinstance(module, DSparkV4ConfidenceHead):
            nn.init.zeros_(module.proj.weight)
        elif isinstance(module, VanillaMarkovHead):
            # LoRA-style init: random embedding, zero output projection.
            nn.init.zeros_(module.markov_w2.weight)

    # -- draft-model protocol ---------------------------------------------
    def pop_confidence_hidden(self) -> Optional[torch.Tensor]:
        hidden = self._confidence_hidden
        self._confidence_hidden = None
        return hidden

    def forward(
        self,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[object] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        anchor_positions: Optional[torch.Tensor] = None,
        block_keep_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del position_ids, attention_mask, block_keep_mask
        if anchor_positions is None:
            raise ValueError(
                "DSparkV4DraftModel requires anchor_positions; run it with "
                "training.attention_backend=native"
            )
        block = self.block_size
        seq_len = target_hidden.shape[1]
        device = noise_embedding.device

        rope_dim = self.config.qk_rope_head_dim
        theta = self.config.rope_theta
        context_positions = torch.arange(seq_len, device=device)
        context_freqs = _freqs_cis(context_positions, rope_dim, theta)
        block_positions = anchor_positions.unsqueeze(-1) + torch.arange(
            block, device=device
        ).view(1, 1, block)
        block_freqs = _freqs_cis(block_positions, rope_dim, theta)

        # Hyper-connection expansion of the block-token embeddings. Stage 0
        # projects the captured features into the shared main stream; the last
        # stage applies hc_head and the final norm (all inside stage forwards
        # so per-stage FSDP wrapping is sound).
        if self.training:
            # Apply the PREVIOUS forward's balancing update before any routing
            # this step. It must not run between a forward and its backward:
            # activation-checkpoint recompute would then route with a mutated
            # bias and produce different expert-segment shapes.
            for stage in self.mtp:
                stage.ffn.apply_pending_balance_update()

        x = noise_embedding.unsqueeze(2).repeat(1, 1, self.config.hc_mult, 1)
        main_x = target_hidden
        final_hidden = prenorm_hidden = None
        use_checkpoint = self.stage_gradient_checkpointing and self.training
        for stage in self.mtp:
            if use_checkpoint:
                from torch.utils.checkpoint import checkpoint

                x, main_x, final_hidden, prenorm_hidden = checkpoint(
                    stage,
                    x,
                    main_x,
                    anchor_positions,
                    context_freqs,
                    block_freqs,
                    block,
                    self.kv_fake_quant,
                    use_reentrant=False,
                )
            else:
                x, main_x, final_hidden, prenorm_hidden = stage(
                    x,
                    main_x,
                    anchor_positions,
                    context_freqs,
                    block_freqs,
                    block,
                    self.kv_fake_quant,
                )
        self._confidence_hidden = prenorm_hidden
        return final_hidden

    def apply_logits_head(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
        prev_token_embeddings: Optional[torch.Tensor] = None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        del prev_token_embeddings
        if prev_token_ids is None:
            raise ValueError("DSparkV4DraftModel requires prev_token_ids")
        return self.markov_head.apply_block_logits(
            base_logits,
            token_ids=prev_token_ids,
            hidden_states=hidden_states,
        )

    def predict_confidence(
        self,
        hidden_states: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if prev_token_ids is None:
            raise ValueError("prev_token_ids is required for DSpark confidence")
        prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(
            hidden_states.dtype
        )
        features = torch.cat([hidden_states, prev_embeddings], dim=-1)
        return self.confidence_head(features).float()


__all__ = [
    "DSparkV4DraftConfig",
    "DSparkV4DraftModel",
    "DSparkV4Stage",
    "DSparkV4Attention",
    "DSparkV4ConfidenceHead",
]
