# coding=utf-8
# Portions derived from DeepSeek's MIT-licensed V4 reference implementation.
#
# MIT License
#
# Copyright (c) 2023 DeepSeek
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Trainable DeepSeek-V4-0731 DSpark, with the official ``mtp.*`` key layout.

Reference: deepseek-ai/DeepSeek-V4-Flash-0731, inference/model.py (MIT).
The three stages retain MQA, grouped low-rank output, attention sinks,
unscaled interleaved RoPE (including output de-rotation), clipped SwiGLU
MoE and mHC. Like the official draft, they have no KV compressor/indexer.

Training gathers each block's fixed context window before attention. This
avoids an O(sequence_length * num_anchors) mask and preserves gradients to
the attention sink without inference-only kernels or detached softmax LSEs.
Weights/activations are trained in floating point; FP8 KV simulation is not
applied. Quantized official weights must first be dequantized by the importer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import initialization as hf_init
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import PreTrainedModel

from .dspark import VanillaMarkovHead
from .registry import register_draft


class DeepseekV4DSparkConfig(PretrainedConfig):
    model_type = "deepseek_v4_dspark"

    def __init__(
        self,
        vocab_size=129280,
        hidden_size=4096,
        num_hidden_layers=3,
        num_target_layers=43,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        q_lora_rank=1024,
        qk_rope_head_dim=64,
        o_groups=8,
        o_lora_rank=1024,
        rope_theta=10000.0,
        sliding_window=128,
        n_routed_experts=256,
        n_shared_experts=1,
        num_experts_per_tok=6,
        moe_intermediate_size=2048,
        routed_scaling_factor=1.5,
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        swiglu_limit=10.0,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        block_size=10,
        attention_chunk_blocks=16,
        use_gradient_checkpointing=False,
        max_position_embeddings=1048576,
        dflash_config=None,
        **kwargs,
    ):
        values = dict(locals())
        for name in ("self", "kwargs", "__class__", "dflash_config"):
            values.pop(name, None)
        for name, value in values.items():
            setattr(self, name, value)
        self.dflash_config = {
            "projector_type": "dspark",
            "target_layer_ids": [40, 41, 42],
            "mask_token_id": 128799,
            "markov_rank": 256,
            "markov_head_type": "vanilla",
            "enable_confidence_head": True,
            "confidence_head_with_markov": True,
            **(dflash_config or {}),
        }
        kwargs.pop("tie_word_embeddings", None)
        super().__init__(tie_word_embeddings=False, **kwargs)
        self._validate()

    def _validate(self):
        if not isinstance(self.use_gradient_checkpointing, bool):
            raise ValueError("use_gradient_checkpointing must be a boolean")
        for name in (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_target_layers",
            "num_attention_heads",
            "head_dim",
            "block_size",
            "max_position_embeddings",
            "q_lora_rank",
            "o_groups",
            "o_lora_rank",
            "sliding_window",
            "n_routed_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "hc_mult",
            "hc_sinkhorn_iters",
            "attention_chunk_blocks",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.num_attention_heads % self.o_groups:
            raise ValueError("o_groups must divide num_attention_heads")
        if self.num_key_value_heads != 1:
            raise ValueError("DeepSeek V4 DSpark requires one shared KV head")
        if not 0 < self.qk_rope_head_dim <= self.head_dim or self.qk_rope_head_dim % 2:
            raise ValueError("qk_rope_head_dim must be positive, even and <= head_dim")
        if self.block_size < 2 or self.num_experts_per_tok > self.n_routed_experts:
            raise ValueError("block_size must be >= 2 and top-k cannot exceed experts")
        if self.n_shared_experts != 1:
            raise ValueError("the official V4 draft has exactly one shared expert")
        if self.scoring_func != "sqrtsoftplus" or not self.norm_topk_prob:
            raise ValueError("V4-0731 uses normalized sqrtsoftplus routing")
        if any(
            value <= 0
            for value in (
                self.rope_theta,
                self.rms_norm_eps,
                self.hc_eps,
                self.routed_scaling_factor,
            )
        ):
            raise ValueError(
                "RoPE theta, normalization epsilons and routing scale must be positive"
            )
        dc = self.dflash_config
        if dc.get("projector_type") != "dspark":
            raise ValueError("DeepSeek V4 requires dflash_config.projector_type=dspark")
        ids = dc.get("target_layer_ids")
        if (
            not ids
            or len(set(ids)) != len(ids)
            or any(
                not isinstance(i, int) or not 0 <= i < self.num_target_layers
                for i in ids
            )
            or ids != sorted(ids)
        ):
            raise ValueError(
                "target_layer_ids must be ordered, unique target layer IDs"
            )
        if dc.get("markov_head_type") != "vanilla" or dc.get("markov_rank", 0) <= 0:
            raise ValueError("the official V4 draft requires a vanilla Markov head")
        if not dc.get("enable_confidence_head") or not dc.get(
            "confidence_head_with_markov"
        ):
            raise ValueError("the official V4 confidence head uses Markov features")
        if not 0 <= dc.get("mask_token_id", -1) < self.vocab_size:
            raise ValueError("mask_token_id must be inside the target vocabulary")
        if getattr(self, "quantization_config", None):
            raise ValueError("import/dequantize the official weights before training")


class DeepseekV4RMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        y = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + self.eps
        )
        return (y * self.weight.float()).to(x.dtype)


def apply_v4_rope(x, positions, rope_dim, theta, *, inverse=False):
    """Interleaved rotation on the LAST channels; pure SWA never uses YaRN."""
    freq = theta ** (-torch.arange(0, rope_dim, 2, device=x.device).float() / rope_dim)
    angles = positions.float().unsqueeze(-1) * freq
    # x is [batch, sequence, heads, dim], including a singleton KV head.
    cos, sin = angles.cos().unsqueeze(-2), angles.sin().unsqueeze(-2)
    if inverse:
        sin = -sin
    tail = x[..., -rope_dim:].float().unflatten(-1, (-1, 2))
    a, b = tail.unbind(-1)
    rotated = torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2)
    return torch.cat((x[..., :-rope_dim], rotated.to(x.dtype)), -1)


def hc_split_sinkhorn(mixes, scale, base, mult, iters, eps):
    """Differentiable equivalent of the official mHC Sinkhorn kernel."""
    pre = torch.sigmoid(mixes[..., :mult] * scale[0] + base[:mult]) + eps
    post = 2 * torch.sigmoid(
        mixes[..., mult : 2 * mult] * scale[1] + base[mult : 2 * mult]
    )
    comb = (mixes[..., 2 * mult :] * scale[2] + base[2 * mult :]).unflatten(
        -1, (mult, mult)
    )
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def hc_pre(x, fn, scale, base, config):
    flat = x.flatten(-2).float()
    mixes = F.linear(flat, fn.float()) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + config.rms_norm_eps
    )
    pre, post, comb = hc_split_sinkhorn(
        mixes,
        scale.float(),
        base.float(),
        config.hc_mult,
        config.hc_sinkhorn_iters,
        config.hc_eps,
    )
    return (pre.unsqueeze(-1) * x.float()).sum(-2).to(x.dtype), post, comb


def hc_post(x, residual, post, comb):
    # comb[source, destination]; transposing this silently changes the model.
    return (
        post.unsqueeze(-1) * x.float().unsqueeze(-2)
        + torch.einsum("...ij,...id->...jd", comb, residual.float())
    ).to(x.dtype)


@dataclass(frozen=True)
class DeepseekV4BlockLayout:
    anchors: torch.Tensor
    keep: torch.Tensor
    context_length: int
    block_size: int


class DeepseekV4DSparkAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        h, d = config.hidden_size, config.head_dim
        self.attn_sink = nn.Parameter(torch.zeros(config.num_attention_heads))
        self.wq_a = nn.Linear(h, config.q_lora_rank, bias=False)
        self.q_norm = DeepseekV4RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank, config.num_attention_heads * d, bias=False
        )
        self.wkv = nn.Linear(h, d, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(d, config.rms_norm_eps)
        self.wo_a = nn.Linear(
            config.num_attention_heads * d // config.o_groups,
            config.o_groups * config.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(config.o_groups * config.o_lora_rank, h, bias=False)

    def forward(self, x, target_hidden, position_ids, layout):
        c = self.config
        b, qlen, _ = x.shape
        n, bs = layout.anchors.shape[1], layout.block_size
        context_len = layout.context_length
        qpos = position_ids[:, context_len:]
        q = self.wq_b(self.q_norm(self.wq_a(x))).view(
            b, qlen, c.num_attention_heads, c.head_dim
        )
        q = (
            q.float()
            * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + c.rms_norm_eps)
        ).to(x.dtype)
        q = apply_v4_rope(q, qpos, c.qk_rope_head_dim, c.rope_theta)
        kv = self.kv_norm(self.wkv(torch.cat((target_hidden, x), dim=1))).unsqueeze(-2)
        kv = apply_v4_rope(kv, position_ids, c.qk_rope_head_dim, c.rope_theta).squeeze(
            -2
        )
        draft_kv = kv[:, context_len:].view(b, n, bs, c.head_dim)
        q = q.view(b, n, bs, c.num_attention_heads, c.head_dim)
        offsets = torch.arange(-c.sliding_window, 0, device=x.device)
        outputs = []
        for start in range(0, n, c.attention_chunk_blocks):
            end = min(n, start + c.attention_chunk_blocks)
            indices = layout.anchors[:, start:end, None] + offsets
            valid = (indices >= 0) & (indices < context_len)
            batch = torch.arange(b, device=x.device)[:, None, None]
            context = kv[batch, indices.clamp(0, context_len - 1)]
            keys = torch.cat((context, draft_kv[:, start:end]), dim=2)
            allowed = torch.cat(
                (
                    valid,
                    torch.ones(b, end - start, bs, dtype=torch.bool, device=x.device),
                ),
                dim=-1,
            )
            queries = q[:, start:end].transpose(2, 3)
            # The same rotated tensor supplies K and V, including RoPE channels.
            logits = torch.einsum("bnhqd,bnkd->bnhqk", queries.float(), keys.float())
            logits = (logits * c.head_dim**-0.5).masked_fill(
                ~allowed[:, :, None, None], -torch.inf
            )
            sink = self.attn_sink.float()[None, None, :, None, None].expand(
                *logits.shape[:-1], 1
            )
            probs = torch.cat((logits, sink), dim=-1).softmax(-1)[..., :-1]
            output = torch.einsum("bnhqk,bnkd->bnhqd", probs, keys.float())
            output = output.transpose(2, 3).to(x.dtype)
            output = output * layout.keep[:, start:end, None, None, None]
            outputs.append(output)
        out = torch.cat(outputs, dim=1).reshape(
            b, qlen, c.num_attention_heads, c.head_dim
        )
        out = apply_v4_rope(out, qpos, c.qk_rope_head_dim, c.rope_theta, inverse=True)
        out = out.reshape(b, qlen, c.o_groups, -1)
        weight = self.wo_a.weight.view(c.o_groups, c.o_lora_rank, -1)
        out = torch.einsum("bsgd,grd->bsgr", out.float(), weight.float()).to(x.dtype)
        return self.wo_b(out.flatten(2))


class DeepseekV4MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size)
        )
        # Selection-only bias is fixed: top-k has no derivative to this tensor.
        self.register_buffer(
            "bias", torch.zeros(config.n_routed_experts, dtype=torch.float32)
        )

    def _apply(self, fn, recurse=True):
        # Module.to(dtype=...) must not round the fixed expert-selection bias.
        bias = self.bias
        super()._apply(fn, recurse=recurse)
        self.bias = bias.to(device=self.weight.device)
        return self

    def forward(self, x):
        scores = F.softplus(F.linear(x.float(), self.weight.float())).sqrt()
        indices = (
            (scores + self.bias.float())
            .topk(self.config.num_experts_per_tok, dim=-1)
            .indices
        )
        weights = scores.gather(-1, indices)
        weights = weights / weights.sum(-1, keepdim=True)
        return indices, weights * self.config.routed_scaling_factor


def _v4_swiglu(gate, up, weights, limit, output_dtype):
    """Keep routing multiplication in FP32 before the BF16 down projection."""
    gate, up = gate.float(), up.float()
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    activated = F.silu(gate) * up
    if weights is not None:
        activated = activated * weights
    return activated.to(output_dtype)


class DeepseekV4Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        h, i = config.hidden_size, config.moe_intermediate_size
        self.w1 = nn.Linear(h, i, bias=False)
        self.w2 = nn.Linear(i, h, bias=False)
        self.w3 = nn.Linear(h, i, bias=False)
        self.limit = config.swiglu_limit

    def forward(self, x, weights=None):
        gate, up = self.w1(x), self.w3(x)
        return self.w2(_v4_swiglu(gate, up, weights, self.limit, x.dtype))


class DeepseekV4MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = DeepseekV4MoEGate(config)
        self.experts = nn.ModuleList(
            DeepseekV4Expert(config) for _ in range(config.n_routed_experts)
        )
        self.shared_experts = DeepseekV4Expert(config)

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        indices, weights = self.gate(flat)
        order = indices.flatten().argsort()
        token_index = torch.div(
            order, self.config.num_experts_per_tok, rounding_mode="floor"
        )
        counts = torch.bincount(indices.flatten(), minlength=len(self.experts)).tolist()
        sorted_x = flat.index_select(0, token_index)
        sorted_w = weights.flatten().index_select(0, order).unsqueeze(-1)
        outputs, start = [], 0
        # FSDP requires identical collective order even for locally unused experts.
        for expert, count in zip(self.experts, counts):
            outputs.append(
                expert(sorted_x[start : start + count], sorted_w[start : start + count])
            )
            start += count
        routed = torch.zeros_like(flat, dtype=torch.float32).index_add(
            0, token_index, torch.cat(outputs).float()
        )
        return (routed + self.shared_experts(flat).float()).to(x.dtype).view(shape)


class DeepseekV4MarkovHead(VanillaMarkovHead):
    def project_bias(self, latent_states):
        return F.linear(latent_states.float(), self.markov_w2.weight.float())


class DeepseekV4ConfidenceHead(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.proj = nn.Linear(size, 1, bias=False)

    def forward(self, features):
        return F.linear(features.float(), self.proj.weight.float()).squeeze(-1)


class DeepseekV4DSparkStage(GradientCheckpointingLayer):
    def __init__(self, config, stage_id):
        super().__init__()
        self.config = config
        self.attn = DeepseekV4DSparkAttention(config)
        self.ffn = DeepseekV4MoE(config)
        self.attn_norm = DeepseekV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = DeepseekV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        mult, h = config.hc_mult, config.hidden_size
        for branch in ("attn", "ffn"):
            self.register_parameter(
                f"hc_{branch}_fn",
                nn.Parameter(torch.empty(mult * (mult + 2), mult * h)),
            )
            self.register_parameter(
                f"hc_{branch}_base", nn.Parameter(torch.zeros(mult * (mult + 2)))
            )
            self.register_parameter(f"hc_{branch}_scale", nn.Parameter(torch.ones(3)))
        if stage_id == 0:
            self.main_proj = nn.Linear(
                h * len(config.dflash_config["target_layer_ids"]), h, bias=False
            )
            self.main_norm = DeepseekV4RMSNorm(h, config.rms_norm_eps)
        if stage_id == config.num_hidden_layers - 1:
            self.norm = DeepseekV4RMSNorm(h, config.rms_norm_eps)
            self.hc_head_fn = nn.Parameter(torch.empty(mult, mult * h))
            self.hc_head_base = nn.Parameter(torch.zeros(mult))
            self.hc_head_scale = nn.Parameter(torch.ones(1))
            rank = config.dflash_config["markov_rank"]
            self.markov_head = DeepseekV4MarkovHead(
                vocab_size=config.vocab_size, markov_rank=rank
            )
            self.confidence_head = DeepseekV4ConfidenceHead(h + rank)

    def forward(self, hidden_states, target_hidden, position_ids, layout):
        x, post, comb = hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.config,
        )
        x = self.attn(self.attn_norm(x), target_hidden, position_ids, layout)
        hidden_states = hc_post(x, hidden_states, post, comb)
        x, post, comb = hc_pre(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.config,
        )
        return hc_post(self.ffn(self.ffn_norm(x)), hidden_states, post, comb)


@register_draft
class DeepseekV4DSparkDraftModel(PreTrainedModel):
    config_class = DeepseekV4DSparkConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    # Dense stage parameters, projection and output heads remain in the root
    # FSDP unit: the latter are also called by the objective outside stage.forward.
    _no_split_modules = ["DeepseekV4Expert"]
    _supports_sdpa = True

    def __init__(self, config):
        super().__init__(config)
        self.target_layer_ids = list(config.dflash_config["target_layer_ids"])
        self.block_size = config.block_size
        self.mask_token_id = config.dflash_config["mask_token_id"]
        self.projector_type = "dspark"
        self.sliding_window = config.sliding_window
        self.layer_types = ["sliding_attention"] * config.num_hidden_layers
        self.enable_confidence_head = True
        self.confidence_head_with_markov = True
        self.mtp = nn.ModuleList(
            DeepseekV4DSparkStage(config, i) for i in range(config.num_hidden_layers)
        )
        self.post_init()
        if config.use_gradient_checkpointing:
            self.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    @property
    def markov_head(self):
        return self.mtp[-1].markov_head

    @property
    def confidence_head(self):
        return self.mtp[-1].confidence_head

    def _init_weights(self, module):
        if isinstance(module, DeepseekV4DSparkStage):
            for name, param in module.named_parameters(recurse=False):
                if name.endswith("_fn"):
                    hf_init.normal_(param, mean=0, std=self.config.initializer_range)
                elif name.endswith("_scale"):
                    hf_init.ones_(param)
                else:
                    hf_init.zeros_(param)
        elif isinstance(module, DeepseekV4MoEGate):
            hf_init.normal_(module.weight, mean=0, std=self.config.initializer_range)
            hf_init.zeros_(module.bias)
        elif isinstance(module, DeepseekV4DSparkAttention):
            hf_init.zeros_(module.attn_sink)
        else:
            super()._init_weights(module)

    def build_training_attention_mask(
        self, *, anchor_positions, block_keep_mask, S, block_size, **kwargs
    ):
        if block_size != self.block_size:
            raise ValueError(
                "training block_size disagrees with the DeepSeek draft config"
            )
        return DeepseekV4BlockLayout(anchor_positions, block_keep_mask, S, block_size)

    def prepare_lm_head_hidden(self, hidden):
        # Confidence uses the PRE-norm HC head output; LM logits use POST-norm.
        return self.mtp[-1].norm(hidden)

    def apply_logits_head(
        self,
        base_logits,
        *,
        prev_token_ids=None,
        prev_token_embeddings=None,
        hidden_states,
    ):
        if prev_token_ids is None:
            raise ValueError("DeepSeek V4 Markov logits require prev_token_ids")
        return self.markov_head.apply_block_logits(
            base_logits, token_ids=prev_token_ids, hidden_states=hidden_states
        )

    def predict_confidence(self, hidden_states, *, prev_token_ids=None):
        if prev_token_ids is None:
            raise ValueError("DeepSeek V4 confidence requires prev_token_ids")
        previous = self.markov_head.get_prev_embeddings(prev_token_ids).to(
            hidden_states.dtype
        )
        return self.confidence_head(
            torch.cat((hidden_states, previous), dim=-1)
        ).float()

    def forward(
        self,
        position_ids=None,
        attention_mask=None,
        noise_embedding=None,
        target_hidden=None,
        **kwargs,
    ):
        if noise_embedding is None or target_hidden is None or position_ids is None:
            raise ValueError(
                "DeepSeek DSpark requires noise_embedding, target_hidden and position_ids"
            )
        b, qlen, h = noise_embedding.shape
        s = target_hidden.shape[1]
        if (
            target_hidden.shape != (b, s, h * len(self.target_layer_ids))
            or h != self.config.hidden_size
        ):
            raise ValueError(
                "target_hidden must concatenate the configured mHC-mean teacher features"
            )
        if position_ids.shape != (b, s + qlen) or qlen % self.block_size or s < 1:
            raise ValueError(
                "position_ids must contain context followed by complete draft blocks"
            )
        if isinstance(attention_mask, dict):
            attention_mask = attention_mask.get("sliding_attention")
        if not isinstance(attention_mask, DeepseekV4BlockLayout):
            raise ValueError(
                "DeepSeek DSpark requires its own fixed-window block layout"
            )
        if (
            attention_mask.anchors.shape != (b, qlen // self.block_size)
            or attention_mask.keep.shape != attention_mask.anchors.shape
            or attention_mask.block_size != self.block_size
            or attention_mask.context_length != s
        ):
            raise ValueError("attention layout shape does not match hidden states")
        main = self.mtp[0].main_norm(self.mtp[0].main_proj(target_hidden))
        x = noise_embedding.unsqueeze(-2).expand(-1, -1, self.config.hc_mult, -1)
        for stage in self.mtp:
            x = stage(x, main, position_ids, attention_mask)
        last = self.mtp[-1]
        flat = x.flatten(-2).float()
        mixes = F.linear(flat, last.hc_head_fn.float()) * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + self.config.rms_norm_eps
        )
        pre = (
            torch.sigmoid(
                mixes * last.hc_head_scale.float() + last.hc_head_base.float()
            )
            + self.config.hc_eps
        )
        return (pre.unsqueeze(-1) * x.float()).sum(-2).to(x.dtype)
