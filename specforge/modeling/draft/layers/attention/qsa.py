"""Qwen-style compressed indexer and sparse GQA, with explicit training KL.

Block-aggregated attention distillation is a SpecForge training choice, not a
claim to reproduce Qwen's unpublished indexer training recipe. RoPE tables are
supplied by the caller; multimodal position preparation stays outside this layer.
"""

import torch
from torch import nn

from .gqa import GQAProjections
from .indexed import IndexedAttention, indexer_kl, positive_int, topk_selection
from .rotary import apply_mla_rope
from .selection import SparseSelection


class IndexRMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        y = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + self.eps
        )
        return (y * (1 + self.weight.float())).to(x.dtype)


class QSAIndexer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = positive_int(config, "indexer_n_heads")
        self.dim = positive_int(config, "indexer_head_dim")
        self.ratio = positive_int(config, "indexer_compress_ratio")
        budget = positive_int(config, "indexer_budget")
        if (
            positive_int(config, "indexer_kv_heads") != 1
            or self.ratio < 2
            or budget % self.ratio
        ):
            raise ValueError(
                "QSA requires one index KV head, ratio >= 2 and divisible budget"
            )
        self.block_topk = budget // self.ratio
        self.index_qk_proj = nn.Linear(
            config.hidden_size, (self.heads + 1) * self.dim, bias=False
        )
        self.q_layernorm = IndexRMSNorm(self.dim, config.rms_norm_eps)
        self.k_layernorm = IndexRMSNorm(self.dim, config.rms_norm_eps)

    def forward(self, hidden, keys, embeddings):
        q = self.index_qk_proj(hidden.detach())[..., : self.heads * self.dim]
        q = self.q_layernorm(q.unflatten(-1, (self.heads, self.dim))).transpose(1, 2)
        raw_k = self.index_qk_proj(keys.detach())[..., -self.dim :]
        blocks = raw_k.shape[1] // self.ratio
        k = (
            raw_k[:, : blocks * self.ratio]
            .unflatten(1, (blocks, self.ratio))
            .float()
            .mean(2)
            .to(raw_k.dtype)
        )
        k = self.k_layernorm(k)[:, None]
        cos, sin = embeddings
        rd = cos.shape[-1]
        if rd > self.dim or rd % 2:
            raise ValueError(
                "QSA index head must cover the attention rotary dimensions"
            )
        q = torch.cat(
            (
                apply_mla_rope(
                    q[..., :rd],
                    cos[:, -q.shape[2] :],
                    sin[:, -q.shape[2] :],
                    interleaved=False,
                ),
                q[..., rd:],
            ),
            -1,
        )
        k = torch.cat(
            (
                apply_mla_rope(
                    k[..., :rd],
                    cos[:, : blocks * self.ratio : self.ratio],
                    sin[:, : blocks * self.ratio : self.ratio],
                    interleaved=False,
                ),
                k[..., rd:],
            ),
            -1,
        )
        return (q.float() @ k.float().transpose(-1, -2)).relu().sum(1) * self.dim**-0.5


class QSAAttention(GQAProjections, IndexedAttention):
    def __init__(self, config, layer_idx, kernels, options):
        super().__init__(config, layer_idx, kernels, options)
        self.indexer = QSAIndexer(config)
        self.freeze_indexer_if_requested()

    def select(self, hidden, keys, context, allowed, positions):
        ratio = self.indexer.ratio
        scores = self.indexer(hidden, keys, context.position_embeddings)
        blocks = scores.shape[-1]
        complete = allowed[..., : blocks * ratio].unflatten(-1, (blocks, ratio)).all(-1)
        grouped_pos = positions[:, : blocks * ratio].unflatten(-1, (blocks, ratio))
        contiguous = (grouped_pos[..., 1:] - grouped_pos[..., :-1] == 1).all(-1)
        complete = complete & contiguous[:, None]
        selected = topk_selection(scores, complete, self.indexer.block_topk, ratio)
        # Visible fragments cannot be pooled without reading masked/future keys.
        # Attend those tokens directly; on contiguous causal input this is the tail.
        incomplete = torch.ones_like(allowed)
        incomplete[..., : blocks * ratio] = ~complete.repeat_interleave(ratio, -1)
        tail_allowed = allowed & incomplete
        tail_width = int(tail_allowed.sum(-1).max())
        tails = topk_selection(
            torch.zeros_like(allowed, dtype=torch.float), tail_allowed, tail_width
        )
        return SparseSelection((selected, tails)), scores, complete

    def indexer_loss(self, scores, teacher, candidates, query_mask):
        ratio = self.indexer.ratio
        blocks = scores.shape[-1]
        target = teacher[..., : blocks * ratio].unflatten(-1, (blocks, ratio)).sum(-1)
        return indexer_kl(scores, target, candidates, query_mask)
