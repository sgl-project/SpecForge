"""DeepSeek-V3.2-style lightning indexer with expanded MLA training attention.

Uses floating point scoring (no inference FP8 quantization). See the V3.2
report, section 2.1, and DeepSeek-V3.2-Exp/inference/model.py:Indexer.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .indexed import IndexedAttention, positive_int, topk_selection
from .mla import MLAProjections
from .rotary import apply_mla_rope
from .selection import SparseSelection


class DSAIndexer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = positive_int(config, "index_n_heads")
        self.head_dim = positive_int(config, "index_head_dim")
        self.topk = positive_int(config, "index_topk")
        self.rope_dim = config.qk_rope_head_dim
        if self.rope_dim > self.head_dim:
            raise ValueError("DSA index_head_dim must cover qk_rope_head_dim")
        self.wq_b = nn.Linear(
            positive_int(config, "q_lora_rank"),
            self.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=config.rms_norm_eps)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)

    def forward(self, hidden, query_latent, key_hidden, embeddings):
        q = (
            self.wq_b(query_latent.detach())
            .unflatten(-1, (self.n_heads, self.head_dim))
            .transpose(1, 2)
        )
        k = self.k_norm(self.wk(key_hidden.detach()))[:, None]
        cos, sin = embeddings
        rd = self.rope_dim
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
            (apply_mla_rope(k[..., :rd], cos, sin, interleaved=False), k[..., rd:]), -1
        )
        weights = F.linear(hidden.detach().float(), self.weights_proj.weight.float())
        weights = weights.transpose(1, 2) * (self.head_dim * self.n_heads) ** -0.5
        return (
            (q.float() @ k.float().transpose(-1, -2)).relu() * weights[..., None]
        ).sum(1)


class DSAAttention(MLAProjections, IndexedAttention):
    def __init__(self, config, layer_idx, kernels, options):
        super().__init__(config, layer_idx, kernels, options)
        self.indexer = DSAIndexer(config)
        self.freeze_indexer_if_requested()

    def select(self, hidden, keys, context, allowed, positions):
        # Keep indexer optimization separate from MLA, including the shared Q latent.
        with torch.no_grad():
            latent = self.q_a_layernorm(self.q_a_proj(hidden))
        scores = self.indexer(hidden, latent, keys, context.position_embeddings)
        selected = topk_selection(scores, allowed, self.indexer.topk)
        support = allowed
        if self.sparse_options.indexer_mode != "warmup" or not self.training:
            support = torch.zeros_like(allowed)
            support.scatter_(-1, selected.indices[:, 0], True)
            support &= allowed
        return SparseSelection((selected,)), scores, support
