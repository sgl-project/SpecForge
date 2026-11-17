# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""EAGLE-4 draft: Eagle3 plus config-gated fusion, window, MoE, and candidate head."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.configuration_llama import LlamaConfig

from specforge.modeling._mask_utils import prepare_decoder_attention_mask
from specforge.modeling.draft.base import Eagle3DraftModel
from specforge.modeling.draft.eagle4_moe import Eagle4Router, update_eagle4_router_bias
from specforge.modeling.draft.eagle4_structure import (
    EAGLE4_ARCH,
    candidate_training_enabled,
    draft_sliding_window,
    efficient_head_export_config,
    parse_efficient_lm_head_config,
    retrieval_positive_count,
)

_CANDIDATE_HEAD_PARAM_NAMES = frozenset(
    {
        "efficient_lm_head.weight",
        "efficient_lm_head_down_proj.weight",
    }
)
from specforge.modeling.draft.llama3_eagle import LlamaDecoderLayer, LlamaRMSNorm
from specforge.modeling.draft.registry import register_draft
from specforge.utils import print_with_rank


def candidate_logits(
    hidden_states: torch.Tensor,
    down_weight: torch.Tensor,
    score_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    num_candidates: int,
) -> torch.Tensor:
    """Low-rank scores pick ``num_candidates`` rows; the dense head rescores them.

    Other vocabulary positions are ``-inf`` so a full-vocab softmax/KL sees only
    the candidate support.
    """

    approx = F.linear(F.linear(hidden_states, down_weight), score_weight)
    k = min(num_candidates, approx.shape[-1])
    indices = approx.topk(k, dim=-1).indices
    selected = lm_head_weight[indices]
    exact = (hidden_states.unsqueeze(-2) * selected).sum(-1)
    approx_selected = approx.gather(-1, indices)
    # Forward is the dense rescore; backward also trains the factor tensors.
    rescored = exact + (approx_selected - approx_selected.detach())
    logits = hidden_states.new_full(
        hidden_states.shape[:-1] + (lm_head_weight.shape[0],),
        torch.finfo(hidden_states.dtype).min,
    )
    return logits.scatter_(-1, indices, rescored.to(logits.dtype))


def cutoff_retrieval_loss(
    dense_logits: torch.Tensor,
    low_rank_scores: torch.Tensor,
    num_candidates: int,
    loss_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pairwise logistic across the ``(C - m + 1)``-th non-positive cutoff.

    ``dense_logits`` (``z``) and ``low_rank_scores`` (``s``) share shape
    ``[..., V]``. Positives are the top-``m`` tokens of ``z``. Temperature-1
    ``softmax(z)`` weights the terms and is stop-gradient. The cutoff index is
    stop-gradient; its score still receives gradient.
    """

    vocab_size = dense_logits.shape[-1]
    positive_count = retrieval_positive_count(num_candidates, vocab_size)
    flat_scores = low_rank_scores.reshape(-1, vocab_size).float()
    with torch.no_grad():
        flat_dense = dense_logits.reshape(-1, vocab_size).float()
        positives = flat_dense.topk(positive_count, dim=-1).indices
        log_normalizer = torch.logsumexp(flat_dense, dim=-1, keepdim=True)
        positive_mass = (flat_dense.gather(-1, positives) - log_normalizer).exp()
    if loss_mask is None:
        token_mask = flat_scores.new_ones(flat_scores.shape[0])
    else:
        token_mask = loss_mask.reshape(flat_scores.shape[0]).to(dtype=flat_scores.dtype)
    denom = token_mask.sum().clamp_min(1.0)

    positive_mask = torch.zeros(
        flat_scores.shape, dtype=torch.bool, device=flat_scores.device
    )
    positive_mask.scatter_(-1, positives, True)
    non_positive = flat_scores.masked_fill(
        positive_mask, torch.finfo(flat_scores.dtype).min
    )
    cutoff_rank = num_candidates - positive_count + 1
    cutoff_choice = non_positive.topk(cutoff_rank, dim=-1)
    cutoff_index = cutoff_choice.indices[:, -1].unsqueeze(-1).detach()
    cutoff = flat_scores.gather(-1, cutoff_index).squeeze(-1)
    positive_scores = flat_scores.gather(-1, positives)
    below_cutoff = (
        (positive_scores <= cutoff.unsqueeze(-1)).to(dtype=flat_scores.dtype).detach()
    )
    per_token = (
        positive_mass
        * below_cutoff
        * F.softplus(cutoff.unsqueeze(-1) - positive_scores)
    ).sum(-1)
    loss = (per_token * token_mask).sum() / denom

    with torch.no_grad():
        retrieved = flat_scores.topk(num_candidates, dim=-1).indices
        hits = (positives.unsqueeze(-1) == retrieved.unsqueeze(1)).any(-1)
        hit_top1 = (hits[:, 0].to(flat_scores.dtype) * token_mask).sum() / denom
        hit_mean = (hits.to(flat_scores.dtype) * token_mask.unsqueeze(-1)).sum() / (
            denom * positive_count
        )
        retrieved_mass = (
            (flat_dense.gather(-1, retrieved) - log_normalizer).exp().sum(-1)
        )
        mass = (retrieved_mass * token_mask).sum() / denom
    metrics = {
        "retrieval_hit_1": hit_top1,
        "retrieval_hit_mean": hit_mean,
        "retrieval_mass": mass,
        "denom": denom,
        "cutoff_index": cutoff_index.squeeze(-1),
        "positives": positives,
    }
    return loss, metrics


@register_draft
class LlamaForCausalLMEagle4(Eagle3DraftModel):
    """Single-layer EAGLE draft with the EAGLE-4 config-gated modules."""

    config_class = LlamaConfig

    def __init__(self, config, quant_config=None, attention_backend="sdpa") -> None:
        super().__init__(config)
        self.config = config
        self.quant_config = quant_config
        architectures = list(getattr(config, "architectures", None) or [])
        if EAGLE4_ARCH not in architectures:
            config.architectures = [EAGLE4_ARCH]

        if config.num_hidden_layers != 1:
            raise ValueError("EAGLE-4 supports one draft layer")
        self.vocab_size = config.vocab_size
        self.draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        if self.draft_vocab_size != self.vocab_size:
            raise ValueError(
                "EAGLE-4 uses a full-vocab head; require draft_vocab_size == vocab_size"
            )
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.midlayer = LlamaDecoderLayer(config, attention_backend=attention_backend)
        self.target_hidden_size = getattr(
            config, "target_hidden_size", config.hidden_size
        )
        self.fc = nn.Linear(
            self.target_hidden_size * 3,
            config.hidden_size,
            bias=False,
        )
        if getattr(config, "fc_norm", False):
            self.fc_norm = nn.ModuleList(
                [
                    LlamaRMSNorm(self.target_hidden_size, eps=config.rms_norm_eps)
                    for _ in range(3)
                ]
            )
        else:
            self.fc_norm = None
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_output = getattr(config, "norm_output", False)
        self.lm_head = nn.Linear(config.hidden_size, self.draft_vocab_size, bias=False)

        candidate_stage = candidate_training_enabled(config)
        if not candidate_stage:
            efficient_head_export_config(config)
        self.efficient_lm_head_config = (
            parse_efficient_lm_head_config(config) if candidate_stage else None
        )
        if self.efficient_lm_head_config is not None:
            if self.draft_vocab_size != self.vocab_size:
                raise ValueError(
                    "efficient_lm_head_config rescores candidates with the "
                    "full-vocab lm_head; train without draft vocab pruning"
                )
            rank = self.efficient_lm_head_config["rank"]
            self.efficient_lm_head_down_proj = nn.Linear(
                config.hidden_size, rank, bias=False
            )
            self.efficient_lm_head = nn.Linear(rank, self.vocab_size, bias=False)
            self.num_candidates = self.efficient_lm_head_config["num_candidates"]
        else:
            self.efficient_lm_head_down_proj = None
            self.efficient_lm_head = None
            self.num_candidates = None

        self.register_buffer("t2d", torch.ones(self.vocab_size, dtype=torch.bool))
        self.register_buffer(
            "d2t", torch.zeros(self.draft_vocab_size, dtype=torch.int64)
        )

        self.post_init()

    def prepare_decoder_attention_mask(
        self,
        attention_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        batch_size: int,
        seq_length: int,
        past_key_values_length: int,
    ) -> torch.Tensor:
        return prepare_decoder_attention_mask(
            attention_mask=attention_mask,
            hidden_states=hidden_states,
            batch_size=batch_size,
            seq_length=seq_length,
            past_key_values_length=past_key_values_length,
            window=draft_sliding_window(self.config),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        expected = self.fc.in_features
        if hidden_states.size(-1) != expected:
            raise ValueError(
                f"Target hidden states size mismatch: {hidden_states.size(-1)} "
                f"!= expected: {expected}"
            )
        if self.fc_norm is not None:
            chunks = hidden_states.chunk(3, dim=-1)
            hidden_states = torch.cat(
                [norm(chunk) for norm, chunk in zip(self.fc_norm, chunks)],
                dim=-1,
            )
        return self.fc(hidden_states)

    def next_depth_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Hidden state the next TTT depth reads.

        ``norm_output`` feeds back the final RMSNorm. Otherwise a post-MLP
        norm is used when present. Logits always come from ``norm(hidden)``.
        """

        if self.norm_output:
            return self.norm(hidden_states)
        post_norm = getattr(self.midlayer, "post_mlp_layernorm", None)
        return hidden_states if post_norm is None else post_norm(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        if self.efficient_lm_head is None:
            return self.lm_head(hidden_states)
        return candidate_logits(
            hidden_states,
            self.efficient_lm_head_down_proj.weight,
            self.efficient_lm_head.weight,
            self.lm_head.weight,
            self.num_candidates,
        )

    def candidate_recall(
        self, hidden_states: torch.Tensor, target_ids: torch.Tensor
    ) -> torch.Tensor:
        """Fraction of ``target_ids`` that fall in the low-rank top-k."""

        if self.efficient_lm_head is None:
            raise RuntimeError("candidate_recall requires efficient_lm_head_config")
        hidden_states = self.norm(hidden_states)
        approx = F.linear(
            F.linear(hidden_states, self.efficient_lm_head_down_proj.weight),
            self.efficient_lm_head.weight,
        )
        k = min(self.num_candidates, approx.shape[-1])
        indices = approx.topk(k, dim=-1).indices
        return (indices == target_ids.unsqueeze(-1)).any(dim=-1).float().mean()

    def freeze_non_candidate_parameters(self) -> None:
        """Train only the low-rank factors; freeze the dense drafter."""

        if self.efficient_lm_head is None:
            raise RuntimeError(
                "freeze_non_candidate_parameters requires efficient_lm_head_config"
            )
        for name, param in self.named_parameters():
            param.requires_grad = name in _CANDIDATE_HEAD_PARAM_NAMES
        # Router load balancing mutates buffers independently of autograd.
        for module in self.modules():
            if isinstance(module, Eagle4Router):
                module.bias_updates_enabled = False
                module.load.zero_()

    def retrieval_loss(
        self,
        hidden_states: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Cutoff retrieval loss on detached draft hidden states.

        Positives are the dense head's top-``m`` tokens, weighted by
        ``softmax(z)`` at temperature 1. ``τ`` is the ``(C - m + 1)``-th
        non-positive low-rank score. The index of ``τ`` is stop-gradient;
        its score still receives gradient.
        """

        if self.efficient_lm_head is None:
            raise RuntimeError("retrieval_loss requires efficient_lm_head_config")
        hidden = self.norm(hidden_states)
        scores = F.linear(
            F.linear(hidden, self.efficient_lm_head_down_proj.weight),
            self.efficient_lm_head.weight,
        )
        dense = F.linear(hidden, self.lm_head.weight)
        return cutoff_retrieval_loss(
            dense,
            scores,
            int(self.num_candidates),
            loss_mask=loss_mask,
        )

    @torch.no_grad()
    def init_efficient_head_from_dense(
        self,
        token_frequency: Optional[torch.Tensor] = None,
        smooth_factor: float = 10.0,
    ) -> None:
        """SVD-initialize the candidate factors from the dense ``lm_head``."""

        if self.efficient_lm_head is None:
            raise RuntimeError(
                "init_efficient_head_from_dense requires efficient_lm_head_config"
            )
        weight = self.lm_head.weight.float()
        if (
            self.efficient_lm_head_config["type"] == "weighted_low_rank"
            and token_frequency is not None
        ):
            freq = token_frequency.to(device=weight.device, dtype=weight.dtype)
            scale = (freq / freq.mean().clamp_min(1e-8) + smooth_factor).sqrt()
            weight = weight * scale.unsqueeze(-1)
        rank = self.efficient_lm_head_down_proj.out_features
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        scores = u[:, :rank] * s[:rank]
        down = vh[:rank]
        self.efficient_lm_head.weight.copy_(
            scores.to(self.efficient_lm_head.weight.dtype)
        )
        self.efficient_lm_head_down_proj.weight.copy_(
            down.to(self.efficient_lm_head_down_proj.weight.dtype)
        )

    def update_moe_router_bias(self) -> Optional[float]:
        return update_eagle4_router_bias(self)

    def backbone(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values=None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        return self.midlayer(
            input_emb=input_embeds,
            hidden_states=hidden_states,
            cache_hidden=cache_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=False,
            use_cache=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ttt_length: int = 1,
    ):
        if ttt_length == 1:
            print_with_rank("using ttt_length 1, no need to cache hidden states")
            cache_hidden = None
        else:
            print_with_rank(f"using ttt_length {ttt_length}, caching hidden states")
            cache_hidden = [[], []]

        batch_size, seq_length, _ = hidden_states.size()
        device = hidden_states.device
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=device
            )
        attention_mask = self.prepare_decoder_attention_mask(
            attention_mask=attention_mask,
            hidden_states=hidden_states,
            batch_size=batch_size,
            seq_length=seq_length,
            past_key_values_length=0,
        )
        hidden_states = self.project_hidden_states(hidden_states)
        hidden_states = self.backbone(
            input_embeds=inputs_embeds,
            hidden_states=hidden_states,
            cache_hidden=cache_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
        return hidden_states


__all__ = ["LlamaForCausalLMEagle4", "candidate_logits", "cutoff_retrieval_loss"]
