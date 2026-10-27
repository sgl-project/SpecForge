"""Shared training plumbing for indexed attention, independent of dense forward."""

from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import Literal

import torch
from transformers import PretrainedConfig

from ..config import AttentionConfig, LayerKernels
from .backends.reference import (
    ExplicitSparseInputs,
    SparseKVSource,
    expanded_kv,
    masked_probabilities,
    reference_sparse_attention,
    selection_mask,
    source_visibility,
)
from .base import Attention, AttentionAuxiliaryLoss, AttentionContext, AttentionOutput
from .selection import IndexedSelection, SparseSelection


@dataclass(frozen=True)
class SparseTrainingConfig:
    backend: Literal["reference", "gathered"]
    indexer_mode: Literal["frozen", "warmup", "joint"] = "joint"
    loss_weight: float = 1.0
    query_chunk_size: int = 32

    def __post_init__(self):
        if self.backend not in ("reference", "gathered"):
            raise ValueError("sparse backend must be explicitly reference or gathered")
        if self.indexer_mode not in ("frozen", "warmup", "joint"):
            raise ValueError("indexer_mode must be frozen, warmup or joint")
        if not 0 <= self.loss_weight < float("inf"):
            raise ValueError("indexer loss_weight must be finite and nonnegative")
        if self.indexer_mode != "frozen" and self.loss_weight == 0:
            raise ValueError("A trainable indexer requires a positive loss_weight")
        if type(self.query_chunk_size) is not int or self.query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be a positive integer")

    @classmethod
    def from_model_config(cls, config: PretrainedConfig) -> "SparseTrainingConfig":
        settings = getattr(config, "sparse_attention", None)
        if not isinstance(settings, dict) or "backend" not in settings:
            raise ValueError("Set sparse_attention.backend explicitly for QSA/DSA")
        return cls(**settings)


def positive_int(config: PretrainedConfig, name: str) -> int:
    value = getattr(config, name, None)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def topk_selection(
    scores: torch.Tensor, allowed: torch.Tensor, budget: int, block_size: int = 1
) -> IndexedSelection:
    width = min(budget, scores.shape[-1])
    # Sorted scores put valid entries first, as required by lengths.
    values, indices = scores.masked_fill(~allowed, float("-inf")).topk(width, -1)
    lengths = torch.isfinite(values).sum(-1)
    return IndexedSelection(indices[:, None], lengths[:, None], block_size=block_size)


def indexer_kl(
    scores: torch.Tensor,
    target: torch.Tensor,
    allowed: torch.Tensor,
    query_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean per-query KL; neither teacher nor indexer inputs receive gradients."""
    target = target.detach().masked_fill(~allowed, 0)
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-30)
    valid = allowed.any(-1) & (target.sum(-1) > 0)
    if query_mask is not None:
        valid = valid & query_mask
    masked = scores.float().masked_fill(~allowed, float("-inf"))
    log_probs = masked.masked_fill(~valid[..., None], 0).log_softmax(-1)
    log_probs = log_probs.masked_fill(~allowed, 0)
    per_query = (target * (target.clamp_min(1e-30).log() - log_probs)).sum(-1)
    return per_query.masked_fill(~valid, 0).sum() / valid.sum().clamp_min(1)


class IndexedAttention(Attention[None]):
    """Stateless training path; concrete classes own projections and indexers."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        kernels: LayerKernels,
        options: AttentionConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.options = options
        self.sparse_options = SparseTrainingConfig.from_model_config(config)
        if config.attention_dropout != 0:
            raise ValueError("Indexed attention currently requires attention_dropout=0")
        self._init_projections(config, kernels)

    def freeze_indexer_if_requested(self):
        if self.sparse_options.indexer_mode == "frozen":
            self.indexer.requires_grad_(False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: AttentionContext,
        state: None = None,
        *,
        output_auxiliary_losses: bool = False,
        auxiliary_query_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        is_causal: bool | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> AttentionOutput[None]:
        if state is not None or use_cache:
            raise NotImplementedError(
                "QSA/DSA currently support stateless training, not KV cache"
            )
        if output_attentions:
            raise NotImplementedError(
                "Indexed attention does not return dense attention weights"
            )
        train_indexer = self.training and self.sparse_options.indexer_mode != "frozen"
        if train_indexer and not output_auxiliary_losses:
            raise ValueError("Trainable indexers require output_auxiliary_losses=True")
        prefix = context.key_value_prefix
        if prefix is None:
            prefix = hidden_states[:, :0]
        q, k, v = self._compute_qkv(hidden_states, prefix, context.position_embeddings)
        key_hidden = torch.cat((prefix, hidden_states), 1)
        kp = context.key_positions
        qp = context.query_positions
        if (kp is None) != (qp is None):
            raise ValueError("Supply both query_positions and key_positions")
        if kp is None:
            kp = torch.arange(k.shape[2], device=k.device)[None]
            qp = kp[:, -q.shape[2] :]
        inputs = ExplicitSparseInputs(
            q, (SparseKVSource("main", k, v, kp),), qp, self.scaling
        )
        options = (
            self.options
            if is_causal is None
            else replace(self.options, is_causal=is_causal)
        )
        allowed, bias = source_visibility(inputs, inputs.sources[0], context, options)
        if not torch.equal(allowed, allowed[:, :1].expand_as(allowed)):
            raise ValueError(
                "Shared indexers require the same visibility across query heads"
            )
        selection, scores, candidates = self.select(
            hidden_states, key_hidden, context, allowed[:, 0], kp
        )
        warmup = self.training and self.sparse_options.indexer_mode == "warmup"
        if warmup:
            indices = torch.arange(k.shape[2], device=k.device).expand(
                q.shape[0], 1, q.shape[2], -1
            )
            selection = SparseSelection(
                (
                    IndexedSelection(
                        indices,
                        torch.full(
                            indices.shape[:-1],
                            k.shape[2],
                            device=k.device,
                            dtype=torch.long,
                        ),
                    ),
                )
            )
        auxiliary = ()
        if train_indexer:
            with torch.no_grad():
                ek, _ = expanded_kv(q, inputs.sources[0])
                teacher_allowed = (
                    allowed
                    if warmup
                    else allowed & selection_mask(inputs, inputs.sources[0], selection)
                )
                parts = []
                for start in range(0, q.shape[2], self.sparse_options.query_chunk_size):
                    end = start + self.sparse_options.query_chunk_size
                    logits = (
                        q[:, :, start:end].float()
                        @ ek.float().transpose(-1, -2)
                        * self.scaling
                    )
                    if bias is not None:
                        logits = logits + bias[:, :, start:end]
                    parts.append(
                        masked_probabilities(
                            logits, teacher_allowed[:, :, start:end]
                        ).mean(1)
                    )
                teacher = torch.cat(parts, dim=1)
            loss = self.indexer_loss(scores, teacher, candidates, auxiliary_query_mask)
            auxiliary = (AttentionAuxiliaryLoss("indexer_kl", loss),)
        if self.sparse_options.backend == "reference":
            result = reference_sparse_attention(inputs, selection, context, options)
        else:
            from .backends.gathered import gathered_sparse_attention

            result = gathered_sparse_attention(
                inputs,
                selection,
                context,
                options,
                query_chunk_size=self.sparse_options.query_chunk_size,
            )
        output = self.o_proj(result.values.flatten(2))
        output = output.masked_fill(~result.valid_queries.any(-1, keepdim=True), 0)
        return AttentionOutput(output, auxiliary_losses=auxiliary)

    def indexer_loss(self, scores, teacher, candidates, query_mask):
        return indexer_kl(scores, teacher, candidates, query_mask)

    @abstractmethod
    def select(
        self,
        hidden: torch.Tensor,
        keys: torch.Tensor,
        context: AttentionContext,
        allowed: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[SparseSelection, torch.Tensor, torch.Tensor]:
        """Return the access plan, indexer scores and KL candidate support."""
        raise NotImplementedError
