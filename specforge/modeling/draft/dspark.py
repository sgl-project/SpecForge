# coding=utf-8
"""DSpark draft model entry point and Markov heads."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .dflash import DFlashDraftModel, resolve_dflash_attention_mode
from .dflash_kernels import DFlashKernels
from .heads.confidence import AcceptRatePredictor
from .heads.markov import GatedMarkovHead, RNNMarkovHead, VanillaMarkovHead, _sample
from .registry import register_draft


def build_markov_head(config, dspark_config: dict) -> Optional[nn.Module]:
    markov_rank = int(dspark_config.get("markov_rank", 0) or 0)
    if markov_rank < 0:
        raise ValueError(f"markov_rank must be >= 0, got {markov_rank}")
    if markov_rank == 0:
        return None

    markov_head_type = str(dspark_config.get("markov_head_type", "vanilla")).lower()
    if markov_head_type == "vanilla":
        return VanillaMarkovHead(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
        )
    if markov_head_type == "gated":
        return GatedMarkovHead(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
            hidden_size=config.hidden_size,
        )
    if markov_head_type == "rnn":
        return RNNMarkovHead(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
            hidden_size=config.hidden_size,
        )
    raise ValueError(f"Unsupported markov_head_type: {markov_head_type!r}")


@register_draft
class DSparkDraftModel(DFlashDraftModel):
    """DFlash backbone with DSpark Markov/confidence heads."""

    expected_projector_type = "dspark"

    def __init__(self, config, dflash_kernels: Optional[DFlashKernels] = None) -> None:
        dflash_config = dict(getattr(config, "dflash_config", None) or {})
        attention_mode = resolve_dflash_attention_mode(config)
        num_heads = int(config.num_attention_heads)
        num_kv_heads = int(config.num_key_value_heads)
        # MLA carries its own head geometry; the query/KV head-count policy
        # below only constrains the GQA/MHA projections.
        if attention_mode != "mla" and num_heads % num_kv_heads:
            raise ValueError(
                "DSpark requires num_key_value_heads to divide "
                f"num_attention_heads, got {num_kv_heads} and {num_heads}"
            )
        if attention_mode == "gqa" and num_kv_heads >= num_heads:
            raise ValueError(
                "DSpark defaults to GQA and requires num_key_value_heads < "
                "num_attention_heads; set dflash_config.attention_mode='mha' "
                "to opt into equal query/KV head counts"
            )
        # 'mha' head-count consistency is enforced by the shared
        # validate_dflash_attention_config at model init.
        dflash_config["attention_mode"] = attention_mode
        projector_type = dflash_config.get("projector_type")
        if projector_type is None:
            dflash_config["projector_type"] = self.expected_projector_type
            config.dflash_config = dflash_config
        elif projector_type != self.expected_projector_type:
            raise ValueError(
                "DSparkDraftModel requires " "dflash_config.projector_type='dspark'."
            )
        config.dflash_config = dflash_config
        super().__init__(config, dflash_kernels=dflash_kernels)

    def _sample_draft_tokens(
        self,
        target: nn.Module,
        draft_hidden: torch.Tensor,
        block_output_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """Generate a block with the same Markov correction used in training."""

        proposal_hidden = draft_hidden[:, -self.block_size + 1 :, :]
        base_logits = target.lm_head(proposal_hidden)
        if self.markov_head is None:
            return _sample(base_logits)
        sampled_tokens, _ = self.markov_head.sample_block_tokens(
            base_logits,
            first_prev_token_ids=block_output_ids[:, 0],
            hidden_states=proposal_hidden,
        )
        return sampled_tokens

    def _init_draft_head(self, config, dflash_config: dict) -> None:
        self.markov_head = build_markov_head(config, dflash_config)
        confidence_alpha = float(dflash_config.get("confidence_head_alpha", 0.0) or 0.0)
        self.enable_confidence_head = bool(
            dflash_config.get("enable_confidence_head", confidence_alpha > 0.0)
        )
        self.confidence_head_with_markov = bool(
            dflash_config.get("confidence_head_with_markov", False)
        )
        if self.confidence_head_with_markov and self.markov_head is None:
            raise ValueError(
                "confidence_head_with_markov=True requires markov_rank > 0"
            )

        self.confidence_head = None
        if self.enable_confidence_head:
            input_dim = config.hidden_size
            if self.confidence_head_with_markov:
                input_dim += self.markov_head.markov_rank
            self.confidence_head = AcceptRatePredictor(input_dim=input_dim)

    def apply_logits_head(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
        prev_token_embeddings: Optional[torch.Tensor] = None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        del prev_token_embeddings
        if self.markov_head is None:
            return base_logits
        if prev_token_ids is None:
            raise ValueError("DSparkDraftModel requires prev_token_ids")
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
        if self.confidence_head is None:
            return None
        if self.confidence_head_with_markov:
            assert self.markov_head is not None
            if prev_token_ids is None:
                raise ValueError("prev_token_ids is required for Markov confidence")
            prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(
                hidden_states.dtype
            )
            hidden_states = torch.cat([hidden_states, prev_embeddings], dim=-1)
        return self.confidence_head(hidden_states).float()


__all__ = [
    "AcceptRatePredictor",
    "DSparkDraftModel",
    "GatedMarkovHead",
    "RNNMarkovHead",
    "VanillaMarkovHead",
    "build_markov_head",
]
