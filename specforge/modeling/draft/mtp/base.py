# coding=utf-8
"""Architecture-independent MTP draft contract.

One registered subclass per model family (e.g. ``qwen3_5.Qwen3_5MTPDraftModel``)
owns the actual trainable network and the native checkpoint key layout for
that family.  This base owns the shared contract the MTP algorithm code relies
on:

- ``embed_tokens`` plus a trainable ``mtp`` module with an ``lm_head``
- ``forward(input_ids, hidden_states, ...)`` -> object exposing ``logits``
- ``forward_hidden(input_ids, hidden_states, ...)`` -> pre-head hidden states
  of shape [batch, seq, hidden], consumed by the chunked training objective
- the native checkpoint key prefix (``mtp.*``) used for native-head init and
  export round-trips
- sharing/freezing the target checkpoint's embedding (and optional lm_head)
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from specforge.modeling.draft.models.specialized.base import NativeDraftAdapter


class MTPDraftModel(nn.Module):
    """Contract shared by all MTP draft architectures."""

    #: Native MTP weights live under this checkpoint key prefix.
    NATIVE_KEY_PREFIX = "mtp."

    #: Target-checkpoint key candidates consulted when merging trained MTP
    #: weights back into a base checkpoint (see ``specforge/export/mtp.py``).
    #: Families override these when the target nests its text decoder
    #: differently (VLM ``model.language_model.*`` vs plain ``model.*``).
    TARGET_EMBED_KEY_CANDIDATES = [
        "model.language_model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "embed_tokens.weight",
    ]
    TARGET_HEAD_KEY_CANDIDATES = [
        "model.language_model.lm_head.weight",
        "model.lm_head.weight",
        "lm_head.weight",
    ]

    checkpoint_adapter: type[NativeDraftAdapter]
    embed_tokens: nn.Embedding
    mtp: nn.Module

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """Run the draft on shifted tokens plus target last hidden states.

        Returns an object exposing ``logits`` of shape [batch, seq, vocab].
        """
        raise NotImplementedError

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the draft backbone and return pre-``lm_head`` hidden states.

        Returns [batch, seq, hidden] without projecting to the vocabulary.
        ``mtp.lm_head`` must apply independently to each token position and
        accept flattened [positions, hidden] inputs. The training wrapper
        optionally checkpoints head/CE chunks; the backbone still processes
        the complete sequence.
        """
        raise NotImplementedError

    def share_target_embeddings(
        self,
        embed_weight: torch.Tensor,
        lm_head_weight: Optional[torch.Tensor] = None,
    ) -> None:
        """Share and freeze the target checkpoint's embedding and lm_head.

        lm_head sharing follows the family config's ``mtp_config.share_lm_head``
        flag (default True) and then requires ``lm_head_weight``.
        """
        self.embed_tokens.weight = embed_weight
        self.embed_tokens.requires_grad_(False)
        mtp_config = getattr(self.config, "mtp_config", None) or {}
        if mtp_config.get("share_lm_head", True):
            if lm_head_weight is None:
                raise ValueError(
                    "share_lm_head is enabled but no target lm_head weight given"
                )
            self.mtp.lm_head.weight = lm_head_weight
            self.mtp.lm_head.requires_grad_(False)

    def native_state_dict(self) -> dict[str, torch.Tensor]:
        """Return weights using the concrete architecture's native layout."""
        return self.checkpoint_adapter.native_state_dict(self.state_dict())

    def required_native_state_keys(self) -> set[str]:
        return self.checkpoint_adapter.required_keys(
            self.state_dict(),
            share_lm_head=(getattr(self.config, "mtp_config", None) or {}).get(
                "share_lm_head", True
            ),
        )

    def allowed_extra_native_state_keys(self) -> set[str]:
        return self.checkpoint_adapter.allowed_extra_keys(
            share_lm_head=(getattr(self.config, "mtp_config", None) or {}).get(
                "share_lm_head", True
            )
        )


__all__ = ["MTPDraftModel"]
