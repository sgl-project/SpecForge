# coding=utf-8
"""Architecture-independent MTP draft contract.

One registered subclass per model family (e.g. ``qwen3_5.Qwen3_5MTPDraftModel``)
owns the actual trainable network and the native checkpoint key layout for
that family.  This base owns the shared contract the MTP algorithm code relies
on:

- ``embed_tokens`` plus a trainable ``mtp`` module with an ``lm_head``
- ``forward(input_ids, hidden_states, ...)`` -> object exposing ``logits``
- the native checkpoint key prefix (``mtp.*``) used for native-head init and
  export round-trips
- sharing/freezing the target checkpoint's embedding (and optional lm_head)
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


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
        """Return the draft weights in the native ``mtp.*`` serving layout."""
        return {
            key: value
            for key, value in self.state_dict().items()
            if key.startswith(self.NATIVE_KEY_PREFIX)
        }

    def required_native_state_keys(self) -> set[str]:
        """Return native keys that must exist for safe target initialization.

        A shared lm_head is deliberately reconstructed from the target model,
        and native Qwen3.5 checkpoints do not need to duplicate it under the
        MTP prefix. Every other native draft tensor must be present; accepting a
        partial prefix match would leave part of the trainable head randomized.
        """
        required = set(self.native_state_dict())
        mtp_config = getattr(self.config, "mtp_config", None) or {}
        if mtp_config.get("share_lm_head", True):
            required.discard(f"{self.NATIVE_KEY_PREFIX}lm_head.weight")
        return required

    def allowed_extra_native_state_keys(self) -> set[str]:
        """Native keys a merged checkpoint may carry that the draft never owns.

        ``export/mtp.merge_mtp_into_base`` backfills shared embeddings into the
        native namespace so serving can instantiate ``mtp.embed_tokens`` (and a
        separate ``mtp.lm_head`` for untied targets).  Re-finetuning a merged
        checkpoint must tolerate those keys instead of rejecting the
        checkpoint as incompatible.
        """
        extra = {f"{self.NATIVE_KEY_PREFIX}embed_tokens.weight"}
        mtp_config = getattr(self.config, "mtp_config", None) or {}
        if mtp_config.get("share_lm_head", True):
            extra.add(f"{self.NATIVE_KEY_PREFIX}lm_head.weight")
        return extra


__all__ = ["MTPDraftModel"]
