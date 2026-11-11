"""Teacher h[t-1] and token[t] predict token[t+1]."""

from dataclasses import dataclass

import torch

from .base import reduce_loss
from .dflash import COMMON_MODULES
from .inputs import gather, sequence_inputs, valid_tokens
from .registry import register_objective


@register_objective(
    "mtp",
    architectures=("MTPDraftModel",),
    inputs=("input_ids", "target_last_hidden_states", "loss_mask"),
    outputs=("logits",),
    trains=COMMON_MODULES,
)
@dataclass
class MTPObjective:
    def __call__(self, runtime, batch, **context):
        model, spec = runtime.draft_model, runtime.draft_model.config.draft.spec_config
        ids, supervised = batch["input_ids"], valid_tokens(batch)
        positions = torch.arange(1, ids.shape[1], device=ids.device).expand(
            ids.shape[0], -1
        )
        padding = batch.get("attention_mask", torch.ones_like(ids, dtype=torch.bool))[
            :, 1:
        ].bool()
        visibility = (
            (positions[:, :, None] >= positions[:, None, :])
            & padding[:, None, :]
            & padding[:, :, None]
        )
        if "lengths" in batch and batch["lengths"].numel() > ids.shape[0]:
            raise ValueError("MTP requires unpacked sequences")
        output = model(
            ids[:, 1:],
            gather(
                batch["target_last_hidden_states"],
                positions + spec.feature_token_offset,
            ).to(dtype=model.embed_tokens.weight.dtype),
            sequence_inputs(model.config.draft, positions, visibility),
        )
        label_positions = positions + spec.prediction_token_offset
        labels = gather(ids, label_positions)
        weights = (
            gather(supervised, label_positions)
            & (label_positions < ids.shape[1])
            & padding
        ).float()
        return reduce_loss(
            runtime.loss(output.logits, labels, batch, label_positions - 1),
            runtime.weighting(weights, positions - positions[:, :1]),
            logits=output.logits,
            labels=labels,
        )
