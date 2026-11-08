"""Teacher h[t-1] and token[t] predict token[t+1]."""

from dataclasses import dataclass, replace

import torch

from specforge.core.chunking import checkpointed_chunk_reduce
from specforge.core.fused_loss import ce_l1_loss

from .base import ObjectiveResult, check_options
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
    objective_chunk_size: int = 4096

    def __post_init__(self):
        check_options(zero_counts={"objective_chunk_size": self.objective_chunk_size})

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
        entries = sequence_inputs(model.config.draft, positions, visibility)
        if "position_ids" in batch:
            if batch["position_ids"].shape != ids.shape:
                raise ValueError("position_ids must align with input_ids")
            entries = tuple(
                replace(e, position_ids=batch["position_ids"][:, 1:]) for e in entries
            )
        output = model(
            ids[:, 1:],
            gather(
                batch["target_last_hidden_states"],
                positions + spec.feature_token_offset,
            ),
            entries,
            output_logits=False,
        )
        label_positions = positions + spec.prediction_token_offset
        labels = gather(ids, label_positions)
        weights = (
            gather(supervised, label_positions)
            & (label_positions < ids.shape[1])
            & padding
        ).float()
        hidden = output.hidden_states.flatten(0, 1)

        def chunk_terms(h, labels, weights):
            logits = model.lm_head(h)
            numerator = (ce_l1_loss(logits, labels).cross_entropy * weights).sum()
            correct = ((logits.argmax(-1) == labels) * weights).sum().detach()
            return numerator, correct

        numerator, correct = checkpointed_chunk_reduce(
            chunk_terms,
            hidden,
            labels.flatten(),
            weights.flatten(),
            chunk_size=self.objective_chunk_size,
            checkpoint=0 < self.objective_chunk_size < hidden.shape[0],
        )
        denominator = weights.sum().detach()
        return ObjectiveResult(
            numerator, denominator, {"accuracy": (correct, denominator)}
        )
