"""Teacher h[t-1] and token[t] predict token[t+1]."""

from dataclasses import dataclass, replace

import torch

from .base import check_options
from .dflash import COMMON_MODULES
from .inputs import gather, sequence_inputs, valid_tokens
from .losses import TokenLoss
from .predictions import PredictionBatch
from .projection import LinearProjection
from .registry import TrainingRequirements, register_protocol


def supervision_weights(batch, spec):
    ids = batch["input_ids"]
    positions = torch.arange(1, ids.shape[1], device=ids.device).expand_as(ids[:, 1:])
    labels = positions + spec.prediction_token_offset
    padding = batch.get("attention_mask", torch.ones_like(ids, dtype=torch.bool))
    return (
        gather(valid_tokens(batch), labels)
        & (labels < ids.shape[1])
        & padding[:, 1:].bool()
    ).float()


def prepare_supervised_rows(batch, config):
    """Select projection rows before the strategy transfers the batch to CUDA."""
    weights = supervision_weights(batch, config.spec_config)
    return {"mtp_supervised_rows": weights.flatten().nonzero().flatten()}


@register_protocol(
    "mtp",
    training=TrainingRequirements(
        capture_method="dflash",
        explicit_initialization=True,
        attention_backends=frozenset({"eager", "sdpa"}),
        host_inputs=prepare_supervised_rows,
    ),
    architectures=("MTPDraftModel",),
    inputs=("input_ids", "target_last_hidden_states", "loss_mask"),
    provides=("logits", "labels", "teacher_logits"),
    trains=COMMON_MODULES,
)
@dataclass
class MTPObjective:
    default_loss = TokenLoss(kind="ce")
    objective_chunk_size: int = 4096

    def __post_init__(self):
        check_options(zero_counts={"objective_chunk_size": self.objective_chunk_size})

    def __call__(self, runtime, batch, **context):
        model, spec = runtime.draft_model, runtime.draft_model.config.draft.spec_config
        ids = batch["input_ids"]
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
            ).to(dtype=model.embed_tokens.weight.dtype),
            entries,
            output_logits=False,
        )
        label_positions = positions + spec.prediction_token_offset
        labels = gather(ids, label_positions)
        weights = supervision_weights(batch, spec)
        predictions = runtime.predictions(
            output.hidden_states,
            batch,
            label_positions - 1,
            labels=labels,
            projection=LinearProjection(model.project, model.lm_head),
            mask=weights,
            # one proposal per position: no depth for position weights to act on
            positions=torch.zeros_like(positions),
            chain_ids=torch.arange(positions.numel(), device=ids.device).reshape(-1, 1),
        )
        rows = batch.get("mtp_supervised_rows")
        if rows is None:
            rows = prepare_supervised_rows(batch, model.config.draft)[
                "mtp_supervised_rows"
            ]
        # An empty selection still projects an empty tensor, preserving the graph
        # through every trained parameter on ranks with no supervised tokens.
        predictions = predictions.select(rows)
        return PredictionBatch((predictions,), chunk_size=self.objective_chunk_size)
