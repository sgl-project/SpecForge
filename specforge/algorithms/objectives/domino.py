"""Domino schedules base and GRU-corrected token cross entropy."""

from dataclasses import dataclass

from .base import check_options
from .dflash import COMMON_MODULES
from .inputs import prepare_blocks
from .losses import TokenLoss
from .predictions import LossTerm, PredictionBatch
from .registry import register_protocol


@register_protocol(
    "domino",
    architectures=("DominoDraftModel",),
    inputs=("input_ids", "hidden_states", "loss_mask"),
    provides=(
        "logits",
        "labels",
        "teacher_logits",
        "chain_positions",
        "acceptance_chain",
    ),
    trains=COMMON_MODULES + ("gru_head",),
)
@dataclass
class DominoObjective:
    num_anchors: int = 512
    loss_decay_gamma: float | None = None
    lambda_base_start: float = 1.0
    lambda_base_decay_ratio: float = 0.5

    default_loss = TokenLoss(kind="ce")

    def __post_init__(self):
        check_options(
            counts={"num_anchors": self.num_anchors},
            positive={"loss_decay_gamma": self.loss_decay_gamma},
            probabilities={
                "lambda_base_start": self.lambda_base_start,
                "lambda_base_decay_ratio": self.lambda_base_decay_ratio,
            },
        )

    def __call__(self, runtime, batch, *, step, total_steps, **context):
        model, spec = runtime.draft_model, runtime.draft_model.config.draft.spec_config
        blocks = prepare_blocks(
            model.config.draft,
            batch,
            num_anchors=self.num_anchors,
            label_offset=int(spec.block.shift_label),
        )
        output = model(
            blocks.input_ids,
            batch["hidden_states"],
            blocks.layer_inputs,
            predecessor_ids=blocks.predecessors.flatten(1),
            output_logits=False,
        )
        if self.lambda_base_start and not total_steps:
            raise ValueError("Domino base-loss schedule requires total_steps")
        warmup = max(1, int((total_steps or 1) * self.lambda_base_decay_ratio))
        weight = self.lambda_base_start * max(0, 1 - step / warmup)
        predictions = runtime.block_predictions(
            output,
            blocks,
            batch,
            projection_inputs={"predecessor_ids": blocks.predecessors},
            terms=(LossTerm("base_logits", weight), LossTerm("logits", 1 - weight)),
        )
        return PredictionBatch((predictions,))
