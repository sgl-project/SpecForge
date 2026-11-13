"""Domino schedules base and GRU-corrected token cross entropy."""

from dataclasses import dataclass

from specforge.data.loss_mask import has_consecutive_supervised_tokens

from ..base import check_options
from ..inputs import prepare_blocks
from ..losses import TokenLoss, decay_weighting
from ..plan import LossPlan, LossTerm
from ..predictions import PredictionBatch
from ..projection import CorrectedProjection
from ..registry import TrainingRequirements, register_protocol
from .dflash import COMMON_MODULES


@register_protocol(
    "domino",
    training=TrainingRequirements(
        capture_method="dflash",
        minimum_loss_tokens=2,
        loss_mask_filter=has_consecutive_supervised_tokens,
    ),
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
    default_weighting_options = ("loss_decay_gamma",)

    @property
    def default_weighting(self):
        return decay_weighting(self.loss_decay_gamma)

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
            label_offset=spec.block.label_offset,
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
            plan=LossPlan(
                terms=(LossTerm("base_logits", weight), LossTerm("logits", 1 - weight)),
                joint_groups=((0, 1),),
            ),
            projection=CorrectedProjection(
                model.project,
                model.project_parts,
                spec.protocol.num_model_slots,
                spec.suffix_start,
            ),
        )
        return PredictionBatch((predictions,))
