"""DSpark CE/L1 distillation and acceptance-confidence supervision."""

from dataclasses import dataclass

import torch

from specforge.data.loss_mask import has_consecutive_supervised_tokens

from ..base import check_options
from ..configuration import LossDefinition
from ..inputs import prepare_blocks
from ..losses import TokenLoss, decay_weighting
from ..plan import LossPlan, LossTerm
from ..predictions import PredictionBatch
from ..registry import TrainingRequirements, register_protocol
from .dflash import COMMON_MODULES


@register_protocol(
    "dspark",
    training=TrainingRequirements(
        capture_method="dspark",
        minimum_loss_tokens=2,
        loss_mask_filter=has_consecutive_supervised_tokens,
    ),
    requires=("teacher_logits",),
    architectures=("DSparkDraftModel",),
    inputs=("input_ids", "hidden_states", "target_last_hidden_states", "loss_mask"),
    provides=(
        "logits",
        "labels",
        "teacher_logits",
        "chain_positions",
        "acceptance_chain",
    ),
    trains=COMMON_MODULES + ("markov_head", "confidence_head"),
)
@dataclass
class DSparkObjective:
    num_anchors: int = 512
    loss_decay_gamma: float | None = None

    @property
    def loss_terms(self):
        return {
            "token": LossDefinition(self.default_loss),
            "confidence": LossDefinition(
                TokenLoss("bce"), modules=("confidence_head",)
            ),
        }

    default_weighting_options = ("loss_decay_gamma",)

    @property
    def default_weighting(self):
        return decay_weighting(self.loss_decay_gamma)

    default_loss = TokenLoss("ce_l1", {"ce_weight": 0.1, "l1_weight": 0.9})

    def __post_init__(self):
        check_options(
            counts={"num_anchors": self.num_anchors},
            positive={"loss_decay_gamma": self.loss_decay_gamma},
        )

    def __call__(self, runtime, batch, **context):
        model = runtime.draft_model
        blocks = prepare_blocks(
            model.config.draft,
            batch,
            num_anchors=self.num_anchors,
            label_offset=1,
            prefix=True,
        )
        output = model(
            blocks.input_ids,
            batch["hidden_states"],
            blocks.layer_inputs,
            predecessor_ids=blocks.predecessors.flatten(1),
            output_logits=False,
        )

        def compute(calc):
            calc.run((0,))
            confidence = calc.outputs()["confidence_scores"]
            if confidence is None:
                calc.add(1, calc.weights.new_zeros(calc.weights.shape))
                return
            with torch.no_grad():
                distance = calc.statistics.get(0, {}).get("teacher_l1")
                if distance is None:
                    distance = calc.evaluate(
                        "distance",
                        TokenLoss(kind="ce"),
                        calc.outputs()["logits"],
                        calc.labels,
                        teacher=calc.teacher,
                        return_teacher_l1=True,
                        term_index=0,
                    ).statistics["teacher_l1"]
                acceptance = (1 - distance / 2).clamp(0, 1)
            values = calc.evaluate_term(1, confidence, acceptance).values
            calc.add(1, values)

        predictions = runtime.block_predictions(
            output,
            blocks,
            batch,
            projection_inputs={"predecessor_ids": blocks.predecessors},
            compute=compute,
            plan=LossPlan(
                (
                    LossTerm(),
                    LossTerm(
                        "confidence",
                        loss=TokenLoss("bce"),
                    ),
                )
            ),
            requires=frozenset({"teacher_logits"}),
            default_loss=self.default_loss,
        )
        return PredictionBatch((predictions,), reduction="global_microbatch")
