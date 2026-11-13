"""DSpark CE/L1 distillation and acceptance-confidence supervision."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from specforge.data.loss_mask import has_consecutive_supervised_tokens

from .base import check_options
from .dflash import COMMON_MODULES
from .inputs import prepare_blocks
from .losses import TokenLoss, decay_weighting
from .predictions import AuxiliaryLoss, PredictionBatch
from .registry import TrainingRequirements, register_protocol


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
        "expert_choices",
    ),
    trains=COMMON_MODULES + ("markov_head", "confidence_head"),
)
@dataclass
class DSparkObjective:
    num_anchors: int = 512
    loss_decay_gamma: float | None = None
    bias_update_rate: float = (
        0.0  # expert_bias step per optimizer step; 0 keeps it fixed
    )
    ce_loss_alpha: float = 0.1
    l1_loss_alpha: float = 0.9
    confidence_loss_alpha: float = 1.0

    default_weighting_options = ("loss_decay_gamma",)
    default_loss_options = ("ce_loss_alpha", "l1_loss_alpha")

    @property
    def default_weighting(self):
        return decay_weighting(self.loss_decay_gamma)

    @property
    def frozen_parameters(self):
        return ("confidence_head",) if self.confidence_loss_alpha == 0 else ()

    @property
    def default_loss(self):
        return TokenLoss(
            kind="ce_l1",
            options={"ce_weight": self.ce_loss_alpha, "l1_weight": self.l1_loss_alpha},
        )

    def __post_init__(self):
        check_options(
            counts={"num_anchors": self.num_anchors},
            positive={"loss_decay_gamma": self.loss_decay_gamma},
            nonnegative={
                "ce_loss_alpha": self.ce_loss_alpha,
                "l1_loss_alpha": self.l1_loss_alpha,
                "confidence_loss_alpha": self.confidence_loss_alpha,
                "bias_update_rate": self.bias_update_rate,
            },
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

        def auxiliary(predictions, projected, teacher, weights):
            confidence = projected["confidence_scores"]
            if confidence is None:
                return AuxiliaryLoss(weights.new_zeros(weights.shape))
            with torch.no_grad():
                distance = (
                    (
                        projected["logits"].float().softmax(-1)
                        - teacher.float().softmax(-1)
                    )
                    .abs()
                    .sum(-1)
                )
                acceptance = (1 - distance / 2).clamp(0, 1)
            return AuxiliaryLoss(
                self.confidence_loss_alpha
                * F.binary_cross_entropy_with_logits(
                    confidence, acceptance, reduction="none"
                )
            )

        predictions = runtime.block_predictions(
            output,
            blocks,
            batch,
            projection_inputs={"predecessor_ids": blocks.predecessors},
            auxiliary=auxiliary,
            requires=frozenset({"teacher_logits"}),
            default_loss=self.default_loss,
        )
        return PredictionBatch((predictions,), reduction="global_microbatch")
