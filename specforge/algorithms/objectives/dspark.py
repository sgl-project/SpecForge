"""DSpark CE/L1 distillation and acceptance-confidence supervision."""

from dataclasses import dataclass

import torch.nn.functional as F

from specforge.core.fused_loss import ce_l1_loss

from .base import check_options, reduce_loss
from .dflash import COMMON_MODULES
from .inputs import prepare_blocks
from .registry import register_objective


@register_objective(
    "dspark",
    architectures=("DSparkDraftModel",),
    inputs=("input_ids", "hidden_states", "target_last_hidden_states", "loss_mask"),
    outputs=("logits", "confidence_scores"),
    trains=COMMON_MODULES + ("markov_head", "confidence_head"),
)
@dataclass
class DSparkObjective:
    num_anchors: int = 512
    loss_decay_gamma: float | None = None
    ce_loss_alpha: float = 0.1
    l1_loss_alpha: float = 0.9
    confidence_loss_alpha: float = 1.0

    def __post_init__(self):
        check_options(
            counts={"num_anchors": self.num_anchors},
            positive={"loss_decay_gamma": self.loss_decay_gamma},
            nonnegative={
                "ce_loss_alpha": self.ce_loss_alpha,
                "l1_loss_alpha": self.l1_loss_alpha,
                "confidence_loss_alpha": self.confidence_loss_alpha,
            },
        )

    def __call__(self, runtime, batch, **context):
        model = runtime.draft_model
        blocks = prepare_blocks(
            model.config.draft,
            batch,
            num_anchors=self.num_anchors,
            gamma=self.loss_decay_gamma,
            label_offset=1,
        )
        output = model(
            blocks.input_ids,
            batch["hidden_states"],
            blocks.layer_inputs,
            predecessor_ids=blocks.predecessors.flatten(1),
        )
        labels = blocks.labels.flatten(1)
        teacher = runtime.teacher_logits(batch, blocks.label_positions.flatten(1) - 1)
        terms = ce_l1_loss(output.logits, labels, teacher_logits=teacher)
        loss = (
            self.ce_loss_alpha * terms.cross_entropy
            + self.l1_loss_alpha * terms.teacher_l1
        )
        if output.confidence_scores is not None:
            acceptance = (1 - terms.teacher_l1.detach() / 2).clamp(0, 1)
            confidence = output.confidence_scores.reshape_as(labels)
            loss = (
                loss
                + self.confidence_loss_alpha
                * F.binary_cross_entropy_with_logits(
                    confidence, acceptance, reduction="none"
                )
            )
        return reduce_loss(
            loss, blocks.weights.flatten(1), logits=output.logits, labels=labels
        )
