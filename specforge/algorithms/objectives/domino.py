"""Domino schedules base and GRU-corrected token cross entropy."""

from dataclasses import dataclass

from specforge.core.fused_loss import ce_l1_loss

from .base import check_options, reduce_loss
from .dflash import COMMON_MODULES
from .inputs import prepare_blocks
from .registry import register_objective


@register_objective(
    "domino",
    architectures=("DominoDraftModel",),
    inputs=("input_ids", "hidden_states", "loss_mask"),
    outputs=("logits", "hidden_states", "correction_logits"),
    trains=COMMON_MODULES + ("gru_head",),
)
@dataclass
class DominoObjective:
    num_anchors: int = 512
    loss_decay_gamma: float | None = None
    lambda_base_start: float = 1.0
    lambda_base_decay_ratio: float = 0.5

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
            gamma=self.loss_decay_gamma,
            label_offset=int(spec.block.shift_label),
        )
        output = model(
            blocks.input_ids,
            batch["hidden_states"],
            blocks.layer_inputs,
            prev_token_ids=(
                blocks.predecessors if spec.block.shift_label else blocks.labels
            ).flatten(1),
        )
        labels = blocks.labels.flatten(1)
        base = model.lm_head(output.hidden_states)
        if self.lambda_base_start and not total_steps:
            raise ValueError("Domino base-loss schedule requires total_steps")
        warmup = max(1, (total_steps or 1) * self.lambda_base_decay_ratio)
        weight = self.lambda_base_start * max(0, 1 - step / warmup)
        loss = (
            weight * ce_l1_loss(base, labels).cross_entropy
            + (1 - weight) * ce_l1_loss(output.logits, labels).cross_entropy
        )
        return reduce_loss(
            loss, blocks.weights.flatten(1), logits=output.logits, labels=labels
        )
