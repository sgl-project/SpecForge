"""Domino schedules base and GRU-corrected token cross entropy."""

from dataclasses import dataclass

from ..losses.domino_reduction import reduce_domino
from .base import COMMON_MODULES, check_options
from .block_options import BlockLossOptions
from .inputs import prepare_blocks
from .registry import register_objective


@register_objective(
    "domino",
    architectures=("DominoDraftModel",),
    inputs=("input_ids", "hidden_states", "loss_mask"),
    outputs=("logits", "base_logits"),
    trains=COMMON_MODULES + ("gru_head",),
)
@dataclass
class DominoObjective(BlockLossOptions):
    lambda_base_start: float = 1.0
    lambda_base_decay_ratio: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        check_options(
            counts={"num_anchors": self.num_anchors},
            positive={"loss_decay_gamma": self.loss_decay_gamma},
            probabilities={
                "lambda_base_start": self.lambda_base_start,
                "lambda_base_decay_ratio": self.lambda_base_decay_ratio,
            },
        )

    def __call__(
        self,
        runtime,
        batch,
        *,
        step,
        total_steps,
        collect_detailed_metrics=True,
        **context,
    ):
        if self.lambda_base_start and not total_steps:
            raise ValueError("Domino base-loss schedule requires total_steps")
        warmup = max(1, int((total_steps or 1) * self.lambda_base_decay_ratio))
        weight = self.lambda_base_start * max(0, 1 - step / warmup)
        model = runtime.draft_model
        offset = int(model.config.draft.spec_config.block.shift_label)
        blocks = prepare_blocks(
            model.config.draft, batch, num_anchors=self.num_anchors, label_offset=offset
        )
        output = model(
            blocks.input_ids,
            batch["hidden_states"],
            blocks.layer_inputs,
            output_logits=False,
        )
        return reduce_domino(
            model,
            self,
            output.hidden_states,
            blocks,
            lambda_base=weight,
            collect_detailed_metrics=collect_detailed_metrics,
        )
