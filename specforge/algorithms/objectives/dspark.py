"""DSpark CE/L1 distillation and acceptance-confidence supervision."""

from dataclasses import dataclass

from ..losses.dspark_reduction import reduce_dspark
from .base import COMMON_MODULES, check_options
from .block_options import BlockLossOptions
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
class DSparkObjective(BlockLossOptions):
    ce_loss_alpha: float = 0.1
    l1_loss_alpha: float = 0.9
    confidence_loss_alpha: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        check_options(
            counts={"num_anchors": self.num_anchors},
            positive={"loss_decay_gamma": self.loss_decay_gamma},
            nonnegative={
                "ce_loss_alpha": self.ce_loss_alpha,
                "l1_loss_alpha": self.l1_loss_alpha,
                "confidence_loss_alpha": self.confidence_loss_alpha,
            },
        )

    def __call__(self, runtime, batch, *, collect_detailed_metrics=True, **context):
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
            output_logits=False,
        )
        return reduce_dspark(
            model,
            self,
            runtime.target_head,
            output_hidden=output.hidden_states,
            target_ids=blocks.labels,
            eval_mask=blocks.weights.bool(),
            prev_token_ids=blocks.predecessors,
            safe_label_indices=blocks.label_positions,
            target_last_hidden_states=batch.get("target_last_hidden_states"),
            anchor_positions=blocks.anchors,
            collect_detailed_metrics=collect_detailed_metrics,
        )
