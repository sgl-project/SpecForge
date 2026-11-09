"""DFlash explicit-token cross entropy over anchored proposals."""

from dataclasses import dataclass

from ..losses.dflash_reduction import reduce_dflash
from .base import COMMON_MODULES
from .block_options import DFlashLossOptions
from .inputs import prepare_blocks
from .registry import register_objective


@register_objective(
    "dflash",
    architectures=("DFlashDraftModel",),
    inputs=("input_ids", "hidden_states", "loss_mask"),
    outputs=("logits",),
    trains=COMMON_MODULES,
)
@dataclass
class DFlashObjective(DFlashLossOptions):
    def __call__(self, runtime, batch, *, collect_detailed_metrics=True, **context):
        model = runtime.draft_model
        blocks = prepare_blocks(model.config.draft, batch, num_anchors=self.num_anchors)
        output = model(
            blocks.input_ids,
            batch["hidden_states"],
            blocks.layer_inputs,
            output_logits=False,
        )
        return reduce_dflash(
            model,
            self,
            runtime.target_head,
            output.hidden_states,
            blocks,
            target_last_hidden_states=batch.get("target_last_hidden_states"),
            collect_detailed_metrics=collect_detailed_metrics,
        )
