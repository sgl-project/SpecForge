"""DFlash2 unary CE plus covered-candidate selector CE."""

from dataclasses import dataclass

from ..losses.dflash_reduction import reduce_dflash
from .base import COMMON_MODULES, check_options
from .block_options import DFlashLossOptions
from .inputs import prepare_blocks
from .registry import register_objective


@register_objective(
    "dflash2",
    architectures=("DFlash2DraftModel",),
    inputs=("input_ids", "hidden_states", "loss_mask"),
    outputs=("logits", "hidden_states"),
    trains=COMMON_MODULES + ("candidate_selector",),
)
@dataclass
class DFlash2Objective(DFlashLossOptions):
    selector_loss_alpha: float = 1.0
    selector_warmup_steps: int = 0
    selector_ramp_steps: int = 0
    selector_warmup_ratio: float = 0.0
    selector_ramp_ratio: float = 0.0
    selector_stop_gradient: bool = False

    def __post_init__(self):
        super().__post_init__()
        check_options(
            counts={"num_anchors": self.num_anchors},
            positive={"loss_decay_gamma": self.loss_decay_gamma},
            nonnegative={"selector_loss_alpha": self.selector_loss_alpha},
            probabilities={
                "selector_warmup_ratio": self.selector_warmup_ratio,
                "selector_ramp_ratio": self.selector_ramp_ratio,
            },
            zero_counts={
                "selector_warmup_steps": self.selector_warmup_steps,
                "selector_ramp_steps": self.selector_ramp_steps,
            },
        )
        if type(self.selector_stop_gradient) is not bool:
            raise ValueError("selector_stop_gradient must be boolean")
        if (self.selector_warmup_steps or self.selector_ramp_steps) and (
            self.selector_warmup_ratio or self.selector_ramp_ratio
        ):
            raise ValueError("choose selector schedule steps or ratios, not both")

    @property
    def frozen_modules(self):
        return ("candidate_selector",) if self.selector_loss_alpha == 0 else ()

    def __call__(
        self, runtime, batch, *, step, collect_detailed_metrics=True, **context
    ):
        scale = (
            min(
                max(
                    (step - self.selector_warmup_steps)
                    / max(self.selector_ramp_steps, 1),
                    0,
                ),
                1,
            )
            if self.selector_ramp_steps
            else float(step >= self.selector_warmup_steps)
        )
        if self.selector_warmup_ratio or self.selector_ramp_ratio:
            total_steps = context.get("total_steps")
            if not total_steps:
                raise ValueError("selector ratio schedule requires total_steps")
            warmup = int(total_steps * self.selector_warmup_ratio)
            ramp = int(total_steps * self.selector_ramp_ratio)
            scale = (
                0.0
                if step < warmup
                else (min((step - warmup + 1) / ramp, 1.0) if ramp else 1.0)
            )
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
            selector_loss_alpha=self.selector_loss_alpha * scale,
            collect_detailed_metrics=collect_detailed_metrics,
        )
