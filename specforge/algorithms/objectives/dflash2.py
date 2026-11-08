"""DFlash2 unary CE plus covered-candidate selector CE."""

from dataclasses import dataclass

from specforge.core.fused_loss import ce_l1_loss

from .base import check_options, reduce_loss
from .dflash import COMMON_MODULES
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
class DFlash2Objective:
    num_anchors: int = 512
    loss_decay_gamma: float | None = None
    selector_loss_alpha: float = 1.0
    selector_warmup_steps: int = 0
    selector_ramp_steps: int = 0
    selector_stop_gradient: bool = False

    def __post_init__(self):
        check_options(
            counts={"num_anchors": self.num_anchors},
            positive={"loss_decay_gamma": self.loss_decay_gamma},
            nonnegative={"selector_loss_alpha": self.selector_loss_alpha},
            zero_counts={
                "selector_warmup_steps": self.selector_warmup_steps,
                "selector_ramp_steps": self.selector_ramp_steps,
            },
        )
        if type(self.selector_stop_gradient) is not bool:
            raise ValueError("selector_stop_gradient must be boolean")

    def __call__(self, runtime, batch, *, step, **context):
        model = runtime.draft_model
        blocks = prepare_blocks(
            model.config.draft,
            batch,
            num_anchors=self.num_anchors,
            gamma=self.loss_decay_gamma,
        )
        output = model(blocks.input_ids, batch["hidden_states"], blocks.layer_inputs)
        labels = blocks.labels.flatten(1)
        terms = ce_l1_loss(
            output.logits, labels, top_k=model.config.draft.spec_config.selector_top_k
        )
        unary, hidden = terms.topk_values, output.hidden_states
        if self.selector_stop_gradient:
            unary, hidden = unary.detach(), hidden.detach()
        scores = model.candidate_selector.score_candidates(
            candidate_ids=terms.topk_ids,
            unary_logits=unary,
            hidden_states=hidden,
            predecessor_ids=blocks.predecessors.flatten(1),
        )
        covered = terms.topk_ids == labels[..., None]
        selector = ce_l1_loss(
            scores, covered.long().argmax(-1)
        ).cross_entropy * covered.any(-1)
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
        loss = terms.cross_entropy + self.selector_loss_alpha * scale * selector
        return reduce_loss(
            loss, blocks.weights.flatten(1), logits=output.logits, labels=labels
        )
