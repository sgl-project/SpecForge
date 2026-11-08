"""DFlash explicit-token cross entropy over anchored proposals."""

from dataclasses import dataclass

from specforge.core.fused_loss import ce_l1_loss

from .base import check_options, reduce_loss
from .inputs import prepare_blocks
from .registry import register_objective

COMMON_MODULES = ("backbone", "fusion", "embed_tokens", "lm_head")


@register_objective(
    "dflash",
    architectures=("DFlashDraftModel",),
    inputs=("input_ids", "hidden_states", "loss_mask"),
    outputs=("logits",),
    trains=COMMON_MODULES,
)
@dataclass
class DFlashObjective:
    num_anchors: int = 512
    loss_decay_gamma: float | None = None

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
            gamma=self.loss_decay_gamma,
        )
        output = model(blocks.input_ids, batch["hidden_states"], blocks.layer_inputs)
        logits = output.logits.reshape(*blocks.labels.shape, -1)
        terms = ce_l1_loss(logits, blocks.labels)
        return reduce_loss(
            terms.cross_entropy, blocks.weights, logits=logits, labels=blocks.labels
        )
