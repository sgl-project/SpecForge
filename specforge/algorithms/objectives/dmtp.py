"""DMTP supervises the suffix after the externally supplied first candidate."""

from dataclasses import dataclass

from specforge.core.fused_loss import ce_l1_loss

from .base import check_options, reduce_loss
from .dflash import COMMON_MODULES
from .inputs import gather, prepare_blocks
from .registry import register_objective


@register_objective(
    "dmtp",
    architectures=("DMTPDraftModel",),
    inputs=("input_ids", "hidden_states", "mtp_hidden_states", "loss_mask"),
    outputs=("logits",),
    trains=COMMON_MODULES + ("mtp_fusion",),
)
@dataclass
class DMTPObjective:
    num_anchors: int = 512
    loss_decay_gamma: float | None = None

    def __post_init__(self):
        check_options(
            counts={"num_anchors": self.num_anchors},
            positive={"loss_decay_gamma": self.loss_decay_gamma},
        )

    def __call__(self, runtime, batch, **context):
        model, spec = runtime.draft_model, runtime.draft_model.config.draft.spec_config
        blocks = prepare_blocks(
            model.config.draft,
            batch,
            num_anchors=self.num_anchors,
            gamma=self.loss_decay_gamma,
            input_offset=1,
            label_offset=1,
            context_offset=spec.context_end_offset,
        )
        mtp = gather(
            batch["mtp_hidden_states"],
            blocks.anchors + 1 + spec.mtp_features.token_offset,
        )
        output = model(
            blocks.input_ids,
            batch["hidden_states"],
            blocks.layer_inputs,
            mtp_features=mtp,
            block_keep_mask=blocks.keep,
        )
        logits = output.logits.reshape(*blocks.labels.shape, -1)
        return reduce_loss(
            ce_l1_loss(logits, blocks.labels).cross_entropy,
            blocks.weights,
            logits=logits,
            labels=blocks.labels,
        )
