"""DFlash explicit-token cross entropy over anchored proposals."""

from dataclasses import dataclass

from specforge.data.loss_mask import has_consecutive_supervised_tokens

from .base import check_options
from .inputs import prepare_blocks
from .losses import TokenLoss, decay_weighting
from .moe_balance import check_aux_loss_type
from .predictions import PredictionBatch
from .registry import TrainingRequirements, register_protocol

COMMON_MODULES = ("backbone", "fusion", "embed_tokens", "lm_head")


@register_protocol(
    "dflash",
    training=TrainingRequirements(
        capture_method="dflash",
        minimum_loss_tokens=2,
        loss_mask_filter=has_consecutive_supervised_tokens,
    ),
    architectures=("DFlashDraftModel",),
    inputs=("input_ids", "hidden_states", "loss_mask"),
    provides=(
        "logits",
        "labels",
        "teacher_logits",
        "chain_positions",
        "acceptance_chain",
        "expert_routing",
    ),
    trains=COMMON_MODULES,
)
@dataclass
class DFlashObjective:
    num_anchors: int = 512
    loss_decay_gamma: float | None = None
    bias_update_rate: float = (
        0.0  # expert_bias step per optimizer step; 0 keeps it fixed
    )
    aux_loss_coef: float = 0.0  # weight of the load-balance loss; 0 skips it
    aux_loss_type: str = (
        "batch_wise"  # its load: this step's so far, or microbatch_wise
    )

    default_loss = TokenLoss(kind="ce")
    default_weighting_options = ("loss_decay_gamma",)

    @property
    def default_weighting(self):
        return decay_weighting(self.loss_decay_gamma)

    def __post_init__(self):
        check_options(
            counts={"num_anchors": self.num_anchors},
            positive={"loss_decay_gamma": self.loss_decay_gamma},
            nonnegative={
                "bias_update_rate": self.bias_update_rate,
                "aux_loss_coef": self.aux_loss_coef,
            },
        )
        check_aux_loss_type(self.aux_loss_type)

    def __call__(self, runtime, batch, **context):
        model = runtime.draft_model
        blocks = prepare_blocks(
            model.config.draft,
            batch,
            num_anchors=self.num_anchors,
        )
        output = model(
            blocks.input_ids,
            batch["hidden_states"],
            blocks.layer_inputs,
            output_logits=False,
        )
        predictions = runtime.block_predictions(output, blocks, batch)
        return PredictionBatch(
            (predictions,), deferred=runtime.deferred_term(output, blocks)
        )
