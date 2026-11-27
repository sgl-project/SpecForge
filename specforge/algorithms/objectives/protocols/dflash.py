"""DFlash explicit-token cross entropy over anchored proposals."""

from dataclasses import dataclass

from specforge.data.loss_mask import has_consecutive_supervised_tokens

from ..base import check_options
from ..diagnostics import block_diagnostics, diagnostic_requirements, report_token_loss
from ..inputs import prepare_anchor_capacity, prepare_blocks
from ..losses import TokenLoss, decay_weighting
from ..predictions import PredictionBatch, block_chunk_size
from ..projection import LinearProjection
from ..registry import TrainingRequirements, register_protocol

COMMON_MODULES = ("backbone", "fusion", "embed_tokens", "lm_head")


@register_protocol(
    "dflash",
    resolve_requires=diagnostic_requirements,
    training=TrainingRequirements(
        host_inputs=prepare_anchor_capacity,
        samples_anchors=True,
        capture_method="dflash",
        minimum_loss_tokens=2,
        loss_mask_filter=has_consecutive_supervised_tokens,
        dataloader_workers=8,
        liger_by_default=True,
    ),
    architectures=("DFlashDraftModel",),
    inputs=("input_ids", "hidden_states", "loss_mask"),
    provides=(
        "logits",
        "labels",
        "teacher_logits",
        "chain_positions",
        "acceptance_chain",
    ),
    # width adapters exist only on drafts narrower than the target
    trains=COMMON_MODULES + ("input_adapter", "output_adapter"),
)
@dataclass
class DFlashObjective:
    num_anchors: int = 512
    teacher_metrics: bool = False
    metric_top_k: int = 16
    loss_decay_gamma: float | None = None

    #: Options that only select what is logged; not part of the resume record.
    logging_options = ("teacher_metrics", "metric_top_k")
    default_loss = TokenLoss(kind="ce")
    default_weighting_options = ("loss_decay_gamma",)

    @property
    def default_weighting(self):
        return decay_weighting(self.loss_decay_gamma)

    def __post_init__(self):
        check_options(
            counts={"num_anchors": self.num_anchors, "metric_top_k": self.metric_top_k},
            positive={"loss_decay_gamma": self.loss_decay_gamma},
        )

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

        def compute(calc):
            calc.run()
            report_token_loss(calc)

        predictions = runtime.block_predictions(
            output,
            blocks,
            batch,
            compute=compute,
            projection=LinearProjection(model.project, model.lm_head),
        )
        chunk_size = block_chunk_size(batch)
        return PredictionBatch(
            (predictions,),
            diagnostics=block_diagnostics(
                runtime,
                predictions,
                blocks,
                enabled=context.get("collect_detailed_metrics", True),
                chunk_size=chunk_size,
                teacher_metrics=self.teacher_metrics,
                top_k=self.metric_top_k,
            ),
            chunk_size=chunk_size,
        )
