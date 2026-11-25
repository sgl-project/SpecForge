"""Vision-dependent loss over DFlash proposal blocks."""

from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F

from specforge.data.loss_mask import has_consecutive_supervised_tokens
from specforge.data.visual_score import SCALE

from ..base import check_options
from ..inputs import gather, prepare_blocks, prepare_full_block_capacity
from ..losses import TokenLoss
from ..predictions import PredictionBatch, block_chunk_size
from ..registry import TrainingRequirements, register_protocol
from .dflash import COMMON_MODULES


def compute_mmflash(calc, *, smoothing, visual_alpha):
    logits = calc.outputs()["logits"]
    labels, mask = calc.labels, calc.weights > 0
    ce = F.cross_entropy(
        logits.flatten(0, 1), labels.flatten(), reduction="none"
    ).reshape_as(labels)
    with torch.no_grad():
        probability = (-ce.to(torch.promote_types(ce.dtype, torch.float32))).exp()
        confidence = ((1 - smoothing) * probability + smoothing).masked_fill(~mask, 1)
        prefix = confidence.cumprod(dim=-1)
        score = calc.predictions.context_inputs["visual_score"]
        visual = score.clamp(0, SCALE).float() / SCALE
        boost = 1 + visual_alpha * visual * (1 - probability)
        weights = prefix * boost
    # The executor applies the supervision mask and sums chunk numerators.
    calc.add(0, ce * weights)


def finalize_mmflash(plan_result):
    result = plan_result.objective_result()
    result.ratios["ce_loss"] = (result.numerator.detach(), result.denominator)
    return result


@register_protocol(
    "mmflash",
    default=False,
    runtime_name="dflash",
    architectures=("DFlashDraftModel",),
    inputs=("input_ids", "hidden_states", "loss_mask", "visual_score"),
    # The objective computes its own weights from draft logits and visual scores.
    provides=("logits", "labels"),
    trains=COMMON_MODULES,
    training=TrainingRequirements(
        offline_modalities=("multimodal",),
        host_inputs=prepare_full_block_capacity,
        loss_mask_filter=has_consecutive_supervised_tokens,
        dataloader_workers=8,
        liger_by_default=True,
    ),
)
@dataclass
class MMFlashObjective:
    num_anchors: int = 512
    smoothing: float = 0.5
    visual_alpha: float = 1.0

    default_loss = TokenLoss(kind="value")

    def __post_init__(self):
        check_options(
            counts={"num_anchors": self.num_anchors},
            probabilities={"smoothing": self.smoothing},
            nonnegative={"visual_alpha": self.visual_alpha},
        )
        if (
            self.smoothing is None
            or self.visual_alpha is None
            or self.num_anchors is None
        ):
            raise ValueError("MMFlash objective options must have concrete values")

    def __call__(self, runtime, batch, **context):
        scores = batch["visual_score"]
        if scores.dtype != torch.long or scores.shape != batch["input_ids"].shape:
            raise ValueError("visual_score must be int64 and aligned to input_ids")
        model = runtime.draft_model
        blocks = prepare_blocks(
            model.config.draft,
            batch,
            num_anchors=self.num_anchors,
            full_block_anchors=True,
        )
        output = model(
            blocks.input_ids,
            batch["hidden_states"],
            blocks.layer_inputs,
            output_logits=False,
        )
        predictions = runtime.block_predictions(
            output,
            blocks,
            batch,
            context_inputs={"visual_score": gather(scores, blocks.label_positions)},
            compute=partial(
                compute_mmflash,
                smoothing=self.smoothing,
                visual_alpha=self.visual_alpha,
            ),
            # Divide once by the number of sequences, even with chunked projection.
            denominator=blocks.weights.new_tensor(batch["input_ids"].shape[0]),
        )
        return PredictionBatch(
            (predictions,),
            chunk_size=block_chunk_size(batch),
            finalize=finalize_mmflash,
        )
