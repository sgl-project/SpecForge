"""DSpine's backbone, pair refinement and inter-layer alignment losses."""

from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist

from specforge.core.chunking import checkpointed_chunk_reduce

from ..losses.blocks import gather_teacher_hidden, logits_loss_terms
from ..losses.dspine import dspine_chunk_terms
from .base import COMMON_MODULES, ObjectiveResult, check_options
from .block_options import BlockLossOptions
from .inputs import prepare_blocks
from .registry import register_objective


@register_objective(
    "dspine",
    architectures=("DSpineDraftModel",),
    inputs=("input_ids", "hidden_states", "target_last_hidden_states", "loss_mask"),
    outputs=("logits", "pre_injection", "layer_features"),
    trains=COMMON_MODULES + ("injection",),
)
@dataclass
class DSpineObjective(BlockLossOptions):
    num_anchors: int = 512
    loss_decay_gamma: float = 7.0
    top_k: int = 16
    ce_weight: float = 0.1
    alignment_weight: float = 0.5
    alignment_warmup_steps: int = 146
    replacement_layers: int = 3
    replacement_probability: float = 0.5
    replacement_hold_ratio: float = 1 / 6
    replacement_end_ratio: float = 1 / 3
    candidate_mass_threshold: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        check_options(
            counts={"num_anchors": self.num_anchors, "top_k": self.top_k},
            positive={"loss_decay_gamma": self.loss_decay_gamma},
            nonnegative={
                "alignment_weight": self.alignment_weight,
                "candidate_mass_threshold": self.candidate_mass_threshold,
            },
            zero_counts={
                "alignment_warmup_steps": self.alignment_warmup_steps,
                "replacement_layers": self.replacement_layers,
            },
            probabilities={
                "ce_weight": self.ce_weight,
                "replacement_probability": self.replacement_probability,
                "replacement_hold_ratio": self.replacement_hold_ratio,
                "replacement_end_ratio": self.replacement_end_ratio,
            },
        )
        if not self.replacement_hold_ratio < self.replacement_end_ratio:
            raise ValueError(
                "replacement_hold_ratio must be below replacement_end_ratio"
            )

    def __call__(self, runtime, batch, *, step, total_steps, alignment_ce, **context):
        model = runtime.draft_model
        blocks = prepare_blocks(
            model.config.draft,
            batch,
            num_anchors=self.num_anchors,
            gamma=self.loss_decay_gamma,
            prefix=True,
        )
        probability = self.replacement_probability if runtime.training else 0
        if probability:
            if not total_steps:
                raise ValueError("DSpine replacement schedule requires total_steps")
            progress = step / total_steps
            probability *= min(
                max(
                    (self.replacement_end_ratio - progress)
                    / (self.replacement_end_ratio - self.replacement_hold_ratio),
                    0,
                ),
                1,
            )
        replacement = (torch.rand_like(blocks.keep.float()) < probability) & blocks.keep
        masks = tuple(
            replacement if index < self.replacement_layers else None
            for index in range(model.backbone.num_layers)
        )
        output = model(
            blocks.input_ids,
            batch["hidden_states"],
            blocks.layer_inputs,
            block_token_ids=blocks.labels,
            replace_masks=masks,
            output_logits=False,
        )
        labels = blocks.labels[..., 1:].flatten(0, 1)
        weights = blocks.weights[..., 1:].flatten(0, 1)
        supervised = weights > 0
        hidden = output.hidden_states[..., 1:, :].flatten(0, 1)
        denominator = weights.sum().detach()
        if context.get("ce_only", False):

            def ce_terms(h, labels, weights, supervised):
                terms = logits_loss_terms(self, model.lm_head(h), labels)
                return (terms.cross_entropy * weights).sum(), (
                    (terms.argmax_ids == labels) * supervised
                ).sum().float()

            ce_num, correct = checkpointed_chunk_reduce(
                ce_terms,
                hidden,
                labels,
                weights,
                supervised,
                chunk_size=self.objective_chunk_blocks,
                checkpoint=False,
            )
            return ObjectiveResult(
                ce_num,
                denominator,
                {
                    "accuracy": (correct, supervised.sum()),
                    "dspine/backbone_ce": (ce_num.detach(), denominator),
                },
            )
        teacher_hidden = gather_teacher_hidden(
            batch["target_last_hidden_states"], blocks.label_positions[..., 1:]
        ).flatten(0, 1)
        features = (
            output.layer_features[..., 1:, :].permute(1, 2, 0, 3, 4).flatten(0, 1)
        )
        ce_num, l1_num, refinement_num, alignment_num, correct, coverage = (
            checkpointed_chunk_reduce(
                partial(dspine_chunk_terms, model, self, runtime.target_head),
                hidden,
                output.pre_injection[..., 1:, :].flatten(0, 1),
                features,
                labels,
                blocks.predecessors[..., 1:].flatten(0, 1),
                teacher_hidden,
                weights,
                supervised,
                chunk_size=self.objective_chunk_blocks,
                checkpoint=self.objective_checkpointing,
            )
        )
        if alignment_ce is None:
            totals = torch.stack((ce_num.detach(), denominator))
            if dist.is_initialized():
                dist.all_reduce(totals)
            alignment_ce = totals[0] / totals[1].clamp_min(1)
        beta = self.alignment_weight * (
            min(step / self.alignment_warmup_steps, 1)
            if self.alignment_warmup_steps
            else 1
        )
        numerator = (
            self.ce_weight * ce_num
            + (1 - self.ce_weight) * l1_num
            + refinement_num
            + beta * alignment_ce.detach() * alignment_num
        )
        count = supervised.sum().detach()
        return ObjectiveResult(
            numerator,
            denominator,
            {
                "accuracy": (correct.detach(), count),
                "dspine/backbone_ce": (ce_num.detach(), denominator),
                "dspine/backbone_l1": (l1_num.detach(), denominator),
                "dspine/refinement_loss": (refinement_num.detach(), denominator),
                "dspine/alignment_loss": (alignment_num.detach(), denominator),
                "dspine/candidate_coverage": (coverage.detach(), count),
            },
        )
