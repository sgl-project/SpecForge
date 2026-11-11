"""DSpine's backbone, pair refinement and inter-layer alignment losses."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from specforge.core.fused_loss import ce_l1_loss

from .base import check_options, reduce_loss
from .dflash import COMMON_MODULES
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
class DSpineObjective:
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
        )
        teacher = runtime.teacher_logits(batch, blocks.label_positions - 1)
        terms = ce_l1_loss(
            output.logits, blocks.labels, teacher_logits=teacher, top_k=self.top_k
        )
        refined = model.refine(output.pre_injection, blocks.predecessors)
        weights = model.lm_head.candidate_weights(terms.topk_ids)
        scores = torch.einsum("...h,...kh->...k", refined, weights).float()
        covered = terms.topk_ids == blocks.labels[..., None]
        refinement = ce_l1_loss(
            scores,
            covered.long().argmax(-1),
            teacher_logits=teacher.gather(-1, terms.topk_ids),
        )
        mass = teacher.float().softmax(-1).gather(-1, terms.topk_ids).sum(-1)
        refined_loss = self.ce_weight * refinement.cross_entropy * covered.any(-1) + (
            1 - self.ce_weight
        ) * refinement.teacher_l1 * (mass > self.candidate_mass_threshold)
        codes = F.embedding(blocks.labels, model.transfer_codes).float()
        alignment = (
            1 - F.cosine_similarity(output.layer_features.float(), codes[None], dim=-1)
        ).sum(0)
        ce_num = (terms.cross_entropy * blocks.weights).sum()
        denominator = blocks.weights.sum().detach()
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
        loss = (
            self.ce_weight * terms.cross_entropy
            + (1 - self.ce_weight) * terms.teacher_l1
            + refined_loss
            + beta * alignment_ce.detach() * alignment
        )
        return reduce_loss(
            loss,
            blocks.weights,
            logits=output.logits,
            labels=blocks.labels,
            ratios={"dspine/backbone_ce": (ce_num.detach(), denominator)},
        )
