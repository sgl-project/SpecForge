"""DSpine's backbone, pair refinement and inter-layer alignment losses."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from specforge.data.loss_mask import has_consecutive_supervised_tokens

from ..base import check_options
from ..configuration import LossDefinition
from ..inputs import prepare_blocks
from ..losses import LossContext, TokenLoss, combine_ce_l1, decay_weighting
from ..plan import LossPlan, LossTerm
from ..predictions import PredictionBatch
from ..registry import TrainingRequirements, WindowStatistic, register_protocol


@register_protocol(
    "dspine",
    training=TrainingRequirements(
        capture_method="dspark",
        minimum_loss_tokens=2,
        loss_mask_filter=has_consecutive_supervised_tokens,
        window_statistic=WindowStatistic(
            ratio_name="dspine/backbone_ce", argument_name="alignment_ce"
        ),
    ),
    requires=("teacher_logits",),
    architectures=("DSpineDraftModel",),
    inputs=("input_ids", "hidden_states", "target_last_hidden_states", "loss_mask"),
    provides=(
        "logits",
        "labels",
        "teacher_logits",
        "chain_positions",
        "acceptance_chain",
    ),
    trains=("backbone", "fusion", "injection"),
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

    legacy_loss_options = ("ce_weight", "alignment_weight")

    @property
    def loss_terms(self):
        return {
            "token": LossDefinition(self.default_loss),
            "refinement": LossDefinition(
                self.default_loss, capabilities=frozenset({"component_masks"})
            ),
            "alignment": LossDefinition(TokenLoss("cosine"), self.alignment_weight),
        }

    default_weighting_options = ("loss_decay_gamma",)

    @property
    def default_weighting(self):
        return decay_weighting(self.loss_decay_gamma)

    @property
    def default_loss(self):
        return TokenLoss(
            kind="ce_l1",
            options={"ce_weight": self.ce_weight, "l1_weight": 1 - self.ce_weight},
        )

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
        if self.candidate_mass_threshold >= 1:
            raise ValueError("candidate_mass_threshold must be below one")
        if not self.replacement_hold_ratio < self.replacement_end_ratio:
            raise ValueError(
                "replacement_hold_ratio must be below replacement_end_ratio"
            )

    def __call__(
        self, runtime, batch, *, step, total_steps, alignment_ce=None, **context
    ):
        model = runtime.draft_model
        blocks = prepare_blocks(
            model.config.draft,
            batch,
            num_anchors=self.num_anchors,
            prefix=True,
        )
        probability = self.replacement_probability if model.training else 0
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
        replacement = (
            (torch.rand_like(blocks.keep.float()) < probability) & blocks.keep
            if probability
            else torch.zeros_like(blocks.keep)
        )
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
        beta = (
            min(step / self.alignment_warmup_steps, 1)
            if self.alignment_warmup_steps
            else 1
        )

        if runtime.loss_configuration is None:
            beta *= self.alignment_weight

        def compute(calc):
            calc.run((0,))
            predictions, labels, teacher = calc.predictions, calc.labels, calc.teacher
            terms = calc.evaluate(
                "candidates",
                TokenLoss(kind="ce"),
                calc.outputs()["logits"],
                labels,
                teacher=teacher,
                return_teacher_l1=True,
                top_k=self.top_k,
                term_index=0,
            )
            ids = terms.statistics["topk_ids"]
            refined = model.refine(
                predictions.context_inputs["pre_injection"],
                predictions.context_inputs["predecessors"],
            )
            candidates = model.lm_head.candidate_weights(ids)
            scores = torch.einsum("...h,...kh->...k", refined, candidates).float()
            covered = ids == labels[..., None]
            mass = teacher.float().softmax(-1).gather(-1, ids).sum(-1)
            refinement_context = LossContext(
                calc.request.terms[1].mask,
                ce_mask=covered.any(-1),
                l1_mask=mass > self.candidate_mass_threshold,
            )
            if runtime.loss_configuration is None:
                # Keep old checkpoint implementation identities. The mathematical
                # combination is shared with the configured CE/L1 formula.
                values = calc.evaluate(
                    "refinement",
                    TokenLoss("ce"),
                    scores,
                    covered.long().argmax(-1),
                    teacher=teacher.gather(-1, ids),
                    return_teacher_l1=True,
                )
                refinement = combine_ce_l1(
                    values.values,
                    values.statistics["teacher_l1"],
                    refinement_context,
                    self.ce_weight,
                    1 - self.ce_weight,
                )
            else:
                refinement = calc.evaluate_term(
                    1,
                    scores,
                    covered.long().argmax(-1),
                    teacher=teacher.gather(-1, ids),
                    context=refinement_context,
                )
            codes = F.embedding(labels, model.transfer_codes).float()
            alignment = calc.evaluate_term(
                2,
                predictions.context_inputs["layer_features"],
                codes[:, :, None],
                fused=False if runtime.loss_configuration is None else None,
            ).values
            ce_num = (terms.values * calc.weights).sum()
            denominator = calc.weights.sum().detach()
            coefficient = alignment_ce
            if coefficient is None:
                totals = torch.stack((ce_num.detach(), denominator))
                coefficient = totals[0] / totals[1].where(totals[1] > 0, 1)
            calc.add(1, refinement.values)
            calc.add(2, coefficient.detach() * alignment)
            calc.ratios["dspine/backbone_ce"] = (ce_num.detach(), denominator)

        predictions = runtime.block_predictions(
            output,
            blocks,
            batch,
            compute=compute,
            plan=LossPlan(
                (
                    LossTerm(),
                    LossTerm("refinement", loss=self.default_loss),
                    LossTerm("alignment", beta, TokenLoss("cosine")),
                )
            ),
            requires=frozenset({"teacher_logits"}),
            default_loss=self.default_loss,
            chunk_safe=alignment_ce is not None,
            context_inputs={
                "pre_injection": output.pre_injection,
                "predecessors": blocks.predecessors,
                "layer_features": output.layer_features.movedim(0, -2),
            },
        )
        return PredictionBatch((predictions,))
