"""DSpine's backbone, pair refinement and inter-layer alignment losses."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from specforge.data.loss_mask import has_consecutive_supervised_tokens
from specforge.lr_scheduler import warmup_steps_from_ratio

from ..base import ObjectiveResult, check_options
from ..configuration import LossDefinition
from ..inputs import prepare_anchor_capacity, prepare_blocks
from ..losses import LossContext, TokenLoss, decay_weighting
from ..plan import LossPlan, LossTerm
from ..predictions import PredictionBatch, block_chunk_size
from ..registry import TrainingRequirements, WindowStatistic, register_protocol


@register_protocol(
    "dspine",
    training=TrainingRequirements(
        host_inputs=prepare_anchor_capacity,
        capture_method="dspark",
        minimum_loss_tokens=2,
        loss_mask_filter=has_consecutive_supervised_tokens,
        dataloader_workers=8,
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
    alignment_warmup_steps: int | None = None
    alignment_warmup_ratio: float = 0.0
    replacement_layers: int = 3
    replacement_probability: float = 0.5
    replacement_hold_ratio: float = 1 / 6
    replacement_end_ratio: float = 1 / 3
    candidate_mass_threshold: float = 1e-4

    @property
    def loss_terms(self):
        return {
            "token": LossDefinition(self.default_loss),
            "refinement": LossDefinition(
                self.default_loss, capabilities=frozenset({"masked_ce_l1"})
            ),
            "alignment": LossDefinition(TokenLoss("cosine"), 0.5),
        }

    default_weighting_options = ("loss_decay_gamma",)

    @property
    def default_weighting(self):
        return decay_weighting(self.loss_decay_gamma)

    default_loss = TokenLoss("ce_l1", {"ce_weight": 0.1, "l1_weight": 0.9})

    @staticmethod
    def resolve_training_options(options, training):
        """Inherit the LR warmup only when the alignment schedule is unspecified."""
        ratio = (
            training.get("warmup_ratio", 0.0)
            if options.get("alignment_warmup_steps") is None
            else 0.0
        )
        return {"alignment_warmup_ratio": ratio, **options}

    def alignment_scale(self, step, total_steps):
        warmup = self.alignment_warmup_steps
        if warmup is None:
            if self.alignment_warmup_ratio and not total_steps:
                raise ValueError("DSpine alignment warmup ratio requires total_steps")
            warmup = warmup_steps_from_ratio(
                self.alignment_warmup_ratio, total_steps or 0
            )
        return min(max(step, 0) / warmup, 1) if warmup else 1

    def __post_init__(self):
        check_options(
            counts={"num_anchors": self.num_anchors, "top_k": self.top_k},
            positive={"loss_decay_gamma": self.loss_decay_gamma},
            nonnegative={
                "candidate_mass_threshold": self.candidate_mass_threshold,
            },
            zero_counts={
                "alignment_warmup_steps": self.alignment_warmup_steps,
                "replacement_layers": self.replacement_layers,
            },
            probabilities={
                "alignment_warmup_ratio": self.alignment_warmup_ratio,
                "replacement_probability": self.replacement_probability,
                "replacement_hold_ratio": self.replacement_hold_ratio,
                "replacement_end_ratio": self.replacement_end_ratio,
            },
        )
        if self.alignment_warmup_ratio is None:
            raise ValueError("alignment_warmup_ratio must be a probability")
        if self.alignment_warmup_steps is not None and self.alignment_warmup_ratio:
            raise ValueError("alignment warmup must use steps or ratio, not both")
        if self.candidate_mass_threshold >= 1:
            raise ValueError("candidate_mass_threshold must be below one")
        if not self.replacement_hold_ratio < self.replacement_end_ratio:
            raise ValueError(
                "replacement_hold_ratio must be below replacement_end_ratio"
            )

    def __call__(
        self,
        runtime,
        batch,
        *,
        step,
        total_steps,
        alignment_ce=None,
        **context,
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
        beta = self.alignment_scale(step, total_steps)

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
            refinement = calc.evaluate_term(
                1,
                scores,
                covered.long().argmax(-1),
                teacher=teacher.gather(-1, ids),
                context=refinement_context,
            )
            codes = F.embedding(labels, model.transfer_codes).float()
            alignment = (
                calc.evaluate_term(
                    2,
                    predictions.context_inputs["layer_features"],
                    codes[:, :, None],
                ).values
                / model.backbone.num_layers
            )
            ce_num = (terms.values * calc.weights).sum()
            denominator = calc.weights.sum().detach()
            coefficient = alignment_ce
            if coefficient is None:
                totals = torch.stack((ce_num.detach(), denominator))
                coefficient = totals[0] / totals[1].where(totals[1] > 0, 1)
            # Keep the two masked means additive until all chunks are available.
            weights = calc.request.terms[1].weights
            calc.add(1, refinement.statistics["masked_ce"])
            calc.partials.update(
                ce_mass=(weights * refinement_context.ce_mask).sum(),
                l1_mass=(weights * refinement_context.l1_mask).sum(),
                l1_sum=(weights * refinement.statistics["masked_l1"]).sum(),
            )
            calc.add(2, coefficient.detach() * alignment)
            calc.ratios["dspine/backbone_ce"] = (ce_num.detach(), denominator)
            for name, values in (
                ("backbone_l1", terms.statistics["teacher_l1"]),
                ("alignment_loss", alignment),
            ):
                calc.ratios[f"dspine/{name}"] = (
                    (values.detach() * calc.weights).sum(),
                    denominator,
                )
            calc.ratios["dspine/candidate_coverage"] = (
                (covered.any(-1) * predictions.mask).sum().detach(),
                predictions.mask.sum().detach(),
            )

        def finalize(result):
            backbone, ce, alignment = result.terms
            stats = result.partials
            supervised = result.ratios["acc"][1]
            ce_mean = ce.numerator / stats["ce_mass"].where(stats["ce_mass"] > 0, 1)
            l1_mean = stats["l1_sum"] / stats["l1_mass"].where(stats["l1_mass"] > 0, 1)
            refinement = (ce_mean + l1_mean) * supervised
            result.ratios["dspine/refinement_loss"] = (
                refinement.detach(),
                result.ratios["dspine/backbone_ce"][1],
            )
            token_weight, refinement_weight, alignment_weight = result.coefficients
            numerator = (
                token_weight * backbone.numerator
                + refinement_weight * refinement
                + alignment_weight * alignment.numerator
            )
            return ObjectiveResult(
                numerator,
                numerator.new_tensor(batch["input_ids"].shape[0]),
                result.ratios,
            )

        predictions = runtime.block_predictions(
            output,
            blocks,
            batch,
            compute=compute,
            plan=LossPlan(
                (
                    LossTerm(),
                    LossTerm("refinement"),
                    LossTerm("alignment", beta),
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
        return PredictionBatch(
            (predictions,), chunk_size=block_chunk_size(batch), finalize=finalize
        )
