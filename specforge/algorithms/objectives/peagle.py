"""P-EAGLE COD sampling and vocabulary-aware parallel distillation."""

from dataclasses import dataclass

import torch

from ..losses.eagle_teacher import teacher_distribution
from .base import ObjectiveResult, check_options
from .inputs import gather, sequence_inputs, valid_tokens
from .peagle_inputs import create_peagle_mask_mod, generate_cod_sample_indices
from .registry import register_objective


@register_objective(
    "peagle",
    architectures=("PEagleDraftModel",),
    inputs=("input_ids", "hidden_states", "target_last_hidden_states", "loss_mask"),
    outputs=("logits",),
    trains=("backbone", "fusion", "embed_tokens", "lm_head", "mask_hidden"),
)
@dataclass
class PEagleObjective:
    num_depths: int = 8
    down_sample_ratio: float = 0.8
    down_sample_ratio_min: float = 0.2

    def __post_init__(self):
        check_options(
            counts={"num_depths": self.num_depths},
            probabilities={
                "down_sample_ratio": self.down_sample_ratio,
                "down_sample_ratio_min": self.down_sample_ratio_min,
            },
        )

    def __call__(self, runtime, batch, **context):
        model, ids = runtime.draft_model, batch["input_ids"]
        if ids.shape[0] != 1:
            raise ValueError("P-EAGLE COD sampling requires batch size one")
        valid = valid_tokens(batch)
        # The capture contract is unshifted; align once before COD expansion.
        shifted = torch.cat((ids[:, 1:], torch.zeros_like(ids[:, :1])), 1)
        supervised = torch.cat((valid[:, 1:], torch.zeros_like(valid[:, :1])), 1)
        lengths = batch.get("lengths", ids.new_tensor([ids.shape[1]]))
        anchors, depths = generate_cod_sample_indices(
            ids.shape[1],
            supervised,
            self.num_depths,
            self.down_sample_ratio,
            self.down_sample_ratio_min,
            lengths=lengths,
        )
        original = (anchors + depths)[None]
        feature_mask = (depths > 0)[None]
        sampled = shifted[:, anchors + depths]
        mask_mod = create_peagle_mask_mod(anchors, depths, lengths, ids.shape[1])
        output = model(
            sampled,
            gather(batch["hidden_states"], original),
            sequence_inputs(
                model.config.draft,
                original,
                mask_mod,
                active_queries=(anchors < lengths.sum())[None],
            ),
            feature_mask=feature_mask,
        )
        teacher, included = teacher_distribution(
            model, runtime.teacher_logits(batch, original + 1)
        )
        weight = gather(supervised, original).float() * included
        if output.logits.is_cuda:
            from specforge.core.loss import LogSoftmaxLoss

            numerator = (
                LogSoftmaxLoss.apply(output.logits, teacher, weight[..., None])
                * weight.numel()
            )
        else:
            ce = -(teacher * output.logits.float().log_softmax(-1)).sum(-1)
            numerator = (ce * weight).sum()
        correct = (output.logits.argmax(-1) == teacher.argmax(-1)).detach()
        ratios = {"accuracy": ((correct * weight).sum(), weight.sum())}
        for depth in range(self.num_depths):
            mask = (depths == depth)[None] * weight
            ratios[f"position_{depth}_accuracy"] = ((correct * mask).sum(), mask.sum())
        return ObjectiveResult(numerator, weight.sum().detach(), ratios)
