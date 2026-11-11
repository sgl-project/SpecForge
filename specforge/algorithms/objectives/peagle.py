"""P-EAGLE COD sampling and vocabulary-aware parallel distillation."""

from dataclasses import dataclass

import torch

from specforge.algorithms.common.cod import (
    create_peagle_mask_mod,
    generate_cod_sample_indices,
)

from .base import ObjectiveResult, check_options, reduce_loss
from .eagle3 import teacher_distribution
from .inputs import gather, sequence_inputs, valid_tokens
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

        if not 0 < self.down_sample_ratio_min <= self.down_sample_ratio:
            raise ValueError("sampling ratios must satisfy 0 < min <= ratio <= 1")

    def __call__(self, runtime, batch, **context):
        model, ids = runtime.draft_model, batch["input_ids"]
        if ids.shape[0] != 1:
            raise ValueError("P-EAGLE COD sampling requires batch size one")
        valid = valid_tokens(batch)
        # The capture contract is unshifted; align once before COD expansion.
        shifted = torch.cat((ids[:, 1:], torch.zeros_like(ids[:, :1])), 1)
        supervised = torch.cat((valid[:, 1:], torch.zeros_like(valid[:, :1])), 1)
        lengths = batch.get("lengths", ids.new_tensor([ids.shape[1]]))
        # A shifted label must stay in the same packed document.
        ends = lengths.to(device=ids.device).cumsum(0)
        supervised &= ~torch.isin(
            torch.arange(1, ids.shape[1] + 1, device=ids.device), ends
        )[None]
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
        index = torch.arange(anchors.numel(), device=ids.device)
        mask_mod = create_peagle_mask_mod(anchors, depths, lengths, ids.shape[1])
        visible = mask_mod(0, 0, index[:, None], index[None])[None]
        output = model(
            sampled,
            gather(batch["hidden_states"], original),
            sequence_inputs(model.config.draft, original, visible),
            feature_mask=feature_mask,
        )
        teacher_logits = runtime.teacher_logits(batch, original + 1)
        teacher, included = teacher_distribution(model, teacher_logits)
        weight = gather(supervised, original).float() * included
        ce = -(teacher * output.logits.float().log_softmax(-1)).sum(-1)
        ce = runtime.loss(
            output.logits,
            teacher.argmax(-1),
            batch,
            original + 1,
            default=ce,
            teacher=teacher_logits[..., model.vocabulary.t2d],
        )
        weight = runtime.weighting(weight, depths[None])
        result = reduce_loss(
            ce, weight, logits=output.logits, labels=teacher.argmax(-1)
        )
        return ObjectiveResult(result.loss, weight.new_ones(()), result.ratios)
