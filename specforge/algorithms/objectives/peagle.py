"""P-EAGLE COD sampling and vocabulary-aware parallel distillation."""

from dataclasses import dataclass

import torch

from specforge.algorithms.common.cod import (
    create_peagle_mask_mod,
    generate_cod_sample_indices,
)

from .base import check_options
from .inputs import gather, sequence_inputs, valid_tokens
from .losses import TokenLoss
from .predictions import PredictionBatch
from .registry import register_protocol


@register_protocol(
    "peagle",
    architectures=("PEagleDraftModel",),
    inputs=("input_ids", "hidden_states", "target_last_hidden_states", "loss_mask"),
    provides=(
        "logits",
        "labels",
        "teacher_logits",
        "chain_positions",
        "acceptance_chain",
    ),
    trains=("backbone", "fusion", "embed_tokens", "lm_head", "mask_hidden"),
)
@dataclass
class PEagleObjective:
    num_depths: int = 8
    down_sample_ratio: float = 0.8
    down_sample_ratio_min: float = 0.2

    default_loss = TokenLoss(kind="soft_ce")

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
        ends = lengths.reshape(-1).to(device=ids.device).cumsum(0)
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
            output_logits=False,
        )
        model.vocabulary.require_ready()
        predictions = runtime.predictions(
            output.hidden_states,
            batch,
            original + 1,
            mask=gather(supervised, original),
            positions=depths[None],
            chain_ids=anchors.reshape(-1, 1),
            vocab_mask=model.vocabulary.t2d,
            requires=frozenset({"teacher_logits"}),
            default_loss=self.default_loss,
        )
        return PredictionBatch((predictions,), reduction="sum_means")
