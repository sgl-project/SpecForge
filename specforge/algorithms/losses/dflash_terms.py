"""Losses: dflash terms."""

from typing import NamedTuple

import torch


class SelectorTokenTerms(NamedTuple):
    """Unweighted selector CE and non-differentiable candidate diagnostics."""

    cross_entropy: torch.Tensor
    target_is_candidate: torch.Tensor
    selected_ids: torch.Tensor


class SelectorTerms(NamedTuple):
    """Additive selector objective and metric terms for one objective chunk."""

    ce_num: torch.Tensor
    probability_num: torch.Tensor
    correct_num: torch.Tensor
    weight_den: torch.Tensor
    covered_num: torch.Tensor

    @classmethod
    def zeros(cls, reference: torch.Tensor) -> "SelectorTerms":
        return cls(
            reference.new_zeros(()),
            reference.new_zeros(()),
            reference.new_zeros(()),
            reference.new_zeros(()),
            reference.new_zeros(()),
        )


class DFlashObjectiveTerms(NamedTuple):
    """Additive terms for DFlash and D-PACE block objectives.

    This tuple covers the standard DFlash and D-PACE loss variants, including
    the optional DFlash2 selector terms. It is not shared with Domino or DSpark;
    those models define separate objective reductions. The fields stay flat and
    tensor-only to satisfy ``checkpointed_chunk_reduce``'s chunk-function contract.
    """

    ce_loss_num: torch.Tensor
    tv_loss_num: torch.Tensor
    loss_den: torch.Tensor
    target_probability_num: torch.Tensor
    correct_num: torch.Tensor
    accuracy_den: torch.Tensor
    selector_ce_num: torch.Tensor
    selector_probability_num: torch.Tensor
    selector_correct_num: torch.Tensor
    selector_weight_den: torch.Tensor
    selector_covered_num: torch.Tensor
