"""Additive, hard-label prefix diagnostics on fixed training blocks.

These measure greedy matches to recorded labels, not stochastic verification
against a target distribution or the cross-block serving walk.
"""

from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F


class PrefixCounts(NamedTuple):
    # Rows are unary greedy, top-K oracle, and selector greedy, in that order.
    # Columns include a zero anchor entry followed by the predicted positions.
    reached: torch.Tensor
    accepted: torch.Tensor
    selector_covered: torch.Tensor
    selector_unary_correct: torch.Tensor
    eligible: torch.Tensor


def _prefix_events(hit: torch.Tensor, supervised: torch.Tensor):
    accepted = (hit & supervised).long().cumprod(dim=-1).bool()
    previous_accepted = torch.cat(
        (torch.ones_like(accepted[..., :1]), accepted[..., :-1]), dim=-1
    )
    return previous_accepted & supervised, accepted


def hard_label_prefix_counts(
    unary_hit: torch.Tensor,
    covered: torch.Tensor,
    selector_hit: Optional[torch.Tensor],
    supervised: torch.Tensor,
) -> PrefixCounts:
    """Count reached and accepted slots, stopping at the first miss or mask.

    Inputs have shape [batch, blocks, predicted positions], without the anchor.
    A slot is reached only if it is supervised and all preceding slots matched.
    Coverage and unary correctness are also counted on the *selector's* reached
    prefixes, so candidate recall and ranking accuracy share a population.
    """
    if selector_hit is None:
        selector_reached = selector_accepted = torch.zeros_like(supervised)
    else:
        selector_reached, selector_accepted = _prefix_events(selector_hit, supervised)
    unary_reached, unary_accepted = _prefix_events(unary_hit, supervised)
    oracle_reached, oracle_accepted = _prefix_events(covered, supervised)

    def count(events):
        return F.pad(events.float().sum(dim=(0, 1)), (1, 0))

    return PrefixCounts(
        reached=torch.stack(
            [count(unary_reached), count(oracle_reached), count(selector_reached)]
        ),
        accepted=torch.stack(
            [count(unary_accepted), count(oracle_accepted), count(selector_accepted)]
        ),
        selector_covered=count(selector_reached & covered),
        selector_unary_correct=count(selector_reached & unary_hit),
        eligible=count(supervised.long().cumprod(dim=-1).bool()),
    )
