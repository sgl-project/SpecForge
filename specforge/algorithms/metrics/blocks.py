"""Metrics: blocks."""

from typing import Dict, Optional, Tuple

import torch

from specforge.algorithms.metrics.dflash_terms import DFlashMetricTerms


def add_position_ratios(
    metrics: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    name: str,
    numerators: torch.Tensor,
    denominators: torch.Tensor,
) -> None:
    """Add one ratio per predicted block position.

    Per-position keys drop the algorithm prefix and live under their own
    ``position_<k>/`` tracker section so each block position renders as one
    dashboard group.
    """

    family = name.split("/", 1)[1]
    for position in range(1, numerators.numel()):
        metrics[f"position_{position}/{family}"] = (
            numerators[position].detach(),
            denominators[position].detach(),
        )


def add_ratio_family(
    metrics: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    name: str,
    numerators: torch.Tensor,
    denominators: torch.Tensor,
) -> None:
    """Add the aggregate and each predicted-position ratio."""

    metrics[name] = (
        numerators.sum().detach(),
        denominators.sum().detach(),
    )
    add_position_ratios(metrics, name, numerators, denominators)


def add_position_counts(
    metrics: Dict[str, torch.Tensor], name: str, counts: torch.Tensor
) -> None:
    family = name.split("/", 1)[1]
    for position in range(1, counts.numel()):
        metrics[f"position_{position}/{family}"] = counts[position].detach()


def add_prefix_metrics(
    ratios: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    counts: Dict[str, torch.Tensor],
    terms: DFlashMetricTerms,
    top_k: int,
    *,
    has_selector: bool,
) -> None:
    """Expose survival and first-failure diagnostics with exact counts."""
    block_den = terms.block_den.detach()
    counts["dflash/hard_label/block_count"] = block_den
    add_position_counts(
        counts, "dflash/hard_label/supervised_count", terms.position_den
    )
    add_position_counts(
        counts, "dflash/hard_label/valid_prefix_count", terms.prefix_eligible_num
    )
    ratios["dflash/hard_label/supervised_prefix_length"] = (
        block_den + terms.prefix_eligible_num.sum().detach(),
        block_den,
    )
    ratios["dflash/hard_label/unary_greedy_accepted_length"] = (
        block_den + terms.prefix_accepted_num[0].sum().detach(),
        block_den,
    )
    paths = [
        "dflash/hard_label/unary_greedy",
        f"dflash/hard_label/unary_top{top_k}_oracle",
    ]
    if has_selector:
        paths.append("dflash2/selector/greedy")
    for index, path in enumerate(paths):
        reached = terms.prefix_reached_num[index]
        accepted = terms.prefix_accepted_num[index]
        add_ratio_family(ratios, f"{path}_prefix_acceptance", accepted, reached)
        add_position_ratios(
            ratios,
            f"{path}_prefix_survival",
            accepted,
            block_den.expand_as(accepted),
        )
        for suffix, values in (
            ("reached_count", reached),
            ("accepted_count", accepted),
            ("first_failure_count", reached - accepted),
        ):
            add_position_counts(counts, f"{path}_{suffix}", values)

    if has_selector:
        reached = terms.prefix_reached_num[2]
        accepted = terms.prefix_accepted_num[2]
        covered = terms.selector_prefix_covered_num
        for name, numerator, denominator in (
            (
                "unary_top1_accuracy",
                terms.selector_prefix_unary_correct_num,
                reached,
            ),
            (f"top{top_k}_recall", covered, reached),
            ("covered_accuracy", accepted, covered),
            ("coverage_miss_rate", reached - covered, reached),
            ("ranking_error_rate", covered - accepted, reached),
        ):
            add_ratio_family(
                ratios,
                f"dflash2/selector/greedy_prefix_{name}",
                numerator,
                denominator,
            )
        for name, values in (
            ("covered_count", covered),
            ("coverage_miss_count", reached - covered),
            ("ranking_error_count", covered - accepted),
        ):
            add_position_counts(counts, f"dflash2/selector/greedy_{name}", values)


def compute_accept_len(
    pred_ids_4d: torch.Tensor,
    target_ids_4d: torch.Tensor,
    valid_mask_4d: torch.Tensor,
) -> torch.Tensor:
    """Compute per-block acceptance length."""
    correct = (pred_ids_4d == target_ids_4d) | (~valid_mask_4d)
    accept_prefix = correct.long().cumprod(dim=2) * valid_mask_4d.long()
    return accept_prefix.sum(dim=2).float()


@torch.no_grad()
def _scatter_accepted_prefix(
    predicted_ids: Optional[torch.Tensor],
    target_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    block_index: Optional[torch.Tensor],
    total_blocks: int,
) -> torch.Tensor:
    """Place strict accepted prefixes back into global sampled-anchor order."""
    if block_index is None or predicted_ids is None:
        return target_ids.new_zeros((), dtype=torch.float32)
    correct = (predicted_ids == target_ids) & valid_mask
    accepted = correct.long().cumprod(dim=-1).sum(dim=-1).float()
    return accepted.new_zeros(accepted.shape[0], total_blocks).scatter_(
        1, block_index.expand_as(accepted), accepted
    )


def compute_walk_accepted_length_terms(
    accepted: torch.Tensor,
    anchor_positions: torch.Tensor,
    valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sum accepted lengths and visits along sorted, possibly sparse anchors."""
    accepted_total = 0.0
    visited_count = 0
    rows = zip(
        anchor_positions.detach().cpu().tolist(),
        accepted.detach().cpu().tolist(),
        valid.detach().cpu().tolist(),
    )
    for positions, counts, valid_blocks in rows:
        resume_at = 0
        for position, count, is_valid in zip(positions, counts, valid_blocks):
            if is_valid and position >= resume_at:
                accepted_total += count + 1
                visited_count += 1
                resume_at = position + count + 1
    return accepted.new_tensor((accepted_total, visited_count)).unbind()


def _expected_accepted_length_num(
    acceptance: torch.Tensor,
    supervised: torch.Tensor,
    block_valid: torch.Tensor,
) -> torch.Tensor:
    """Sum ``1 + sum_k prod_{j<=k} a_j`` over valid blocks.

    ``acceptance`` holds the per-slot acceptance event or probability for the
    predicted slots; the chain stops at the first unsupervised slot.
    """

    alive = acceptance.float() * supervised.float()
    return ((1.0 + alive.cumprod(dim=-1).sum(dim=-1)) * block_valid).sum()
