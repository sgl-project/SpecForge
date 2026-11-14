"""Detached, chunked diagnostics for anchored DFlash predictions."""

from functools import partial

import torch

from specforge.algorithms.metrics.dflash import (
    dflash2_selector_diagnostics,
    dflash_greedy_diagnostics,
    dflash_teacher_terms,
    dflash_unary_diagnostics,
    reduce_dflash_metric_terms,
)
from specforge.algorithms.metrics.dflash_terms import DFlashMetricTerms
from specforge.algorithms.metrics.report import format_block_metrics
from specforge.losses.position_weights import DPACE_LOSS_TYPES


def block_diagnostics(
    runtime,
    predictions,
    blocks,
    *,
    enabled,
    teacher_metrics=False,
    top_k=16,
    selector=None,
):
    """Build a deferred report; no projection occurs when diagnostics are disabled."""
    if not enabled:
        return None
    return partial(
        collect_block_metrics,
        runtime,
        predictions,
        blocks,
        top_k=selector.top_k if selector is not None else top_k,
        selector=selector,
        teacher_metrics=teacher_metrics,
    )


def diagnostic_weights(runtime, predictions, slices):
    """Reuse objective weighting, including complete-chain D-PACE statistics."""
    from .execution import ChainStatistics, chain_weights

    weighting = predictions.plan.terms[0].weighting or runtime.weighting
    if weighting.kind in DPACE_LOSS_TYPES:
        probabilities = []
        for part in slices:
            p = predictions.select(part)
            logits = p.project()["logits"].float()
            probabilities.append(
                logits.softmax(-1).gather(-1, p.labels[..., None]).squeeze(-1)
            )
        statistics = ChainStatistics(predictions.mask, torch.cat(probabilities))
        return chain_weights([predictions], [statistics], weighting)[0][0].view_as(
            predictions.mask
        )
    return weighting(predictions.mask.float(), predictions.positions)


@torch.no_grad()
def collect_block_metrics(
    runtime, predictions, blocks, *, top_k, selector, teacher_metrics=True
):
    """Project bounded chunks and return additive ratios/counts without changing loss."""
    units, size = predictions.mask.shape
    chunk_size = runtime.execution.chunk_size or 128
    slices = [
        slice(start, min(start + chunk_size, units))
        for start in range(0, units, chunk_size)
    ]
    weights = diagnostic_weights(runtime, predictions, slices)
    terms = None
    for part in slices:
        p = predictions.select(part)
        hidden = p.hidden.unsqueeze(0)
        labels, mask = p.labels.unsqueeze(0), p.mask.unsqueeze(0)
        logits = p.project()["logits"].float().unsqueeze(0)
        width = min(top_k, logits.shape[-1])
        unary = dflash_unary_diagnostics(logits, labels, top_k=width)
        loss_weights = weights[part].unsqueeze(0)
        position_den = mask.sum((0, 1))
        selected = None
        if selector is not None:
            predecessors = blocks.predecessors.reshape(units, size)[part].unsqueeze(0)
            selected = dflash2_selector_diagnostics(
                candidate_selector=selector,
                unary=unary,
                hidden=hidden,
                target_ids=labels,
                predecessor_ids=predecessors,
                loss_weights=loss_weights,
                weight_mask=mask,
            )
        greedy = dflash_greedy_diagnostics(
            candidate_selector=selector,
            unary=unary,
            hidden=hidden,
            target_ids=labels,
            weight_mask=mask,
            position_den=position_den,
        )
        teacher_logits = (
            p.teacher.logits().unsqueeze(0)
            if teacher_metrics and p.teacher is not None
            else None
        )
        teacher = dflash_teacher_terms(
            lambda value: value,
            unary=unary,
            greedy_ids=greedy.selected_ids,
            aligned_target_hidden=teacher_logits,
            weight_mask=mask,
            position_den=position_den,
        )
        chunk = reduce_dflash_metric_terms(
            unary=unary,
            selector=selected,
            greedy=greedy,
            teacher=teacher,
            target_ids=labels,
            weight_mask=mask,
            loss_weights=loss_weights,
            position_den=position_den,
            block_index=torch.arange(part.start, part.stop, device=hidden.device)[None],
            total_blocks=units,
        )
        terms = (
            chunk
            if terms is None
            else DFlashMetricTerms(*(a + b for a, b in zip(terms, chunk)))
        )
    terms = terms._replace(
        unary_block_accepted=terms.unary_block_accepted.view_as(blocks.anchors),
        selector_block_accepted=(
            terms.selector_block_accepted.view_as(blocks.anchors)
            if selector is not None
            else terms.selector_block_accepted
        ),
    )
    return format_block_metrics(
        terms,
        blocks.weights,
        blocks.anchors,
        width,
        has_selector=selector is not None,
        has_teacher=teacher_metrics and predictions.teacher is not None,
    )


def diagnostic_requirements(objective):
    """Request teacher capture only when the objective enables teacher diagnostics."""
    if type(objective.teacher_metrics) is not bool:
        raise ValueError("teacher_metrics must be boolean")
    return frozenset({"teacher_logits"}) if objective.teacher_metrics else frozenset()


def prediction_diagnostics(
    runtime, predictions, blocks, measure, *, enabled, slots=slice(None)
):
    """Defer algorithm metrics over whole prediction blocks until a logging step."""
    if not enabled:
        return None
    return partial(
        collect_prediction_metrics, runtime, predictions, blocks, measure, slots=slots
    )


@torch.no_grad()
def collect_prediction_metrics(runtime, predictions, blocks, measure, *, slots):
    """Merge additive chunk reports before computing walks across sampled anchors.

    The measure callback receives projected outputs, labels, masks and optional
    teacher logits. It returns ratio pairs and named greedy token predictions.
    """
    from specforge.algorithms.metrics.blocks import compute_walk_accepted_length_terms

    from .execution import prediction_slices

    ratios, prefixes = {}, {}
    slices = list(prediction_slices(predictions, runtime.execution.chunk_size or 128))
    weights = diagnostic_weights(runtime, predictions, slices)
    for part in slices:
        chunk = predictions.select(part)
        teacher = chunk.teacher.logits() if chunk.teacher is not None else None
        values, tokens = measure(
            chunk.project(), chunk.labels, chunk.mask, teacher, weights[part]
        )
        for name, pair in values.items():
            ratios[name] = (
                tuple(a + b for a, b in zip(ratios[name], pair))
                if name in ratios
                else pair
            )
        for name, ids in tokens.items():
            accepted = (ids[..., slots] == chunk.labels[..., slots]) & (
                chunk.mask[..., slots] > 0
            )
            prefixes.setdefault(name, []).append(
                accepted.long().cumprod(-1).sum(-1).float()
            )
    valid = (predictions.mask[..., slots] > 0).any(-1).view_as(blocks.anchors)
    for name, chunks in prefixes.items():
        ratios[f"{name}/walk_accepted_length"] = compute_walk_accepted_length_terms(
            torch.cat(chunks).view_as(blocks.anchors), blocks.anchors, valid
        )
    return ratios, {}
