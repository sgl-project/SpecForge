"""Shared eager and chunked loss execution over deferred predictions."""

from dataclasses import dataclass

import torch

from specforge.core.position_weights import DPACE_LOSS_TYPES, dpace_weights

from .base import ObjectiveResult, microbatch_mean
from .losses import LossContext


@dataclass
class Projected:
    output: dict
    teacher: torch.Tensor | None
    full_teacher: torch.Tensor | None
    labels: torch.Tensor
    mask: torch.Tensor
    teacher_mass: torch.Tensor | None


def project_predictions(predictions, loss, weighting):
    selected = predictions.default_loss if loss.kind == "default" else loss
    required = selected.requires | weighting.requires | predictions.requires
    missing = required - predictions.capabilities
    if missing:
        raise ValueError(f"predictions lack required capabilities: {sorted(missing)}")
    output = predictions.project()
    teacher = predictions.teacher.logits() if "teacher_logits" in required else None
    labels, mask = predictions.labels, predictions.mask
    full_teacher, mass = teacher, None
    if predictions.vocab_mask is not None:
        mask = mask * predictions.vocab_mask[teacher.argmax(-1)]
        if selected.registration.contextual:
            mass = teacher.float().softmax(-1)[..., predictions.vocab_mask].sum(-1)
        teacher = teacher[..., predictions.vocab_mask]
        labels = teacher.argmax(-1)
    return Projected(output, teacher, full_teacher, labels, mask, mass)


def evaluate_predictions(
    predictions, loss, weighting, *, projected=None, weights=None, denominator=None
):
    selected = predictions.default_loss if loss.kind == "default" else loss
    projected = projected or project_predictions(predictions, loss, weighting)
    output, teacher = projected.output, projected.teacher
    logits, labels, mask = output["logits"], projected.labels, projected.mask
    full_teacher = projected.full_teacher
    if weights is None:
        weights = weighting(mask.float(), predictions.positions)
    per_token = sum(
        term.coefficient
        * selected(
            output[term.output],
            labels,
            teacher=teacher,
            context=LossContext(
                mask,
                projected.teacher_mass,
                predictions.denominator if denominator is None else None,
            ),
        )
        for term in predictions.terms
    )
    ratios = {}
    if predictions.auxiliary is not None:
        auxiliary = predictions.auxiliary(predictions, output, teacher, weights)
        per_token = per_token + auxiliary.values
        ratios.update(auxiliary.ratios)
    prediction = logits.argmax(-1)
    metric_weights = mask.float()
    target = labels
    if predictions.draft_to_target is not None:
        prediction = prediction + predictions.draft_to_target[prediction]
        target = full_teacher.argmax(-1)
        metric_weights = predictions.mask
    accuracy = (
        ((prediction == target) * metric_weights).sum().detach(),
        metric_weights.sum().detach(),
    )
    ratios[predictions.accuracy_name] = accuracy
    if predictions.report_accuracy:
        ratios["acc"] = accuracy
    denominator = (
        denominator
        if denominator is not None
        else (
            weights.sum().detach()
            if predictions.denominator is None
            else predictions.denominator
        )
    )
    return ObjectiveResult((per_token * weights).sum(), denominator, ratios)


@torch.no_grad()
def chain_weights(predictions, projected, weighting):
    """Compute complete-chain credit, then balance valid anchors per sequence.

    IDs are stable across prediction sets and slices. Sorting by chain and position
    allows EAGLE depths and P-EAGLE COD rows to share the block weighting kernel.
    """
    chain = torch.cat([p.chain_ids.flatten() for p in predictions])
    sequence = torch.cat([p.sequence_ids.flatten() for p in predictions])
    position = torch.cat([p.positions.flatten() for p in predictions])
    supervised = torch.cat([p.mask.flatten().float() for p in predictions])
    # the loss mask also drops targets outside a reduced draft vocabulary
    mask = torch.cat([p.mask.flatten().float() for p in projected])
    probability = torch.cat(
        [
            p.output["logits"]
            .float()
            .softmax(-1)
            .gather(-1, p.labels[..., None])
            .flatten()
            for p in projected
        ]
    )
    # the draft can never propose an unreachable target: the chain stops there
    probability = torch.where(mask > 0, probability, 0.0)
    _, chain = chain.unique(return_inverse=True)
    order = position.argsort(stable=True)
    order = order[chain[order].argsort(stable=True)]
    sorted_chain = chain[order]
    counts = torch.bincount(sorted_chain)
    starts = counts.cumsum(0) - counts
    offsets = torch.arange(chain.numel(), device=chain.device) - starts[sorted_chain]
    sorted_position = position[order]
    duplicate = (sorted_chain[1:] == sorted_chain[:-1]) & (
        sorted_position[1:] == sorted_position[:-1]
    )
    if duplicate.any():
        raise ValueError("prediction chains must have unique positions")
    chain_sequence = sequence[order[starts]]
    if (sequence != chain_sequence[chain]).any():
        raise ValueError("a prediction chain cannot span sequences")
    dense_mask = mask.new_zeros((counts.numel(), int(counts.max())))
    dense_supervised = torch.zeros_like(dense_mask)
    dense_probability = torch.ones_like(dense_mask)
    dense_mask[sorted_chain, offsets] = mask[order]
    dense_supervised[sorted_chain, offsets] = supervised[order]
    dense_probability[sorted_chain, offsets] = probability[order]
    credit = dpace_weights(
        dense_probability,
        dense_supervised,
        alpha=weighting.alpha,
        mode=weighting.kind,
    )
    valid = dense_mask.any(-1)
    _, sequence_index = chain_sequence.unique(return_inverse=True)
    anchors = torch.zeros_like(sequence_index, dtype=mask.dtype).scatter_add(
        0, sequence_index, valid.float()
    )
    scale = valid.float() / anchors[sequence_index].clamp_min(1)
    weights = torch.empty_like(mask)
    weights[order] = (credit * dense_mask * scale[:, None])[sorted_chain, offsets]
    # One contribution per valid chain; the total denominator is valid sequences.
    denominator = torch.zeros_like(mask)
    denominator[order[starts]] = scale
    sizes = [p.mask.numel() for p in predictions]
    return list(zip(weights.split(sizes), denominator.split(sizes)))


def combine_results(results):
    ratios = {}
    for result in results:
        for key, pair in result.ratios.items():
            ratios[key] = (
                tuple(a + b for a, b in zip(ratios[key], pair))
                if key in ratios
                else pair
            )
    return ObjectiveResult(
        sum(r.numerator for r in results), sum(r.denominator for r in results), ratios
    )


def evaluate_chunks(predictions, loss, weighting, chunk_size):
    selected = predictions.default_loss if loss.kind == "default" else loss
    length = predictions.hidden.shape[0]
    # Contextual losses use whole-set statistics; preserve that reduction.
    if not chunk_size or chunk_size >= length or selected.registration.contextual:
        return evaluate_predictions(predictions, loss, weighting)
    from torch.utils.checkpoint import checkpoint

    results = []
    for start in range(0, length, chunk_size):
        item = predictions.select(slice(start, start + chunk_size))
        if torch.is_grad_enabled():
            result = checkpoint(
                evaluate_predictions, item, loss, weighting, use_reentrant=False
            )
        else:
            result = evaluate_predictions(item, loss, weighting)
        results.append(result)
    result = combine_results(results)
    # A supplied denominator belongs to the whole prediction set, not each chunk.
    if predictions.denominator is not None:
        result.denominator = predictions.denominator
    return result


def execute_batch(batch, loss, weighting, *, training):
    if weighting.kind in DPACE_LOSS_TYPES:
        predictions = list(batch.predictions)
        if not predictions:
            raise ValueError("protocol must produce at least one prediction set")
        projected = [project_predictions(p, loss, weighting) for p in predictions]
        weighted = chain_weights(predictions, projected, weighting)
        results = [
            evaluate_predictions(
                p,
                loss,
                weighting,
                projected=q,
                weights=w.reshape_as(p.mask),
                denominator=d.sum(),
            )
            for p, q, (w, d) in zip(predictions, projected, weighted)
        ]
    else:
        results = [
            evaluate_chunks(item, loss, weighting, batch.chunk_size)
            for item in batch.predictions
        ]
    if not results:
        raise ValueError("protocol must produce at least one prediction set")
    result = combine_results(results)
    if batch.reduction == "sum_means" and weighting.kind not in DPACE_LOSS_TYPES:
        return ObjectiveResult(
            sum(item.loss for item in results),
            results[0].denominator.new_ones(()),
            result.ratios,
        )
    if batch.reduction == "global_microbatch":
        return microbatch_mean(result, global_denominator=training)
    if batch.reduction not in ("tokens", "sum_means"):
        raise ValueError(f"unknown protocol reduction: {batch.reduction!r}")
    return result
