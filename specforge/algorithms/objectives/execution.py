"""Shared eager and chunked loss execution over deferred predictions."""

from dataclasses import dataclass

import torch

from specforge.core.position_weights import DPACE_LOSS_TYPES, dpace_weights

from .base import ObjectiveResult, microbatch_mean
from .execution_options import LossExecution
from .losses import LossContext


@dataclass
class Projected:
    output: dict
    teacher: torch.Tensor | None
    teacher_ids: torch.Tensor | None
    labels: torch.Tensor
    mask: torch.Tensor
    teacher_mass: torch.Tensor | None


def project_predictions(predictions, loss, weighting, execution=LossExecution()):
    selected = predictions.default_loss if loss.kind == "default" else loss
    required = selected.requires | weighting.requires | predictions.requires
    missing = required - predictions.capabilities
    if missing:
        raise ValueError(f"predictions lack required capabilities: {sorted(missing)}")
    output = predictions.project()
    teacher = teacher_ids = mass = None
    labels, mask = predictions.labels, predictions.mask
    if "teacher_logits" in required:
        if execution.teacher_chunk_size:
            teacher, teacher_ids, mass = predictions.teacher.compact(
                predictions.vocab_mask, execution.teacher_chunk_size
            )
        else:
            teacher = predictions.teacher.logits()
            teacher_ids = teacher.argmax(-1)
            if predictions.vocab_mask is not None:
                if selected.registration.contextual:
                    mass = (
                        teacher.float().softmax(-1)[..., predictions.vocab_mask].sum(-1)
                    )
                teacher = teacher[..., predictions.vocab_mask]
    if predictions.vocab_mask is not None:
        mask = mask * predictions.vocab_mask[teacher_ids]
        labels = teacher.argmax(-1)
    return Projected(output, teacher, teacher_ids, labels, mask, mass)


def evaluate_predictions(
    predictions,
    loss,
    weighting,
    *,
    projected=None,
    weights=None,
    denominator=None,
    execution=LossExecution(),
):
    selected = predictions.default_loss if loss.kind == "default" else loss
    if (
        execution.fused
        and projected is None
        and predictions.fused_evaluator is not None
    ):
        result = predictions.fused_evaluator(
            predictions, selected, weighting, weights=weights, denominator=denominator
        )
        if result is not None:
            return result
    projected = projected or project_predictions(
        predictions, loss, weighting, execution
    )
    output, teacher = projected.output, projected.teacher
    logits, labels, mask = output["logits"], projected.labels, projected.mask
    if weights is None:
        weights = weighting(mask.float(), predictions.positions)
    per_token = sum(
        term.coefficient
        * selected(
            output[term.output],
            labels,
            teacher=teacher,
            fused=execution.fused,
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
        target = projected.teacher_ids
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


@dataclass
class ChainStatistics:
    mask: torch.Tensor
    probability: torch.Tensor


@torch.no_grad()
def chain_statistics(projected):
    probability = (
        projected.output["logits"]
        .float()
        .softmax(-1)
        .gather(-1, projected.labels[..., None])
        .squeeze(-1)
    )
    return ChainStatistics(projected.mask, probability)


@torch.no_grad()
def chain_weights(predictions, projected, weighting):
    """Balance full acceptance chains without retaining vocabulary-sized tensors."""
    statistics = [
        value if isinstance(value, ChainStatistics) else chain_statistics(value)
        for value in projected
    ]
    if not any(p.mask.numel() for p in predictions):
        return [(p.mask.float(), p.mask.float()) for p in predictions]
    chain = torch.cat([p.chain_ids.flatten() for p in predictions])
    sequence = torch.cat([p.sequence_ids.flatten() for p in predictions])
    position = torch.cat([p.positions.flatten() for p in predictions])
    supervised = torch.cat([p.mask.flatten().float() for p in predictions])
    mask = torch.cat([p.mask.flatten().float() for p in statistics])
    probability = torch.cat([p.probability.flatten() for p in statistics])
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


def prediction_slices(predictions, chunk_size):
    length = predictions.hidden.shape[0]
    size = chunk_size or max(length, 1)
    # An empty projection keeps zero gradients connected to the model.
    return [slice(start, start + size) for start in range(0, max(length, 1), size)]


def evaluate_chunks(
    predictions,
    loss,
    weighting,
    chunk_size,
    *,
    execution=LossExecution(),
    weights=None,
    denominator=None,
):
    selected = predictions.default_loss if loss.kind == "default" else loss
    # Contextual losses and local DSpine alignment need whole-set statistics.
    if selected.registration.contextual or not predictions.chunk_safe:
        chunk_size = 0
    if (
        execution.trim
        and weighting.kind not in DPACE_LOSS_TYPES
        and not selected.registration.contextual
    ):
        predictions = predictions.select(predictions.mask.any(-1).nonzero().flatten())
    from torch.utils.checkpoint import checkpoint

    results = []
    for part in prediction_slices(predictions, chunk_size):
        item = predictions.select(part)
        kwargs = dict(
            execution=execution,
            weights=None if weights is None else weights[part],
            denominator=None if denominator is None else denominator[part].sum(),
        )
        if chunk_size and execution.checkpoint and torch.is_grad_enabled():
            result = checkpoint(
                evaluate_predictions,
                item,
                loss,
                weighting,
                use_reentrant=False,
                **kwargs,
            )
        else:
            result = evaluate_predictions(item, loss, weighting, **kwargs)
        results.append(result)
    result = combine_results(results)
    if predictions.denominator is not None and denominator is None:
        result.denominator = predictions.denominator
    return result


def execute_batch(batch, loss, weighting, *, training, execution=LossExecution()):
    chunk_size = (
        batch.chunk_size if execution.chunk_size is None else execution.chunk_size
    )
    if weighting.kind in DPACE_LOSS_TYPES:
        predictions = list(batch.predictions)
        if not predictions:
            raise ValueError("protocol must produce at least one prediction set")
        if not chunk_size:
            projected = [
                project_predictions(p, loss, weighting, execution) for p in predictions
            ]
            weighted = chain_weights(predictions, projected, weighting)
            results = [
                evaluate_predictions(
                    p,
                    loss,
                    weighting,
                    projected=q,
                    execution=execution,
                    weights=w.reshape_as(p.mask),
                    denominator=d.sum(),
                )
                for p, q, (w, d) in zip(predictions, projected, weighted)
            ]
        else:
            statistics = []
            device = predictions[0].hidden.device
            devices = [device.index] if device.type == "cuda" else []
            # This pass collects detached chain probabilities. Restore RNG so the
            # differentiable pass sees the same stochastic projector outputs.
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                for prediction in predictions:
                    pieces = [
                        chain_statistics(
                            project_predictions(
                                prediction.select(part), loss, weighting, execution
                            )
                        )
                        for part in prediction_slices(prediction, chunk_size)
                    ]
                    statistics.append(
                        ChainStatistics(
                            torch.cat([p.mask for p in pieces]),
                            torch.cat([p.probability for p in pieces]),
                        )
                    )
            weighted = chain_weights(predictions, statistics, weighting)
            results = [
                evaluate_chunks(
                    p,
                    loss,
                    weighting,
                    chunk_size,
                    execution=execution,
                    weights=w.reshape_as(p.mask),
                    denominator=d.reshape_as(p.mask),
                )
                for p, (w, d) in zip(predictions, weighted)
            ]
    else:
        results = [
            evaluate_chunks(item, loss, weighting, chunk_size, execution=execution)
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
