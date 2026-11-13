"""Shared eager and chunked loss execution over deferred predictions."""

from dataclasses import dataclass

import torch

from specforge.core.position_weights import DPACE_LOSS_TYPES, dpace_weights

from .base import ObjectiveResult
from .execution_options import LossExecution
from .implementations import ExecutionRequest, TermInputs, select_implementation
from .losses import LossContext
from .reductions import PlanResult, combine_results


@dataclass
class Projected:
    output: dict
    teacher: torch.Tensor | None
    teacher_ids: torch.Tensor | None
    labels: torch.Tensor
    mask: torch.Tensor
    teacher_mass: torch.Tensor | None


def selected_losses(predictions, loss):
    default = predictions.default_loss if loss.kind == "default" else loss
    return tuple(
        term.loss if term.loss is not None and term.loss.kind != "default" else default
        for term in predictions.plan.terms
    )


def prepare_supervision(predictions, loss, weighting, execution):
    losses = selected_losses(predictions, loss)
    required = weighting.requires | predictions.requires
    for term, selected in zip(predictions.plan.terms, losses):
        required |= selected.requires
        if term.weighting is not None:
            required |= term.weighting.requires
    missing = required - predictions.capabilities
    if missing:
        raise ValueError(f"predictions lack required capabilities: {sorted(missing)}")
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
                if any(selected.registration.contextual for selected in losses):
                    mass = (
                        teacher.float().softmax(-1)[..., predictions.vocab_mask].sum(-1)
                    )
                teacher = teacher[..., predictions.vocab_mask]
    if predictions.vocab_mask is not None:
        mask = mask * predictions.vocab_mask[teacher_ids]
        labels = teacher.argmax(-1)
    return Projected({}, teacher, teacher_ids, labels, mask, mass)


def project_predictions(predictions, loss, weighting, execution=LossExecution()):
    projected = prepare_supervision(predictions, loss, weighting, execution)
    projected.output = predictions.project()
    return projected


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
    p = predictions
    projected = projected or prepare_supervision(p, loss, weighting, execution)
    losses = selected_losses(p, loss)
    inputs = []
    for term, selected in zip(p.plan.terms, losses):
        mask = projected.mask
        if term.mask is not None:
            mask = term.mask
            if p.vocab_mask is not None:
                mask = mask * p.vocab_mask[projected.teacher_ids]
        term_weights = weights
        if term_weights is None:
            term_weights = (term.weighting or weighting)(mask.float(), p.positions)
        fixed = term.denominator if term.denominator is not None else p.denominator
        term_denominator = denominator if denominator is not None else fixed
        if term_denominator is None:
            term_denominator = term_weights.sum().detach()
        inputs.append(
            TermInputs(
                selected,
                mask,
                term_weights,
                term_denominator,
                LossContext(
                    mask, projected.teacher_mass, fixed if denominator is None else None
                ),
            )
        )
    numerators, predicted = {}, {}

    def outputs():
        if not projected.output:
            projected.output.update(p.project())
        return projected.output

    request = ExecutionRequest(
        p, tuple(inputs), projected.labels, projected.teacher, outputs, projected.output
    )

    def reference(index):
        term, item = p.plan.terms[index], inputs[index]
        values = item.loss(
            outputs()[term.output],
            projected.labels,
            teacher=projected.teacher,
            context=item.context,
        )
        numerators[index] = (values * item.weights).sum()

    def execute(group):
        entry = select_implementation(request, group) if execution.fused else None
        if entry is None:
            return False
        result = entry.compute(request, group)
        if set(result.numerators) != set(group):
            raise ValueError(
                f"{entry.name} did not return exactly the requested loss terms"
            )
        numerators.update(result.numerators)
        predicted.update(result.predictions)
        return True

    for group in p.plan.groups:
        if not execute(group):
            for index in group:
                if len(group) == 1 or not execute((index,)):
                    reference(index)

    ratios = {}
    if p.auxiliary is not None:
        if not p.plan.shared_reduction:
            raise ValueError(
                "Algorithm auxiliary callbacks require shared supervision and reduction"
            )
        auxiliary = p.auxiliary(p, outputs(), projected.teacher, inputs[0].weights)
        # Auxiliary values already include their own coefficients.
        auxiliary_result = ObjectiveResult(
            (auxiliary.values * inputs[0].weights).sum(), inputs[0].denominator
        )
        ratios.update(auxiliary.ratios)
    else:
        auxiliary_result = None

    prediction = predicted.get("logits")
    if prediction is None:
        prediction = outputs()["logits"].argmax(-1)
    metric_weights, target = projected.mask.float(), projected.labels
    if p.draft_to_target is not None:
        prediction = prediction + p.draft_to_target[prediction]
        target, metric_weights = projected.teacher_ids, p.mask
    accuracy = (
        ((prediction == target) * metric_weights).sum().detach(),
        metric_weights.sum().detach(),
    )
    ratios[p.accuracy_name] = accuracy
    if p.report_accuracy:
        ratios["acc"] = accuracy
    terms = tuple(
        ObjectiveResult(numerators[i], item.denominator)
        for i, item in enumerate(inputs)
    )
    coefficients = tuple(term.coefficient for term in p.plan.terms)
    if auxiliary_result is not None:
        terms += (auxiliary_result,)
        coefficients += (1.0,)
    return PlanResult(terms, coefficients, p.plan.shared_reduction, ratios)


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
    contextual = any(
        selected.registration.contextual
        for selected in selected_losses(predictions, loss)
    )
    # Contextual losses and local DSpine alignment need whole-set statistics.
    if (
        contextual
        or not predictions.chunk_safe
        or not predictions.projection.chunk_safe
    ):
        chunk_size = 0
    if execution.trim and weighting.kind not in DPACE_LOSS_TYPES and not contextual:
        keep = predictions.mask.any(-1)
        for term in predictions.plan.terms:
            if term.mask is not None:
                keep |= term.mask.any(-1)
        predictions = predictions.select(keep.nonzero().flatten())
    from torch.utils.checkpoint import checkpoint

    results = []
    for part in prediction_slices(predictions, chunk_size):
        item = predictions.select(part)
        kwargs = dict(
            execution=execution,
            weights=None if weights is None else weights[part],
            denominator=None if denominator is None else denominator[part].sum(),
        )
        if (
            chunk_size
            and execution.checkpoint
            and predictions.projection.recompute_safe
            and torch.is_grad_enabled()
        ):
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
    if denominator is None:
        for index, term in enumerate(predictions.plan.terms):
            fixed = (
                term.denominator
                if term.denominator is not None
                else predictions.denominator
            )
            if fixed is not None:
                result.terms[index].denominator = fixed
        if predictions.auxiliary is not None and predictions.denominator is not None:
            result.terms[-1].denominator = predictions.denominator
    return result


def execute_batch(batch, loss, weighting, *, training, execution=LossExecution()):
    chunk_size = (
        batch.chunk_size if execution.chunk_size is None else execution.chunk_size
    )
    if weighting.kind in DPACE_LOSS_TYPES:
        predictions = list(batch.predictions)
        if not predictions:
            raise ValueError("protocol must produce at least one prediction set")
        if any(not p.plan.shared_reduction for p in predictions):
            raise ValueError(
                "D-PACE currently requires shared loss masks and weighting"
            )
        if any(
            not p.projection.chunk_safe or not p.projection.recompute_safe
            for p in predictions
        ):
            chunk_size = 0
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
    if batch.reduction == "sum_means" and weighting.kind not in DPACE_LOSS_TYPES:
        # Prediction sets can have distinct plans; only their metrics are additive.
        ratios = {}
        for item in results:
            for name, pair in item.ratios.items():
                ratios[name] = (
                    tuple(a + b for a, b in zip(ratios[name], pair))
                    if name in ratios
                    else pair
                )
        return ObjectiveResult(
            sum(item.loss for item in results),
            results[0].terms[0].denominator.new_ones(()),
            ratios,
        )
    result = combine_results(results)
    if batch.reduction == "global_microbatch":
        return result.objective_result(global_denominator=training, normalize=True)
    if batch.reduction not in ("tokens", "sum_means"):
        raise ValueError(f"unknown protocol reduction: {batch.reduction!r}")
    if not result.shared_reduction:
        raise ValueError(
            "Independent term reductions require sum_means or global_microbatch"
        )
    return result.objective_result()
