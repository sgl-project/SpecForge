"""Shared eager and chunked loss execution over deferred predictions."""

from dataclasses import dataclass

import torch

from specforge.core.position_weights import DPACE_LOSS_TYPES, dpace_weights

from .base import ObjectiveResult
from .execution_options import LossExecution
from .implementations.registry import ExecutionRequest, ExecutionSelection, TermInputs
from .losses import LossContext
from .reductions import PlanResult, combine_results


@dataclass
class Projected:
    """Prepared supervision and a per-evaluation projection cache.

    Attributes:
        output: Named draft outputs, initially empty for lazy projection.
        teacher: Optional teacher logits, restricted to the draft vocabulary.
        teacher_ids: Optional teacher argmax IDs in the full target vocabulary.
        labels: Effective loss targets, shaped [units, positions].
        mask: Effective supervision mask, including vocabulary eligibility.
        teacher_mass: Optional teacher probability mass retained by the draft
            vocabulary, needed by contextual losses.
    """

    output: dict
    teacher: torch.Tensor | None
    teacher_ids: torch.Tensor | None
    labels: torch.Tensor
    mask: torch.Tensor
    teacher_mass: torch.Tensor | None


def selected_losses(predictions, loss):
    """Resolve each term's loss against prediction and runtime defaults.

    Args:
        predictions: Prediction set containing the plan and algorithm default.
        loss: Runtime TokenLoss selection; kind="default" uses the algorithm default.

    Returns:
        A tuple of concrete TokenLoss selections in plan term order.
    """
    default = predictions.default_loss if loss.kind == "default" else loss
    return tuple(
        term.loss if term.loss is not None and term.loss.kind != "default" else default
        for term in predictions.plan.terms
    )


def prepare_supervision(predictions, loss, weighting, execution):
    """Resolve targets and teacher data without projecting draft logits.

    Args:
        predictions: Deferred prediction set and its supervision inputs.
        loss: Runtime loss selection, overridden by explicit per-term losses.
        weighting: Runtime position-weighting rule.
        execution: Execution options, including optional tiled teacher projection.

    Returns:
        Projected supervision with an empty draft output cache. Teacher logits
        are computed only when required; vocabulary mapping adjusts labels/masks.

    Raises:
        ValueError: Required capabilities or compact-teacher inputs are missing.
    """
    losses = selected_losses(predictions, loss)
    required = weighting.requires | predictions.requires
    for term, selected in zip(predictions.plan.terms, losses):
        required |= selected.requires
        if term.weighting is not None:
            required |= term.weighting.requires
    for term in predictions.plan.auxiliary_terms:
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
    """Prepare supervision and materialize the ordinary draft outputs.

    Args:
        predictions: Deferred prediction set to evaluate.
        loss: Runtime mathematical loss selection.
        weighting: Runtime position-weighting rule.
        execution: Teacher-projection and execution options.

    Returns:
        Projected data containing both resolved supervision and draft outputs.
        This path is also used to obtain complete-chain weighting statistics.
    """
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
    selection=None,
    prediction_index=0,
):
    """Execute a prediction set and retain separate statistics per term.

    Args:
        predictions: One complete set or chunk of aligned prediction units.
        loss: Runtime loss selection; explicit term losses take precedence.
        weighting: Default position-weighting rule.
        projected: Optional previously prepared/projected data to reuse.
        weights: Optional externally prepared weights aligned with the mask,
            used by complete-chain weighting.
        denominator: Optional scalar contribution overriding local denominators.
        execution: Options controlling optimized implementation selection.
        selection: Run-local resolved implementations and resume checks.
        prediction_index: Stable index of this set within the protocol batch.

    Returns:
        PlanResult with weighted, unnormalized term sums and metric counts.
        Joint ranges may fall back to independent execution. No chunk means
        or term coefficients are combined here.

    Raises:
        ValueError: Inputs lack required capabilities, a kernel returns the
            wrong term indices, or auxiliary reduction contracts are unsupported.
        RuntimeError: A selected kernel fails; errors are not silently retried.
    """
    p = predictions
    selection = selection or ExecutionSelection()
    projected = projected or prepare_supervision(p, loss, weighting, execution)
    losses = selected_losses(p, loss)
    default_weights = (
        weighting(projected.mask.float(), p.positions) if weights is None else weights
    )
    inputs = []
    for term, selected in zip(
        p.plan.all_terms, losses + (None,) * len(p.plan.auxiliary_terms)
    ):
        mask = projected.mask
        if term.mask is not None:
            mask = term.mask
            if p.vocab_mask is not None:
                mask = mask * p.vocab_mask[projected.teacher_ids]
        term_weights = default_weights
        if term.mask is not None or term.weighting is not None:
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
    numerators, predicted, statistics = {}, {}, {}

    def outputs():
        if not projected.output:
            projected.output.update(p.project())
        return projected.output

    request = ExecutionRequest(
        p, tuple(inputs), projected.labels, projected.teacher, outputs, projected.output
    )

    def reference(index):
        term, item = p.plan.terms[index], inputs[index]
        result = item.loss.evaluate(
            outputs()[term.output],
            projected.labels,
            teacher=projected.teacher,
            context=item.context,
        )
        numerators[index] = (result.values * item.weights).sum()
        if p.auxiliary is not None:
            statistics[index] = {
                name: value.detach() for name, value in result.statistics.items()
            }

    def execute(group):
        entry = (
            selection.resolve(request, group, prediction_index)
            if execution.fused
            else None
        )
        if entry is None:
            return False
        if not p.hidden.numel():
            # Keep empty-rank gradients without launching a zero-sized kernel.
            for index in group:
                reference(index)
            return True
        result = entry.compute(request, group)
        if set(result.numerators) != set(group):
            raise ValueError(
                f"{entry.name} did not return exactly the requested loss terms"
            )
        numerators.update(result.numerators)
        predicted.update(result.predictions)
        if not result.statistics.keys() <= set(group):
            raise ValueError("Implementation statistics reference an unexecuted term")
        if p.auxiliary is not None:
            statistics.update(
                {
                    index: {name: value.detach() for name, value in values.items()}
                    for index, values in result.statistics.items()
                }
            )
        return True

    for group in p.plan.groups:
        if not execute(group):
            for index in group:
                if len(group) == 1 or not execute((index,)):
                    reference(index)

    ratios = {}
    if p.auxiliary is not None:
        auxiliary = p.auxiliary(
            p, outputs(), projected.teacher, default_weights, statistics
        )
        names = {term.output for term in p.plan.auxiliary_terms}
        if auxiliary.values.keys() != names:
            raise ValueError("Auxiliary results must match the declared term names")
        for index, term in enumerate(p.plan.auxiliary_terms, len(p.plan.terms)):
            values = auxiliary.values[term.output]
            if values.shape != p.mask.shape:
                raise ValueError("Auxiliary values must align with prediction units")
            numerators[index] = (values * inputs[index].weights).sum()
        ratios.update(auxiliary.ratios)

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
    coefficients = tuple(term.coefficient for term in p.plan.all_terms)
    return PlanResult(terms, coefficients, p.plan.shared_reduction, ratios)


@dataclass
class ChainStatistics:
    """Detached per-position masks and acceptance probabilities."""

    mask: torch.Tensor
    probability: torch.Tensor


@torch.no_grad()
def chain_statistics(projected):
    """Return detached chain masks/probabilities from resolved logits and labels."""
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
    """Compute complete-chain weights before any differentiable chunk pass.

    Args:
        predictions: Aligned prediction sets with chain, sequence and position IDs.
        projected: Matching Projected objects or detached ChainStatistics.
        weighting: D-PACE weighting configuration.

    Returns:
        One pair of flattened (weights, denominator contributions) per set.
        Chain contributions sum to the valid-sequence denominator.

    Raises:
        ValueError: Chain positions repeat or a chain spans multiple sequences.
    """
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
    """Return unit-axis slices, including one slice for an empty input.

    Args:
        predictions: Prediction set whose hidden first dimension counts units.
        chunk_size: Maximum units per slice; zero means the entire set.

    Returns:
        A list of slices preserving whole blocks/tokens and empty gradient paths.
    """
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
    selection=None,
    prediction_index=0,
):
    """Evaluate chunks and merge each term before denominator normalization.

    Args:
        predictions: Aligned prediction set and optional per-term overrides.
        loss: Runtime loss selection.
        weighting: Default weighting rule.
        chunk_size: Requested units per chunk; zero disables splitting.
        execution: Trimming, checkpointing and kernel-selection options.
        weights: Optional externally computed per-position weights.
        denominator: Optional per-position denominator contributions, summed
            within each chunk and then merged across chunks.
        selection: Run-local resolved implementations and resume checks.
        prediction_index: Stable index of this set within the protocol batch.

    Returns:
        PlanResult with additive chunk statistics merged and fixed whole-set
        denominators restored once. Contextual/unsafe projections stay unsplit;
        unsafe replay disables checkpointing.
    """
    selection = selection or ExecutionSelection()
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
        for term in predictions.plan.all_terms:
            if term.mask is not None:
                keep |= term.mask.any(-1)
        predictions = predictions.select(keep.nonzero().flatten())
    from torch.utils.checkpoint import checkpoint

    results = []
    for part in prediction_slices(predictions, chunk_size):
        item = predictions.select(part)
        kwargs = dict(
            execution=execution,
            selection=selection,
            prediction_index=prediction_index,
            weights=None if weights is None else weights[part],
            denominator=None if denominator is None else denominator[part].sum(),
        )
        if (
            0 < chunk_size < predictions.hidden.shape[0]
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
        for index, term in enumerate(predictions.plan.all_terms):
            fixed = (
                term.denominator
                if term.denominator is not None
                else predictions.denominator
            )
            if fixed is not None:
                result.terms[index].denominator = fixed
    return result


def execute_batch(
    batch, loss, weighting, *, training, execution=LossExecution(), selection=None
):
    """Execute a protocol's prediction batch under its reduction contract.

    Args:
        batch: PredictionBatch with prediction sets, reduction and chunk default.
        loss: Runtime mathematical loss selection.
        weighting: Runtime position weights or complete-chain weighting policy.
        training: Whether global_microbatch synchronizes denominators across ranks.
        execution: Overrides for chunking, checkpointing and optimized execution.
        selection: Persistent implementation choices; standalone calls use a local selection.

    Returns:
        ObjectiveResult containing the trainer loss numerator/denominator and
        metric counts. D-PACE weights are prepared over complete chains.

    Raises:
        ValueError: The batch is empty, its reduction is unknown, or independent
            term reductions are requested with an unsupported batch contract.
    """
    selection = selection or ExecutionSelection()
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
                    selection=selection,
                    prediction_index=index,
                    weights=w.reshape_as(p.mask),
                    denominator=d.sum(),
                )
                for index, (p, q, (w, d)) in enumerate(
                    zip(predictions, projected, weighted)
                )
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
                    selection=selection,
                    prediction_index=index,
                    weights=w.reshape_as(p.mask),
                    denominator=d.reshape_as(p.mask),
                )
                for index, (p, (w, d)) in enumerate(zip(predictions, weighted))
            ]
    else:
        results = [
            evaluate_chunks(
                item,
                loss,
                weighting,
                chunk_size,
                execution=execution,
                selection=selection,
                prediction_index=index,
            )
            for index, item in enumerate(batch.predictions)
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
