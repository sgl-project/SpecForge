"""Shared eager loss execution; protocols never select a projection backend."""

from .base import ObjectiveResult, microbatch_mean


def evaluate_predictions(predictions, loss, weighting):
    selected = predictions.default_loss if loss.kind == "default" else loss
    required = selected.requires | weighting.requires | predictions.requires
    missing = required - predictions.capabilities
    if missing:
        raise ValueError(f"predictions lack required capabilities: {sorted(missing)}")
    output = predictions.project()
    logits = output["logits"]
    teacher = predictions.teacher.logits() if "teacher_logits" in required else None
    labels, mask = predictions.labels, predictions.mask
    full_teacher = teacher
    if predictions.vocab_mask is not None:
        mask = mask * predictions.vocab_mask[teacher.argmax(-1)]
        teacher = teacher[..., predictions.vocab_mask]
        labels = teacher.argmax(-1)
    weights = weighting(mask.float(), predictions.positions)
    per_token = sum(
        term.coefficient * selected(output[term.output], labels, teacher=teacher)
        for term in predictions.terms
    )
    ratios = {}
    if predictions.auxiliary is not None:
        auxiliary = predictions.auxiliary(predictions, output, teacher, weights)
        per_token = per_token + auxiliary.values
        ratios.update(auxiliary.ratios)
    prediction = logits.argmax(-1)
    metric_weights = weights
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
        weights.sum().detach()
        if predictions.denominator is None
        else predictions.denominator
    )
    return ObjectiveResult((per_token * weights).sum(), denominator, ratios)


def execute_batch(batch, loss, weighting, *, training):
    results = [
        evaluate_predictions(item, loss, weighting) for item in batch.predictions
    ]
    if not results:
        raise ValueError("protocol must produce at least one prediction set")
    ratios = {}
    for result in results:
        for key, pair in result.ratios.items():
            ratios[key] = (
                tuple(a + b for a, b in zip(ratios[key], pair))
                if key in ratios
                else pair
            )
    if batch.reduction == "sum_means":
        return ObjectiveResult(
            sum(result.loss for result in results),
            results[0].denominator.new_ones(()),
            ratios,
        )
    result = ObjectiveResult(
        sum(r.numerator for r in results), sum(r.denominator for r in results), ratios
    )
    if batch.reduction == "global_microbatch":
        return microbatch_mean(result, global_denominator=training)
    if batch.reduction != "tokens":
        raise ValueError(f"unknown protocol reduction: {batch.reduction!r}")
    return result
