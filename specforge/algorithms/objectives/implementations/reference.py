"""Reference and precomputed values use the same execution result contract."""

from .registry import GroupResult, register_implementation


def result_group(request, index, result):
    """Keep per-position values when requested; otherwise return a weighted sum."""
    if result.values.shape != request.terms[index].weights.shape:
        raise ValueError("Loss values must align with prediction units")
    return GroupResult(
        (
            {}
            if request.unreduced
            else {index: (result.values * request.terms[index].weights).sum()}
        ),
        statistics={index: result.statistics},
        values={index: result.values} if request.unreduced else {},
    )


def singleton(request, group):
    return None if len(group) == 1 else "reference executes one term at a time"


@register_implementation(
    "reference", supports=singleton, priority=-100, optimized=False
)
def reference(request, group):
    """Evaluate a registered mathematical loss without an optimized kernel."""
    (index,) = group
    term = request.predictions.plan.terms[index]
    inputs = request.terms[index]
    result = inputs.loss.evaluate(
        request.outputs()[term.output],
        request.labels,
        teacher=request.teacher,
        context=inputs.context,
    )
    return result_group(request, index, result)


@register_implementation(
    "value",
    optimized=False,
    supports=lambda request, group: (
        None
        if len(group) == 1 and request.terms[group[0]].loss.kind == "value"
        else "requires one precomputed term"
    ),
)
def value(request, group):
    """Precomputed model losses need reduction, not a fused numerical kernel."""
    return reference(request, group)
