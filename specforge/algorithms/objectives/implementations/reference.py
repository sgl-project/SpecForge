"""Reference and precomputed values use the same execution result contract."""

from .registry import GroupResult, register_implementation


def result_group(request, index, result):
    """Keep per-position values when requested; otherwise return a weighted sum."""
    term = request.terms[index]
    if term.reduction == "scalar":
        if result.values.ndim != 0:
            raise ValueError("Scalar loss must return a scalar tensor")
        return GroupResult(
            {index: result.values}, statistics={index: result.statistics}
        )
    if result.values.shape != term.weights.shape:
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
    if len(group) != 1:
        return "reference executes one term at a time"
    return None


@register_implementation(
    "reference", supports=singleton, priority=-100, category="reference"
)
def reference(request, group):
    """Evaluate a registered mathematical loss without an optimized kernel."""
    (index,) = group
    inputs = request.terms[index]
    result = inputs.loss.evaluate(
        request.tensor(index),
        request.labels,
        teacher=request.teacher,
        context=inputs.context,
        requested_outputs=request.requested_outputs,
    )
    return result_group(request, index, result)


@register_implementation(
    "value",
    losses=("value",),
    category="reference",
    supports=lambda request, group: (
        None
        if not request.requested_outputs
        and len(group) == 1
        and request.terms[group[0]].loss.kind == "value"
        else "requires one precomputed term"
    ),
)
def value(request, group):
    """Precomputed model losses need reduction, not a fused numerical kernel."""
    return reference(request, group)
