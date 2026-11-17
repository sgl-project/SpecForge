"""Joint CE implementation for the Domino block-suffix projection."""

from ..projection import DominoSuffixProjection
from .ce import projection_ce_support
from .registry import GroupResult, register_implementation


def _domino_suffix_support(request, group):
    """Return None or a reason the base/final CE pair cannot be fused."""
    p = request.predictions
    if not isinstance(p.projection, DominoSuffixProjection):
        return "requires a Domino block-suffix projection"
    if p.hidden.shape[1] != p.projection.block_size:
        return "requires one complete Domino block per projection unit"
    if len(group) != 2 or {p.plan.terms[i].output for i in group} != {
        "logits",
        "base_logits",
    }:
        return "requires the base/final CE pair"
    if any(
        p.plan.terms[i].mask is not None or p.plan.terms[i].weighting is not None
        for i in group
    ):
        return "joint CE requires shared supervision weights"
    return projection_ce_support(request, group)


@register_implementation(
    "domino_suffix_ce", supports=_domino_suffix_support, priority=20
)
def domino_suffix_ce(request, group):
    """Compute base and corrected CE without materializing final logits.

    Args:
        request: Validated DominoSuffixProjection with labels and shared weights.
        group: Two term indices naming base_logits and logits.

    Returns:
        GroupResult containing each weighted CE sum and final predicted IDs.
        Coefficients and normalization are applied later by the executor.
    """
    from specforge.core.domino_loss import domino_weighted_cross_entropy

    p, projection = request.predictions, request.predictions.projection
    base, correction = projection.project_parts(p.hidden, **p.projection_inputs)
    units, positions = p.hidden.shape[:2]
    if base.ndim != 3 or base.shape[:2] != (units, positions):
        raise ValueError(
            "Domino base logits must have shape [units, block_size, vocabulary]"
        )
    expected = (
        units,
        1,
        projection.block_size - projection.suffix_start,
        base.shape[-1],
    )
    if correction.shape != expected:
        raise ValueError(
            "Domino corrections must have shape [units, 1, suffix_size, vocabulary]"
        )
    final, original, predicted, _ = domino_weighted_cross_entropy(
        base.flatten(0, 1),
        correction.reshape(-1, correction.shape[-1]),
        request.labels.flatten(),
        request.terms[group[0]].weights.flatten(),
        projection.block_size,
        projection.suffix_start,
        use_fused=True,
    )
    values = {"logits": final, "base_logits": original}
    return GroupResult(
        {i: values[p.plan.terms[i].output] for i in group},
        {"logits": predicted.reshape_as(request.labels)},
    )
