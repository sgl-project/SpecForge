"""Add term statistics before applying their normalization and coefficients."""

from dataclasses import dataclass, field

from .base import ObjectiveResult


@dataclass
class PlanResult:
    """Additive per-term statistics before final normalization.

    Attributes:
        terms: ObjectiveResult entries whose numerators exclude term coefficients.
        coefficients: Multipliers aligned with terms.
        shared_reduction: Whether terms share the same normalization contract.
        ratios: Named metric numerator/denominator pairs, independent of loss.
    """

    terms: tuple[ObjectiveResult, ...]
    coefficients: tuple[float, ...]
    shared_reduction: bool
    ratios: dict
    partials: dict = field(default_factory=dict)

    @property
    def loss(self):
        """Return the sum of individually normalized, coefficient-scaled losses."""
        return sum(c * term.loss for c, term in zip(self.coefficients, self.terms))

    def objective_result(self, *, normalize=False, with_terms=False):
        """Convert term statistics to the trainer's scalar-result contract.

        Args:
            normalize: Normalize each term by its own denominator on this rank,
                even when the terms share one.
            with_terms: Also return every term, for a runner that pools each
                denominator over the ranks before the backward.

        Returns:
            ObjectiveResult retaining a shared denominator when possible, otherwise
            storing the already-normalized loss with denominator one. Metrics are
            preserved and remain the caller's responsibility to aggregate.
        """
        if with_terms and not normalize:
            raise ValueError("terms are returned with per-term normalization only")
        if normalize:
            # A denominator normalizes and is never differentiated, here or
            # where a runner pools it.
            denominators = [term.denominator.detach() for term in self.terms]
            result = ObjectiveResult(
                sum(
                    c * (term.numerator / d.where(d > 0, 1))
                    for c, term, d in zip(self.coefficients, self.terms, denominators)
                ),
                denominators[0].new_ones(()),
                self.ratios,
            )
            if with_terms:
                result.terms = tuple(
                    (c, term.numerator, d)
                    for c, term, d in zip(self.coefficients, self.terms, denominators)
                )
            return result
        if self.shared_reduction:
            return ObjectiveResult(
                sum(
                    c * term.numerator for c, term in zip(self.coefficients, self.terms)
                ),
                self.terms[0].denominator,
                self.ratios,
            )
        return ObjectiveResult(
            self.loss, self.terms[0].denominator.new_ones(()), self.ratios
        )


def combine_results(results):
    """Add chunk statistics without averaging chunk means.

    Args:
        results: Nonempty sequence of PlanResult objects with aligned term order,
            coefficients and reduction contract.

    Returns:
        A PlanResult with term numerators, denominators and metric pairs summed.
        The caller restores fixed whole-set denominators after chunk merging.

    Raises:
        ValueError: Coefficients or shared-reduction declarations differ.
    """
    first = results[0]
    if any(
        result.coefficients != first.coefficients
        or result.shared_reduction != first.shared_reduction
        for result in results[1:]
    ):
        raise ValueError("Additive prediction sets must use the same loss plan")
    ratios = {}
    for result in results:
        for key, pair in result.ratios.items():
            ratios[key] = (
                tuple(a + b for a, b in zip(ratios[key], pair))
                if key in ratios
                else pair
            )
    terms = tuple(
        ObjectiveResult(
            sum(result.terms[i].numerator for result in results),
            sum(result.terms[i].denominator for result in results),
        )
        for i in range(len(first.terms))
    )
    partials = {key: sum(r.partials[key] for r in results) for key in first.partials}
    return PlanResult(
        terms, first.coefficients, first.shared_reduction, ratios, partials
    )
