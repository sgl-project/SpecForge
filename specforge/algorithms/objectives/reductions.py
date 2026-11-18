"""Add term statistics before applying their normalization and coefficients."""

from dataclasses import dataclass

from .base import ObjectiveResult, microbatch_mean


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

    @property
    def loss(self):
        """Return the sum of individually normalized, coefficient-scaled losses."""
        return sum(c * term.loss for c, term in zip(self.coefficients, self.terms))

    def objective_result(self, *, global_denominator=False, normalize=False):
        """Convert term statistics to the trainer's scalar-result contract.

        Args:
            global_denominator: All-reduce each denominator when distributed is
                initialized, with scaling to compensate for averaged rank gradients.
            normalize: Normalize each term now even without global synchronization.

        Returns:
            ObjectiveResult retaining a shared denominator when possible, otherwise
            storing the already-normalized loss with denominator one. Metrics are
            preserved and remain the caller's responsibility to aggregate.
        """
        if normalize or global_denominator:
            loss = sum(
                c * microbatch_mean(term, global_denominator=global_denominator).loss
                for c, term in zip(self.coefficients, self.terms)
            )
            return ObjectiveResult(
                loss, self.terms[0].denominator.new_ones(()), self.ratios
            )
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
    return PlanResult(terms, first.coefficients, first.shared_reduction, ratios)
