"""Add term statistics before applying their normalization and coefficients."""

from dataclasses import dataclass, field

from .base import ObjectiveResult, microbatch_mean


@dataclass
class PlanResult:
    """Additive per-term statistics before final normalization.

    Attributes:
        terms: ObjectiveResult entries whose numerators exclude term coefficients.
        coefficients: Multipliers aligned with terms.
        shared_reduction: Whether terms share the same normalization contract.
        ratios: Named metric numerator/denominator pairs, independent of loss.
        scalar_terms: Indices of terms holding one already normalized scalar.
        shared_positions: Whether the positional terms share one denominator,
            which the scalar terms then join.
    """

    terms: tuple[ObjectiveResult, ...]
    coefficients: tuple[float, ...]
    shared_reduction: bool
    ratios: dict
    partials: dict = field(default_factory=dict)
    scalar_terms: tuple[int, ...] = ()
    shared_positions: bool = False

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
        positional = [i for i in range(len(self.terms)) if i not in self.scalar_terms]
        if self.shared_positions and positional:
            # the scalar terms join the positional mean, scaled by its denominator
            denominator = self.terms[positional[0]].denominator
            return ObjectiveResult(
                sum(self.coefficients[i] * self.terms[i].numerator for i in positional)
                + sum(
                    self.coefficients[i] * self.terms[i].loss * denominator
                    for i in self.scalar_terms
                ),
                denominator,
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
        or result.scalar_terms != first.scalar_terms
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
        terms,
        first.coefficients,
        first.shared_reduction,
        ratios,
        partials,
        scalar_terms=first.scalar_terms,
        shared_positions=first.shared_positions,
    )
