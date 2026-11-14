"""One chunk-local execution service for losses and intermediate measurements."""

from dataclasses import replace

from .implementations.registry import ExecutionRequest, compute_loss
from .losses import LossOutput, TokenLoss


class LossComputation:
    """Run a chunk's declared terms and dependent calculations through one registry.

    Protocols may inspect predictions, teacher, weights and differentiable
    statistics, call model heads, then run further terms. No tensors are cached
    across chunks or checkpoint replays. Stable call keys identify intermediate
    requests for distributed agreement and resume validation.
    """

    def __init__(self, request, selection, execution, prediction_index):
        self.request = request
        self.predictions = request.predictions
        self.selection = selection
        self.execution = execution
        self.prediction_index = prediction_index
        self.numerators = {}
        self.predicted = {}
        self.statistics = {}
        self.ratios = {}
        self._calls = set()

    @property
    def teacher(self):
        return self.request.teacher

    @property
    def labels(self):
        return self.request.labels

    @property
    def weights(self):
        return self.request.terms[0].weights

    def outputs(self):
        """Materialize the projection once within this chunk."""
        return self.request.outputs()

    def _execute(self, request, group, key, *, fused=None):
        return compute_loss(
            request,
            group,
            fused=self.execution.fused if fused is None else fused,
            selection=self.selection,
            key=key,
        )

    def evaluate_term(
        self, index, logits, labels, *, teacher=None, context=None, fused=None
    ):
        """Evaluate the declared formula on algorithm-prepared inputs."""
        term = self.predictions.plan.terms[index]
        return self.evaluate(
            term.name or term.output,
            self.request.terms[index].loss,
            logits,
            labels,
            teacher=teacher,
            context=context,
            term_index=index,
            fused=fused,
        )

    def run(self, group=None, *, fused=None):
        """Execute declared terms once.

        The shared fused setting controls automatic implementation selection.
        """
        groups = self.predictions.plan.groups if group is None else (tuple(group),)
        for indices in groups:
            if any(i in self.numerators for i in indices):
                raise ValueError("Loss terms must execute exactly once")
            result = self._execute(
                self.request,
                indices,
                self.prediction_index,
                fused=fused,
            )
            self._collect(result, indices)

    def _collect(self, result, indices):
        """Merge one implementation result into this chunk's loss plan."""
        if set(result.numerators) != set(indices):
            raise ValueError("Implementation did not return the requested loss terms")
        if not result.statistics.keys() <= set(indices):
            raise ValueError("Implementation statistics reference an unexecuted term")
        self.numerators.update(result.numerators)
        self.predicted.update(result.predictions)
        self.statistics.update(result.statistics)

    def add(self, index, values):
        """Submit computed values for a declared term without recomputing its formula.

        Values retain gradients. The same registry and term weights/denominator
        apply, including explicit masks for router or other model regularizers.
        Scalar terms run once over the full prediction set without token weighting.
        """
        if index in self.numerators:
            raise ValueError("Loss terms must execute exactly once")
        terms = list(self.request.terms)
        terms[index] = replace(terms[index], loss=TokenLoss("value"))
        request = replace(self.request, terms=tuple(terms), tensors={index: values})
        result = self._execute(request, (index,), self.prediction_index)
        if result is None:
            raise ValueError("No implementation for computed loss")
        self._collect(result, (index,))

    def evaluate(
        self,
        key,
        loss,
        logits,
        labels,
        *,
        teacher=None,
        context=None,
        fused=None,
        term_index=None,
        **requested_outputs,
    ):
        """Return unreduced LossOutput through the ordinary implementation registry.

        Args:
            key: Unique, stable name within this chunk procedure.
            loss: Mathematical loss definition; output requests are separate.
            logits: Already-computed scores; no extra projection is performed.
            labels: Hard or soft targets aligned with this chunk's positions.
            teacher: Optional aligned teacher logits.
            context: Optional normalization context for whole-set losses.
            fused: Override automatic optimization for this internal request.
            term_index: Optional owner of the input context.
            **requested_outputs: Extra results validated and computed by the
                selected implementation, e.g. return_teacher_l1=True or top_k=8.

        The algorithm controls gradient truncation and how results are consumed.
        These values are not added to the objective until a declared term runs.
        """
        if key in self._calls:
            raise ValueError(f"Repeated computation key: {key}")
        self._calls.add(key)
        if (
            loss.registration.input_kind != "features"
            and labels.shape != self.predictions.mask.shape
        ):
            raise ValueError("Computation labels must align with prediction units")
        if (
            term_index is None
            and loss.registration.scope == "prediction"
            and (self.predictions.chunk_safe and self.predictions.projection.chunk_safe)
        ):
            raise ValueError("Contextual intermediate losses require chunk_safe=False")
        inputs = replace(
            self.request.terms[0 if term_index is None else term_index],
            loss=loss,
            reduction="positions",
            context=(
                context
                if context is not None
                else self.request.terms[0 if term_index is None else term_index].context
            ),
        )
        request = ExecutionRequest(
            predictions=None,
            terms=(inputs,),
            labels=labels,
            teacher=teacher,
            outputs=None,
            materialized={},
            tensors={0: logits},
            unreduced=True,
            requested_outputs=requested_outputs,
        )
        result = self._execute(
            request,
            (0,),
            f"{self.prediction_index}/compute/{key}",
            fused=fused,
        )
        if result is None or set(result.values) != {0}:
            raise ValueError(
                "Implementation must return the requested per-position values"
            )
        return LossOutput(result.values[0], result.statistics.get(0, {}))
