"""One chunk-local execution service for losses and intermediate measurements."""

from dataclasses import replace

from .implementations.registry import ExecutionRequest, compute_loss
from .losses import LossContext, LossOutput


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

    def _execute(self, request, group, key, *, impl=None, fused=None):
        return compute_loss(
            request,
            group,
            impl=impl,
            fused=self.execution.fused if fused is None else fused,
            selection=self.selection,
            key=key,
        )

    def run(self, group=None, *, impl=None, fused=None):
        """Execute declared terms once.

        impl selects a specialized implementation and requires fused=False.
        Otherwise fused inherits the runtime default and selects generic kernels.
        """
        groups = self.predictions.plan.groups if group is None else (tuple(group),)
        for indices in groups:
            if any(i in self.numerators for i in indices):
                raise ValueError("Loss terms must execute exactly once")
            result = self._execute(
                self.request, indices, self.prediction_index, impl=impl, fused=fused
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
        """Supply per-position model loss values to a declared 'value' term.

        Values retain gradients. The same registry and term weights/denominator
        apply, including explicit masks for router or other model regularizers.
        Scalar terms run once over the full prediction set without token weighting.
        """
        if self.request.terms[index].loss.kind != "value":
            raise ValueError("Computed values require a value loss term")
        if index in self.numerators:
            raise ValueError("Loss terms must execute exactly once")
        request = replace(self.request, tensors={index: values})
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
        impl=None,
        fused=None,
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
            impl: Optional specialized implementation.
            fused: Override automatic optimization; must be False with impl.
            **requested_outputs: Extra results validated and computed by the
                selected implementation, e.g. return_teacher_l1=True or top_k=8.

        The algorithm controls gradient truncation and how results are consumed.
        These values are not added to the objective until a declared term runs.
        """
        if key in self._calls:
            raise ValueError(f"Repeated computation key: {key}")
        self._calls.add(key)
        if labels.shape != self.predictions.mask.shape:
            raise ValueError("Computation labels must align with prediction units")
        if loss.registration.scope == "prediction" and (
            self.predictions.chunk_safe and self.predictions.projection.chunk_safe
        ):
            raise ValueError("Contextual intermediate losses require chunk_safe=False")
        inputs = replace(
            self.request.terms[0],
            loss=loss,
            reduction="positions",
            context=(
                context
                if context is not None
                else LossContext(self.request.terms[0].mask)
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
            impl=impl,
            fused=fused,
        )
        if result is None or set(result.values) != {0}:
            raise ValueError(
                "Implementation must return the requested per-position values"
            )
        return LossOutput(result.values[0], result.statistics.get(0, {}))
