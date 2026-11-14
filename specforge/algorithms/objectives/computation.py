"""One chunk-local execution service for losses and intermediate measurements."""

from dataclasses import replace

from .implementations.reference import reference
from .implementations.registry import ExecutionRequest, GroupResult
from .losses import LossContext, LossOutput
from .plan import LossPlan, LossTerm
from .predictions import Teacher
from .projection import Projection


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

    def _execute(self, request, group, key):
        entry = self.selection.resolve(request, group, key, fused=self.execution.fused)
        if entry is None:
            return None
        # Empty inputs retain zero gradients without launching zero-sized kernels.
        if not request.predictions.hidden.numel():
            result = GroupResult({})
            for index in group:
                item = reference(request, (index,))
                result.numerators.update(item.numerators)
                result.statistics.update(item.statistics)
                result.values.update(item.values)
            return result
        return entry.compute(request, group)

    def run(self, group=None):
        """Execute declared terms once, preserving joint-kernel opportunities."""
        groups = self.predictions.plan.groups if group is None else (tuple(group),)
        for indices in groups:
            if any(i in self.numerators for i in indices):
                raise ValueError("Loss terms must execute exactly once")
            result = self._execute(self.request, indices, self.prediction_index)
            if result is None:
                if len(indices) == 1:
                    raise ValueError("No implementation for loss term")
                for index in indices:
                    self.run((index,))
                continue
            if set(result.numerators) != set(indices):
                raise ValueError(
                    "Implementation did not return the requested loss terms"
                )
            if not result.statistics.keys() <= set(indices):
                raise ValueError(
                    "Implementation statistics reference an unexecuted term"
                )
            self.numerators.update(result.numerators)
            self.predicted.update(result.predictions)
            self.statistics.update(result.statistics)

    def add(self, index, values):
        """Supply per-position model loss values to a declared 'value' term.

        Values retain gradients. The same registry and term weights/denominator
        apply, including explicit masks for router or other model regularizers.
        Cross-token model losses must disable chunking and trimming.
        """
        if self.request.terms[index].loss.kind != "value":
            raise ValueError("Computed values require a value loss term")
        outputs = self.outputs()
        name = self.predictions.plan.terms[index].output
        if name in outputs:
            raise ValueError(f"Computed loss would overwrite projection output: {name}")
        outputs[name] = values
        self.run((index,))

    def evaluate(
        self,
        key,
        loss,
        logits,
        labels,
        *,
        teacher=None,
        context=None,
        return_teacher_l1=False,
        top_k=None,
    ):
        """Return unreduced LossOutput through the ordinary implementation registry.

        Args:
            key: Unique, stable name within this chunk procedure.
            loss: Mathematical loss definition; output requests are separate.
            logits: Already-computed scores; no extra projection is performed.
            labels: Hard or soft targets aligned with this chunk's positions.
            teacher: Optional aligned teacher logits.
            context: Optional normalization context for whole-set losses.
            return_teacher_l1: Also return draft/teacher L1 in statistics, not in the loss.
            top_k: Also return this many candidate IDs and logits in statistics.
                These extra outputs currently accompany CE.

        The algorithm controls gradient truncation and how results are consumed.
        These values are not added to the objective until a declared term runs.
        """
        if type(return_teacher_l1) is not bool:
            raise ValueError("return_teacher_l1 must be boolean")
        if top_k is not None and (type(top_k) is not int or top_k < 1):
            raise ValueError("top_k must be a positive integer")
        requested_outputs = {}
        if return_teacher_l1:
            requested_outputs["teacher_l1"] = True
        if top_k is not None:
            requested_outputs["top_k"] = top_k
        if requested_outputs and loss.kind != "ce":
            raise ValueError("L1/top-k output requests currently require CE")
        if return_teacher_l1 and (teacher is None or teacher.shape != logits.shape):
            raise ValueError("return_teacher_l1 requires aligned teacher logits")
        if key in self._calls:
            raise ValueError(f"Repeated computation key: {key}")
        self._calls.add(key)
        if labels.shape != self.predictions.mask.shape:
            raise ValueError("Computation labels must align with prediction units")
        if loss.registration.contextual and (
            self.predictions.chunk_safe and self.predictions.projection.chunk_safe
        ):
            raise ValueError("Contextual intermediate losses require chunk_safe=False")
        outputs = {"logits": logits}
        predictions = replace(
            self.predictions,
            hidden=logits if logits.ndim == 3 else logits[..., None],
            labels=labels,
            teacher=None if teacher is None else Teacher(teacher),
            vocab_mask=None,
            draft_to_target=None,
            context_inputs={},
            projection=Projection(lambda hidden: outputs),
            projection_inputs={},
            plan=LossPlan((LossTerm(loss=loss),)),
            compute=None,
        )
        inputs = replace(
            self.request.terms[0],
            loss=loss,
            context=(
                context
                if context is not None
                else LossContext(self.request.terms[0].mask)
            ),
        )
        request = ExecutionRequest(
            predictions,
            (inputs,),
            labels,
            teacher,
            lambda: outputs,
            outputs,
            unreduced=True,
            requested_outputs=requested_outputs,
        )
        result = self._execute(request, (0,), f"{self.prediction_index}/compute/{key}")
        if result is None or set(result.values) != {0}:
            raise ValueError(
                "Implementation must return the requested per-position values"
            )
        return LossOutput(result.values[0], result.statistics.get(0, {}))
