"""Conformance kit of a step runner: what it has to reproduce, and proof that the checks can fail.

Three parts, after the training-backend design's *Conformance* section.

Numerics, by replay. ``declaration_cases`` is one recorded case for each thing
a program can declare to a runner (each reduction, a replay statistic,
evaluation diagnostics), not one per recipe: a recipe is a combination of
these. A runner of the ``exact`` class reproduces the reference's recording of
them on the same machine (``tests.test_runtime.trajectories``).

The kit must be able to fail. ``MUTATIONS`` are three deliberately broken
references. A recording of each has to differ from the reference's, or the
replay proves nothing.

Protocol, by a fake stream. ``ProtocolChecks`` drives the run loop over a
runner with a stream that can be read once and a ledger of what happened:
the window spans the epoch boundary, samples are committed after the update
that consumed them and in order, an incomplete window is never committed, and
commits are flushed before a checkpoint.

``RunnerChecks`` holds a runner to the reference runner's updates in one
process, bit for bit, for each declaration.

The recorded cases train the runner that ``specforge train`` assembles, which
is the reference: no configuration key selects another. A candidate is
recorded by entering a substitution around every rank's run
(``record(cases, around_training=...)``); until a runner of several ranks has
one, the two mixins are what hold it, in one process.

``one_rank_cases`` are the recorded cases a one-process runner can place, and
``trained_by_the_plain_runner`` is the plain runner's substitution.
"""

from __future__ import annotations

import copy
import tempfile
import time
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

import torch
from torch import nn

from specforge.algorithms.objectives.base import ObjectiveResult
from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.checkpoint import CheckpointManager
from specforge.training.contracts import IncompleteWindow, StepContext
from specforge.training.controller import TrainerController
from specforge.training.frame import run_window
from specforge.training.runner import DataParallelRunner
from tests.test_runtime._fixtures import FakeProgram, fake_spec
from tests.test_runtime.trajectories import Case, differences

# The architecture whose objective makes each declaration.
DECLARATIONS = {
    "reduction: window": "DFlashDraftModel",
    "reduction: microbatch": "DSparkDraftModel",
    "statistic: replay": "DSpineDraftModel",
    "evaluation diagnostics": "Eagle3DraftModel",
}


def declared(training):
    """The declarations a protocol's training requirements make to a runner."""
    found = {f"reduction: {training.reduction}"}
    if training.window_statistic is not None:
        found.add("statistic: replay")
    if training.evaluation_diagnostics:
        found.add("evaluation diagnostics")
    return found


def declaration_cases():
    """One recorded case per declaration.

    Two ranks and a window of two micro-batches: the smallest topology in
    which every reduction crosses both ranks and micro-batches. The case that
    declares evaluation diagnostics also evaluates.
    """
    return [
        Case(
            architecture,
            world_size=2,
            accumulation_steps=2,
            evaluated=declaration == "evaluation diagnostics",
        )
        for declaration, architecture in DECLARATIONS.items()
    ]


def one_rank_cases():
    """Each declaration on one rank: alone, in a window of two, and evaluated.

    Under DDP in fp32, the wrapper a CPU host can place. One-rank FSDP is
    another reference: it casts buffers to fp32 while it runs.
    """
    return [
        Case(architecture, **topology)
        for architecture in DECLARATIONS.values()
        for topology in (dict(), dict(accumulation_steps=2), dict(evaluated=True))
    ]


@contextmanager
def gradients_scaled_by_world_size():
    """The rescale forgets that the wrapper already averaged over the ranks."""
    original = FSDPTrainingBackend.scale_gradients

    def scale_gradients(self, factor):
        original(self, factor * torch.distributed.get_world_size())

    with patch.object(FSDPTrainingBackend, "scale_gradients", scale_gradients):
        yield


@contextmanager
def clipping_skipped():
    """The optimizer reports the gradient norm and never clips to it."""
    from specforge.optimizer import BF16Optimizer

    original = BF16Optimizer._reduce_grad_norm

    def unclipped(self, total_norm_sq):
        total_norm, coefficient = original(self, total_norm_sq)
        return total_norm, torch.ones_like(coefficient)

    with patch.object(BF16Optimizer, "_reduce_grad_norm", unclipped):
        yield


@contextmanager
def schedule_shifted_by_one_step():
    """The learning-rate schedule starts one update ahead."""
    from specforge.optimizer import BF16Optimizer

    original = BF16Optimizer.__init__

    def shifted(self, *arguments, **keywords):
        original(self, *arguments, **keywords)
        self.scheduler.step()

    with patch.object(BF16Optimizer, "__init__", shifted):
        yield


#: Each must change the reference's recording.
MUTATIONS = {
    "gradients scaled by the world size": gradients_scaled_by_world_size,
    "clipping skipped": clipping_skipped,
    "learning-rate schedule shifted by one step": schedule_shifted_by_one_step,
}


def reference_runner(program, spec):
    """The DDP / FSDP runner over ``program`` in one process, unwrapped.

    With the optimizer of a real run, so a job also clips, follows the
    schedule and keeps fp32 masters.
    """
    from specforge.optimizer import BF16Optimizer

    backend = FSDPTrainingBackend(
        ParallelConfig(),
        optimizer_factory=lambda module: BF16Optimizer(
            module, lr=0.05, max_grad_norm=1.0, warmup_ratio=0.5, total_steps=4
        ),
    )
    return DataParallelRunner(backend.prepare_model(program, wrap=False), backend, spec)


def plain_runner(program, spec):
    """The plain runner over ``program``, with the reference job's optimizer."""
    from specforge.optimizer import BF16Optimizer
    from specforge.training.backends.plain import PlainRunner

    return PlainRunner(
        program,
        spec,
        optimizer_factory=lambda module: BF16Optimizer(
            module, lr=0.05, max_grad_norm=1.0, warmup_ratio=0.5, total_steps=4
        ),
        world_size=1,
    )


@contextmanager
def trained_by_the_plain_runner():
    """Have ``Trainer`` assemble the plain runner where it assembles the reference.

    No configuration key selects a runner yet, so a recording of the plain
    runner through ``specforge train`` replaces the two names ``Trainer``
    builds its runner from.
    """
    from specforge.training import trainer
    from specforge.training.backends.plain import PlainRunner

    class Assembly:
        """Stands in for the backend ``Trainer`` constructs and wraps with."""

        def __init__(self, parallel, *, optimizer_factory):
            self.world_size = parallel.world_size
            self.optimizer_factory = optimizer_factory

        def prepare_model(self, model, *, optimizer_target, frozen, blocks):
            # The plain runner reads the program's own declaration.
            assert frozen == model.frozen
            self.optimizer_target = optimizer_target
            return model

    def runner(program, assembly, spec):
        return PlainRunner(
            program,
            spec,
            optimizer_factory=assembly.optimizer_factory,
            optimizer_target=assembly.optimizer_target,
            world_size=assembly.world_size,
        )

    with (
        patch.object(trainer, "FSDPTrainingBackend", Assembly),
        patch.object(trainer, "DataParallelRunner", runner),
    ):
        yield


def one_rank_state(recording):
    """A one-rank recording without where the optimizer state was written.

    DDP writes the optimizer once, in the checkpoint's shared file, and the
    recorder reads each replica's own state besides; a runner without a
    wrapper writes it in its rank's file. The values are kept, under one
    name. A run that saved no optimizer state at all is refused: two such
    runs would compare equal.
    """
    view = {}
    for name, updates in recording.items():
        view[name] = []
        for update in updates:
            (rank,) = update["ranks"]
            shared = dict(update["shared"])
            optimizer = shared.pop("replicated_optimizer_state", None)
            if optimizer is None:
                optimizer = rank["optimizer"]
            if not optimizer:
                raise AssertionError(f"{name}: a checkpoint holds no optimizer state")
            view[name].append(
                {
                    **{
                        key: update[key]
                        for key in ("checkpoint", "logged", "evaluated", "applied_lr")
                    },
                    "shared": shared,
                    "optimizer": optimizer,
                    "rank": {key: rank[key] for key in rank if key != "optimizer"},
                }
            )
    return view


def batch(index, value=None):
    """A batch of two samples; their ids and ``x`` say where it is in the stream."""
    value = float(index + 1) if value is None else value
    return TrainBatch(
        sample_ids=[f"s{index}.0", f"s{index}.1"],
        strategy="kit",
        tensors={"x": torch.tensor([value])},
    )


class KitProgram(FakeProgram):
    """A two-weight program that can make every declaration.

    Under ``window`` its denominator is the batch's ``x``, so micro-batches
    weigh differently. Under ``microbatch`` it returns two terms with
    denominators of their own and a numerator that carries no graph, so a
    runner that took the numerator for the loss has nothing to differentiate.
    With ``replay`` it turns the window's pooled sums into its forward's
    argument itself, by a function that is not their ratio alone, and its
    prepass reports sums that depend on the noise it drew in training mode.
    Its forward draws from the generator and reads the step, and reports one
    ratio only when detailed metrics are asked for: a runner that restores
    the wrong RNG state, runs a prepass in another mode, or drops the
    schedule or the flag, shows in the update or in what it reports.
    """

    def __init__(
        self,
        *,
        reduction="window",
        replay=False,
        diagnostics=False,
        dtype=torch.float32,
    ):
        super().__init__()
        torch.manual_seed(11)
        self.linear = nn.Linear(1, 2, bias=False).to(dtype)
        self.reduction = reduction
        self.replay = replay
        self.diagnostics = diagnostics
        #: ``collect_detailed_metrics`` as each forward received it.
        self.detailed = []

    def spec(self):
        return fake_spec(
            "x",
            name="kit",
            reduction=self.reduction,
            replay_statistics=("scale",) if self.replay else (),
            evaluation_diagnostics=self.diagnostics,
        )

    def replay_arguments(self, replay_values, *, prepass=False):
        if not self.replay or prepass or not replay_values:
            return {}
        count, total = replay_values["scale"]
        # Not their ratio alone: means in place of sums would show.
        return {"scale": count / (total + 1)}

    def forward(
        self,
        tensors,
        *,
        step=0,
        collect_detailed_metrics=True,
        prepass=False,
        scale=None,
        **context,
    ):
        self.detailed.append(collect_detailed_metrics)
        weight = self.linear.weight
        x = tensors["x"].to(weight.dtype)
        noise = (
            torch.rand(2, device=weight.device)
            if self.training
            else torch.zeros(2, device=weight.device)
        )
        first, second = (self.linear(x) * (1 + noise).to(weight.dtype)).unbind()
        first = first * (1 + 0.25 * step)
        if scale is not None:
            first = first * scale.to(weight.dtype)
        size = x.sum().detach().float()
        result = ObjectiveResult(first + 0.5 * second * second, size)
        result.ratios["acc"] = (size, size + 1)
        if collect_detailed_metrics:
            result.ratios["detail"] = (size, size + 3)
        if prepass:
            result.statistics["scale"] = torch.stack((size * (1 + noise[0]), size + 2))
        if self.reduction == "microbatch":
            terms = ((1.0, first, size), (0.5, second * second, size + 1))
            result = replace(
                result,
                numerator=sum(c * n / d for c, n, d in terms).detach(),
                denominator=size.new_ones(()),
                terms=terms if self.training else (),
            )
        return result


class _Observed:
    """A runner as given, with what it was handed and asked written to a ledger."""

    def __init__(self, runner, ledger, in_flight):
        self._runner, self._ledger, self._in_flight = runner, ledger, in_flight

    def __getattr__(self, name):
        return getattr(self._runner, name)

    def forward_backward(self, batch, ctx, *, is_boundary):
        self._ledger.append(("forward", tuple(batch.sample_ids)))
        return self._runner.forward_backward(batch, ctx, is_boundary=is_boundary)

    def apply_step(self, pending):
        metrics = self._runner.apply_step(pending)
        self._ledger.append(("update",))
        return metrics

    def checkpoint_state(self):
        self._ledger.append(("snapshot", self._in_flight()))
        return self._runner.checkpoint_state()


class _Stream:
    """An epoch of batches that can be read once; every read is in the ledger."""

    def __init__(self, ledger, batches_per_epoch):
        self._ledger, self._size, self._read = ledger, batches_per_epoch, 0

    def __iter__(self):
        start, self._read = self._read, self._read + self._size
        for index in range(start, start + self._size):
            handed = batch(index)
            self._ledger.append(("batch", tuple(handed.sample_ids)))
            yield handed


class _LedgerCheckpoints(CheckpointManager):
    def __init__(self, output_dir, run_id, ledger):
        super().__init__(output_dir, run_id)
        self._ledger = ledger

    def save(self, state, step, *, rank_state=None):
        self._ledger.append(("checkpoint", step))
        return super().save(state, step, rank_state=rank_state)


def protocol_ledger(
    runner, *, batches_per_epoch=3, epochs=2, accumulation_steps=2, async_ack=False
):
    """Run the run loop over ``runner`` and return what happened, in order.

    Entries: ``("batch", ids)`` as the stream hands a batch out,
    ``("forward", ids)`` as the runner is handed it, ``("update",)`` as the
    runner returns from an optimizer step, ``("commit", ids, step)`` as
    samples are acknowledged, ``("snapshot", acknowledgements in flight)`` as
    a checkpoint's state is asked of the runner, ``("checkpoint", step)`` as
    it starts to be written, and ``("raised", type name)`` if the run ended in
    an incomplete window. An asynchronous acknowledgement takes a moment
    here, so a checkpoint that did not wait for it finds it in flight or not
    yet made.
    """
    ledger, flying = [], []

    def acknowledge(ids, step):
        if async_ack:
            flying.append(step)
            time.sleep(0.05)
        ledger.append(("commit", tuple(ids), step))
        if async_ack:
            flying.remove(step)

    with tempfile.TemporaryDirectory() as directory:
        controller = TrainerController(
            _Observed(runner, ledger, lambda: len(flying)),
            strategy_name="kit",
            rank=0,
            world_size=1,
            accumulation_steps=accumulation_steps,
            run_id="kit",
            output_dir=directory,
            num_epochs=epochs,
            save_interval=1,
            ack_fn=acknowledge,
            async_ack=async_ack,
            checkpoint_manager=_LedgerCheckpoints(directory, "kit", ledger),
        )
        try:
            controller.fit(_Stream(ledger, batches_per_epoch))
        except IncompleteWindow as error:
            ledger.append(("raised", type(error).__name__))
    return ledger


def protocol_violations(ledger, *, accumulation_steps=2):
    """What a ledger shows that the commit contract forbids.

    The runner is handed the stream's batches, each once and in order. An
    update consumes one whole window of them. A commit names every sample its
    step's update consumed, follows that update, and comes in step order.
    Nothing is committed once the run has raised. A checkpoint's state is
    collected, and the checkpoint written, with every update so far committed
    and no acknowledgement in flight. A run that ended without raising
    committed every update.
    """
    found = []
    read, handed, window, consumed = [], 0, [], []
    committed, raised = 0, False
    for kind, *entry in ledger:
        if kind == "batch":
            read.append(entry[0])
        elif kind == "forward":
            if handed >= len(read) or read[handed] != entry[0]:
                found.append(
                    f"the runner was handed {entry[0]}, not the stream's next batch"
                )
            handed += 1
            window.append(entry[0])
        elif kind == "update":
            if len(window) != accumulation_steps:
                found.append(
                    f"update {len(consumed) + 1} consumed {len(window)} "
                    f"micro-batches, not {accumulation_steps}"
                )
            consumed.append(tuple(sample for ids in window for sample in ids))
            window = []
        elif kind == "raised":
            raised = True
        elif kind == "commit":
            ids, step = entry
            if raised:
                found.append(f"step {step} was committed after the run raised")
            if step != committed + 1:
                found.append(f"commit of step {step} follows {committed} commits")
            if step > len(consumed):
                found.append(f"step {step} was committed after {len(consumed)} updates")
            elif ids != consumed[step - 1]:
                found.append(
                    f"step {step} committed {ids}; its update consumed "
                    f"{consumed[step - 1]}"
                )
            committed += 1
        elif kind == "snapshot":
            if entry[0] or committed < len(consumed):
                found.append(
                    f"a checkpoint's state was collected after {committed} commits "
                    f"of {len(consumed)} updates, {entry[0]} in flight"
                )
        elif kind == "checkpoint" and committed < entry[0]:
            found.append(f"checkpoint {entry[0]} was written after {committed} commits")
    if not raised and committed != len(consumed):
        found.append(f"{len(consumed)} updates were applied and {committed} committed")
    return found


def _ids(*indices):
    """The sample ids of the stream's batches at ``indices``, in order."""
    return tuple(sample for index in indices for sample in batch(index).sample_ids)


class ProtocolChecks:
    """What the run loop and a runner owe the data plane.

    Mix into a ``unittest.TestCase`` and set ``make_runner`` to a function
    ``(program, spec) -> StepRunner``. The program declares no replay
    statistic: a prepass is not in the ledger.
    """

    make_runner = staticmethod(reference_runner)

    def ledger(self, **options):
        program = KitProgram()
        return protocol_ledger(self.make_runner(program, program.spec()), **options)

    def entries(self, ledger, kind):
        return [entry for entry in ledger if entry[0] == kind]

    def test_window_spans_the_epoch_boundary(self):
        """Three batches an epoch, two epochs, a window of two: three updates."""
        ledger = self.ledger()
        self.assertEqual(self.entries(ledger, "raised"), [])
        self.assertEqual(len(self.entries(ledger, "update")), 3)
        for kind in ("batch", "forward"):
            self.assertEqual(
                [entry[1] for entry in self.entries(ledger, kind)],
                [_ids(index) for index in range(6)],
            )

    def test_samples_are_committed_after_their_update_in_order(self):
        ledger = self.ledger()
        self.assertEqual(self.entries(ledger, "raised"), [])
        self.assertEqual(
            self.entries(ledger, "commit"),
            [
                ("commit", _ids(0, 1), 1),
                ("commit", _ids(2, 3), 2),
                ("commit", _ids(4, 5), 3),
            ],
        )
        self.assertEqual(protocol_violations(ledger), [])

    def test_incomplete_window_is_not_committed(self):
        ledger = self.ledger(epochs=1)
        self.assertEqual(ledger[-1], ("raised", "IncompleteWindow"))
        self.assertEqual(len(self.entries(ledger, "update")), 1)
        # The third batch was handed out and belongs to no committed window.
        self.assertIn(("batch", _ids(2)), ledger)
        self.assertEqual(self.entries(ledger, "commit"), [("commit", _ids(0, 1), 1)])
        self.assertEqual(protocol_violations(ledger), [])

    def test_commits_are_flushed_before_a_checkpoint(self):
        for async_ack in (False, True):
            with self.subTest(async_ack=async_ack):
                ledger = self.ledger(async_ack=async_ack)
                self.assertEqual(self.entries(ledger, "raised"), [])
                self.assertEqual(
                    self.entries(ledger, "checkpoint"),
                    [("checkpoint", step) for step in (1, 2, 3)],
                )
                self.assertEqual(len(self.entries(ledger, "snapshot")), 3)
                self.assertEqual(protocol_violations(ledger), [])


def _train(runner, program, windows, *, first_step=0):
    """Run ``windows``; return what every update reported and what the run left."""
    trajectory = []
    for step, window in enumerate(windows, start=first_step):
        ctx = StepContext(global_step=step, accumulation_steps=len(window))
        metrics = run_window(runner, iter(window), ctx)
        trajectory.append(
            {
                "metrics": dict(metrics.materialize()),
                # Before clipping: a gradient scaled wrongly shows here even
                # where the clip then hides it from the weights.
                "grad_norm": metrics.grad_norm,
                "learning_rate": metrics.learning_rate,
                "weights": program.linear.weight.detach().clone(),
            }
        )
    state = runner.checkpoint_state()
    optimizer = (
        state.rank["optimizer"] if state.replicated is None else state.replicated
    )
    return {
        "updates": trajectory,
        # What the run would be exported and resumed from: the draft's
        # weights as the runner hands them over, and the optimizer's moments,
        # step counts, schedule position and fp32 masters.
        "draft": {
            name: tensor.detach().clone() for name, tensor in state.draft.items()
        },
        "optimizer": copy.deepcopy(optimizer),
        "generator": torch.get_rng_state(),
        # Whether each forward was asked for the detailed metrics.
        "detailed": list(program.detailed),
    }


class RunnerChecks:
    """A runner reproduces the reference runner in one process.

    Mix into a ``unittest.TestCase`` and set ``make_runner``. One job per
    declaration; each compares every update's metrics, gradient norm,
    learning rate and weights, and the draft, optimizer and generator state at
    the end, with the reference runner's, bit for bit. Both runners are built
    on the host: an accelerator has a generator of its own.
    """

    make_runner = staticmethod(reference_runner)

    #: Windows of two micro-batches with different weights.
    windows = (
        (batch(0, 1.0), batch(1, 3.0)),
        (batch(2, 2.0), batch(3, 0.5)),
        (batch(4, 1.5), batch(5, 2.5)),
    )

    def trained(self, make_runner, program):
        """Train ``program`` from a fixed generator state, set once it is built."""
        runner = make_runner(program, program.spec())
        torch.manual_seed(5)
        return _train(runner, program, self.windows)

    def assert_reproduces_the_reference(self, **declarations):
        expected = self.trained(reference_runner, KitProgram(**declarations))
        actual = self.trained(self.make_runner, KitProgram(**declarations))
        self.assertEqual(differences(actual, expected), [])
        # The job is not vacuous: the weights moved, and differently each update.
        first, second = (update["weights"] for update in expected["updates"][:2])
        self.assertFalse(torch.equal(first, second))

    def test_window_reduction(self):
        self.assert_reproduces_the_reference()

    def test_window_reduction_in_bfloat16(self):
        """The optimizer's fp32 masters hold bits the weights do not."""
        self.assert_reproduces_the_reference(dtype=torch.bfloat16)

    def test_microbatch_reduction(self):
        self.assert_reproduces_the_reference(reduction="microbatch")

    def test_replay_statistic(self):
        self.assert_reproduces_the_reference(replay=True)

    def test_handle_is_single_use(self):
        program = KitProgram()
        runner = self.make_runner(program, program.spec())
        pending = runner.forward_backward(batch(0), StepContext(), is_boundary=True)
        runner.apply_step(pending)
        with self.assertRaises(RuntimeError):
            runner.apply_step(pending)

    def test_batch_without_a_declared_input_is_refused(self):
        program = KitProgram()
        runner = self.make_runner(program, program.spec())
        unfit = TrainBatch(["s0"], "kit", {"y": torch.ones(1)})
        with self.assertRaises(ValueError):
            runner.forward_backward(unfit, StepContext(), is_boundary=True)
        # Before the program ran.
        self.assertEqual(program.detailed, [])

    def test_evaluation_reports_what_the_program_computed(self):
        for diagnostics in (False, True):
            with self.subTest(diagnostics=diagnostics):
                results = []
                for make_runner in (reference_runner, self.make_runner):
                    program = KitProgram(diagnostics=diagnostics)
                    runner = make_runner(program, program.spec())
                    before = program.linear.weight.detach().clone()
                    generator = torch.get_rng_state()
                    # The declaration decides, not what the caller asks for.
                    ctx = StepContext(collect_detailed_metrics=not diagnostics)
                    with runner.evaluation():
                        self.assertFalse(program.training)
                        result = runner.evaluate(batch(0, 3.0), ctx)
                    self.assertEqual(program.detailed, [diagnostics])
                    self.assertFalse(result.numerator.requires_grad)
                    self.assertTrue(program.training)
                    torch.testing.assert_close(
                        program.linear.weight, before, rtol=0, atol=0
                    )
                    self.assertTrue(torch.equal(torch.get_rng_state(), generator))
                    results.append(
                        {
                            "loss": (result.numerator, result.denominator),
                            **result.ratios,
                        }
                    )
                self.assertEqual(differences(results[1], results[0]), [])

    def test_resume_continues_the_same_trajectory(self):
        """In bfloat16: the optimizer's fp32 masters hold bits the weights do not."""
        dtype = torch.bfloat16
        expected = self.trained(self.make_runner, KitProgram(dtype=dtype))
        first, second = KitProgram(dtype=dtype), KitProgram(dtype=dtype)
        runner = self.make_runner(first, first.spec())
        torch.manual_seed(5)
        run_window(
            runner,
            iter(self.windows[0]),
            StepContext(global_step=0, accumulation_steps=2),
        )
        state = runner.checkpoint_state()
        # The draft as exported: its keys, dtype and bits are the uninterrupted
        # run's after the same update.
        second.linear.load_state_dict({"weight": state.draft["linear.weight"].clone()})
        self.assertEqual(
            differences(
                dict(second.linear.state_dict()),
                {"weight": expected["updates"][0]["weights"]},
            ),
            [],
        )
        resumed = self.make_runner(second, second.spec())
        # Another generator state than the one saved: resume has to restore it.
        torch.manual_seed(99)
        saved = dict(state.rank)
        if state.replicated is not None:
            saved["optimizer"] = state.replicated
        resumed.load_resume(copy.deepcopy(saved))
        # Two more updates: the second one runs at the schedule's next rate.
        actual = _train(resumed, second, self.windows[1:], first_step=1)
        self.assertEqual(differences(actual["updates"], expected["updates"][1:]), [])
        for kept in ("draft", "optimizer", "generator"):
            self.assertEqual(differences(actual[kept], expected[kept]), [], kept)
