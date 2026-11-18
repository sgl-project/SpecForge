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
"""

from __future__ import annotations

import tempfile
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
from tests.test_runtime.trajectories import Case

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


def batch(index, value=None):
    """A one-sample batch whose id and ``x`` are its position in the stream."""
    value = float(index + 1) if value is None else value
    return TrainBatch(
        sample_ids=[f"s{index}"], strategy="kit", tensors={"x": torch.tensor([value])}
    )


class KitProgram(FakeProgram):
    """A two-weight program that can make every declaration.

    Under ``window`` its denominator is the batch's ``x``, so micro-batches
    weigh differently. Under ``microbatch`` it returns two terms with
    denominators of their own. With ``replay`` its loss is scaled by the
    window's pooled statistic, and a prepass reports that statistic. Its
    forward draws from the generator, so a replay that restored the wrong RNG
    state shows in the update.
    """

    def __init__(self, *, reduction="window", replay=False):
        super().__init__()
        torch.manual_seed(11)
        self.linear = nn.Linear(1, 2, bias=False)
        self.reduction = reduction
        self.replay = replay

    def spec(self):
        return fake_spec(
            "x",
            name="kit",
            reduction=self.reduction,
            replay_statistics=("scale",) if self.replay else (),
        )

    def forward(self, tensors, *, prepass=False, replay_values=None, **context):
        x = tensors["x"]
        noise = torch.rand(2) if self.training else torch.zeros(2)
        first, second = (self.linear(x) * (1 + noise)).unbind()
        if replay_values:
            count, total = replay_values["scale"]
            first = first * (count / total)
        result = ObjectiveResult(first + 0.5 * second * second, x.sum().detach())
        result.ratios["acc"] = (x.sum().detach(), x.sum().detach() + 1)
        if prepass:
            result.statistics["scale"] = torch.stack(
                (x.sum().detach(), x.sum().detach() + 2)
            )
        if self.reduction == "microbatch":
            terms = (
                (1.0, first, x.sum().detach()),
                (0.5, second * second, x.sum().detach() + 1),
            )
            result = replace(
                result,
                numerator=sum(c * n / d for c, n, d in terms),
                denominator=torch.ones(()),
                terms=terms if self.training else (),
            )
        return result


class _Observed:
    """A runner as given, with every applied update written to a ledger."""

    def __init__(self, runner, ledger):
        self._runner, self._ledger = runner, ledger

    def __getattr__(self, name):
        return getattr(self._runner, name)

    def apply_step(self, pending):
        metrics = self._runner.apply_step(pending)
        self._ledger.append(("update",))
        return metrics


class _Stream:
    """An epoch of batches that can be read once; every read is in the ledger."""

    def __init__(self, ledger, batches_per_epoch):
        self._ledger, self._size, self._read = ledger, batches_per_epoch, 0

    def __iter__(self):
        start, self._read = self._read, self._read + self._size
        for index in range(start, start + self._size):
            self._ledger.append(("batch", f"s{index}"))
            yield batch(index)


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

    Entries: ``("batch", id)`` as the stream hands a batch out, ``("update",)``
    as the runner returns from an optimizer step, ``("commit", ids, step)`` as
    samples are acknowledged, ``("checkpoint", step)`` as a checkpoint is
    written, and ``("raised", type name)`` if the run ended in an exception.
    """
    ledger = []
    with tempfile.TemporaryDirectory() as directory:
        controller = TrainerController(
            _Observed(runner, ledger),
            strategy_name="kit",
            accumulation_steps=accumulation_steps,
            run_id="kit",
            output_dir=directory,
            num_epochs=epochs,
            save_interval=1,
            ack_fn=lambda ids, step: ledger.append(("commit", tuple(ids), step)),
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

    A sample is committed only after the update that consumed it; commits
    follow the stream's order, one whole window each, so an incomplete window
    is never committed; every commit up to a checkpoint's step precedes it.
    """
    found = []
    handed_out = [entry[1] for entry in ledger if entry[0] == "batch"]
    updates = committed = 0
    for entry in ledger:
        if entry[0] == "update":
            updates += 1
        elif entry[0] == "commit":
            _, ids, step = entry
            window = tuple(
                handed_out[committed * accumulation_steps :][:accumulation_steps]
            )
            if step != committed + 1:
                found.append(f"commit of step {step} follows {committed} commits")
            if ids != window or len(window) != accumulation_steps:
                found.append(f"step {step} committed {ids}, not the window {window}")
            if updates < step:
                found.append(f"step {step} was committed after {updates} updates")
            committed += 1
        elif entry[0] == "checkpoint" and committed < entry[1]:
            found.append(f"checkpoint {entry[1]} was written after {committed} commits")
    return found


class ProtocolChecks:
    """What the run loop and a runner owe the data plane.

    Mix into a ``unittest.TestCase`` and set ``make_runner`` to a function
    ``(program, spec) -> StepRunner``.
    """

    make_runner = staticmethod(reference_runner)

    def ledger(self, **options):
        program = KitProgram()
        return protocol_ledger(self.make_runner(program, program.spec()), **options)

    def test_window_spans_the_epoch_boundary(self):
        """Three batches an epoch, two epochs, a window of two: three updates."""
        ledger = self.ledger()
        self.assertEqual(ledger.count(("update",)), 3)
        self.assertEqual(
            [entry[1] for entry in ledger if entry[0] == "batch"],
            [f"s{index}" for index in range(6)],
        )

    def test_samples_are_committed_after_their_update_in_order(self):
        ledger = self.ledger()
        self.assertEqual(
            [entry for entry in ledger if entry[0] == "commit"],
            [
                ("commit", ("s0", "s1"), 1),
                ("commit", ("s2", "s3"), 2),
                ("commit", ("s4", "s5"), 3),
            ],
        )
        self.assertEqual(protocol_violations(ledger), [])

    def test_incomplete_window_is_not_committed(self):
        ledger = self.ledger(epochs=1)
        self.assertEqual(ledger[-1], ("raised", "IncompleteWindow"))
        self.assertEqual(ledger.count(("update",)), 1)
        # The third batch was handed out and belongs to no committed window.
        self.assertIn(("batch", "s2"), ledger)
        self.assertEqual(
            [entry for entry in ledger if entry[0] == "commit"],
            [("commit", ("s0", "s1"), 1)],
        )
        self.assertEqual(protocol_violations(ledger), [])

    def test_commits_are_flushed_before_a_checkpoint(self):
        for async_ack in (False, True):
            with self.subTest(async_ack=async_ack):
                ledger = self.ledger(async_ack=async_ack)
                self.assertEqual(
                    [entry for entry in ledger if entry[0] == "checkpoint"],
                    [("checkpoint", step) for step in (1, 2, 3)],
                )
                self.assertEqual(protocol_violations(ledger), [])


def _train(runner, program, windows):
    """Run ``windows``; return every update's loss, gradient norm and weights."""
    torch.manual_seed(5)
    trajectory = []
    for step, window in enumerate(windows):
        ctx = StepContext(global_step=step, accumulation_steps=len(window))
        metrics = run_window(runner, iter(window), ctx)
        trajectory.append(
            (
                metrics.loss,
                # Before clipping: a gradient scaled wrongly shows here even
                # where the clip then hides it from the weights.
                metrics.grad_norm,
                program.linear.weight.detach().clone(),
            )
        )
    return trajectory


class RunnerChecks:
    """A runner reproduces the reference runner's updates in one process.

    Mix into a ``unittest.TestCase`` and set ``make_runner``. One job per
    declaration; each compares losses and weights after every update with the
    reference runner's, bit for bit.
    """

    make_runner = staticmethod(reference_runner)

    #: Two windows of two micro-batches with different weights.
    windows = (
        (batch(0, 1.0), batch(1, 3.0)),
        (batch(2, 2.0), batch(3, 0.5)),
    )

    def assert_reproduces_the_reference(self, **declarations):
        candidate, reference = KitProgram(**declarations), KitProgram(**declarations)
        expected = _train(
            reference_runner(reference, reference.spec()), reference, self.windows
        )
        actual = _train(
            self.make_runner(candidate, candidate.spec()), candidate, self.windows
        )
        self.assertEqual(len(actual), len(expected))
        for update, reference_update in zip(actual, expected):
            self.assertEqual(update[:2], reference_update[:2])
            torch.testing.assert_close(update[2], reference_update[2], rtol=0, atol=0)
        # The job is not vacuous: the weights moved, and differently each update.
        self.assertFalse(torch.equal(expected[0][2], expected[1][2]))

    def test_window_reduction(self):
        self.assert_reproduces_the_reference()

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

    def test_evaluation_leaves_training_untouched(self):
        program = KitProgram()
        runner = self.make_runner(program, program.spec())
        before = program.linear.weight.detach().clone()
        generator = torch.get_rng_state()
        with runner.evaluation():
            self.assertFalse(program.training)
            result = runner.evaluate(batch(0), StepContext())
        self.assertFalse(result.numerator.requires_grad)
        self.assertTrue(program.training)
        torch.testing.assert_close(program.linear.weight, before, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), generator))

    def test_resume_continues_the_same_trajectory(self):
        uninterrupted = KitProgram()
        expected = _train(
            self.make_runner(uninterrupted, uninterrupted.spec()),
            uninterrupted,
            self.windows,
        )
        first, second = KitProgram(), KitProgram()
        runner = self.make_runner(first, first.spec())
        torch.manual_seed(5)
        run_window(
            runner,
            iter(self.windows[0]),
            StepContext(global_step=0, accumulation_steps=2),
        )
        state = runner.checkpoint_state()
        second.linear.load_state_dict({"weight": state.draft["linear.weight"].clone()})
        resumed = self.make_runner(second, second.spec())
        # Another generator state than the one saved: resume has to restore it.
        torch.manual_seed(99)
        saved = dict(state.rank)
        if state.replicated is not None:
            saved["optimizer"] = state.replicated
        # The optimizer logs its restore on rank 0 of a process group.
        with patch("specforge.optimizer.print_on_rank0"):
            resumed.load_resume(saved)
        metrics = run_window(
            resumed,
            iter(self.windows[1]),
            StepContext(global_step=1, accumulation_steps=2),
        )
        self.assertEqual((metrics.loss, metrics.grad_norm), expected[1][:2])
        torch.testing.assert_close(second.linear.weight, expected[1][2], rtol=0, atol=0)
