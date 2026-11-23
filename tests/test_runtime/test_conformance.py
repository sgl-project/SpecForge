"""The conformance kit holds for the reference runner, and it can fail."""

import contextlib
import copy
import dataclasses
import os
import unittest
from unittest.mock import patch

import torch

from specforge.algorithms.objectives.registry import (
    protocol_registrations,
    resolve_objective,
)
from specforge.training.backends.plain import PlainRunner
from specforge.training.contracts import StepContext
from specforge.training.controller import TrainerController
from specforge.training.frame import run_window
from specforge.training.runner import DataParallelRunner
from tests.test_runtime.conformance import (
    DECLARATIONS,
    MUTATIONS,
    KitProgram,
    ProtocolChecks,
    RunnerChecks,
    batch,
    declaration_cases,
    declared,
    one_rank_cases,
    one_rank_state,
    plain_runner,
    protocol_violations,
    reference_runner,
    trained_by_the_plain_runner,
)
from tests.test_runtime.test_canonical_end_to_end import drafts, recomputed_tolerance
from tests.test_runtime.trajectories import cases, differences, record


class ReferenceProtocolTest(ProtocolChecks, unittest.TestCase):
    """The run loop over the DDP / FSDP runner."""


class ReferenceRunnerTest(RunnerChecks, unittest.TestCase):
    """The DDP / FSDP runner against itself: the jobs are deterministic."""


class PlainProtocolTest(ProtocolChecks, unittest.TestCase):
    """The run loop over the plain runner."""

    make_runner = staticmethod(plain_runner)


class PlainRunnerTest(RunnerChecks, unittest.TestCase):
    """The plain runner against the reference runner, bit for bit."""

    make_runner = staticmethod(plain_runner)

    def test_job_of_several_ranks_is_refused(self):
        program = KitProgram()
        with self.assertRaisesRegex(ValueError, "this job has 2 ranks"):
            PlainRunner(
                program,
                program.spec(),
                optimizer_factory=lambda module: None,
                world_size=2,
            )

    def test_trainable_module_declared_frozen_is_refused(self):
        """As the reference refuses it: the optimizer would train it."""
        program = KitProgram()
        program.frozen = (program.linear,)
        with self.assertRaisesRegex(ValueError, "only frozen parameters"):
            plain_runner(program, program.spec())

    def test_runner_issues_no_collective_inside_a_process_group(self):
        """Not even in a group of several ranks that it was wrongly built in."""
        program = KitProgram(reduction="microbatch")
        runner = plain_runner(program, program.spec())
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.get_rank", return_value=1),
            patch("torch.distributed.all_reduce") as reduce,
            patch("torch.distributed.all_gather_object") as gather,
        ):
            run_window(runner, [batch(0), batch(1)], StepContext(accumulation_steps=2))
            runner.reduce_sum(torch.ones(1))
            runner.reduce_max(torch.ones(1))
            runner.agree(None, value="kernels")
            for failure in ("disk full", ""):
                # Any string is a failure, an empty one included.
                with self.assertRaisesRegex(RuntimeError, f"rank 0: {failure}$"):
                    runner.agree(failure)
            saved = runner.checkpoint_state()
        reduce.assert_not_called()
        gather.assert_not_called()

    def test_resume_needs_no_process_group(self):
        program = KitProgram()
        runner = plain_runner(program, program.spec())
        run_window(runner, [batch(0)], StepContext())
        saved = runner.checkpoint_state()
        again = KitProgram()
        no_group = RuntimeError("Default process group has not been initialized")
        with (
            patch("torch.distributed.is_initialized", return_value=False),
            # Whatever an earlier test left initialized.
            patch("torch.distributed.get_rank", side_effect=no_group),
        ):
            resumed = plain_runner(again, again.spec())
            resumed.load_resume(copy.deepcopy(saved.rank))
        self.assertEqual(
            differences(
                resumed.checkpoint_state().rank["optimizer"], saved.rank["optimizer"]
            ),
            [],
        )


class PlainTrajectoryTest(unittest.TestCase):
    """One rank through ``specforge train`` on the host: the plain runner against DDP."""

    def test_plain_runner_reproduces_the_reference_at_one_rank(self):
        # On the host whatever the machine has: the gate is bitwise there.
        with patch.dict(os.environ, {"SPECFORGE_DEVICE": "cpu"}):
            placeable = {case.name for case in cases()}
            selected = [case for case in one_rank_cases() if case.name in placeable]
            self.assertEqual(len(selected), len(one_rank_cases()))
            reference = record(selected)
            plain = record(selected, around_training=trained_by_the_plain_runner)
        # Metrics, learning rates, weights, optimizer and RNG state.
        self.assertEqual(
            differences(one_rank_state(plain), one_rank_state(reference)), []
        )
        # What does differ is where the optimizer state is written: once in the
        # shared file under DDP, in the rank's file without a wrapper.
        for name, updates in plain.items():
            for update, expected in zip(updates, reference[name]):
                self.assertNotIn("replicated_optimizer_state", update["shared"])
                self.assertTrue(update["ranks"][0]["optimizer"])
                self.assertTrue(expected["shared"]["replicated_optimizer_state"])
                self.assertIsNone(expected["ranks"][0]["optimizer"])
                self.assertEqual(update["replicas"], [])


class _Unscaled(DataParallelRunner):
    """Applies the accumulated gradients without the window's rescale."""

    def apply_step(self, pending):
        with patch.object(self.backend, "scale_gradients", lambda factor: None):
            return super().apply_step(pending)


class _ReplayOnAnotherGenerator(DataParallelRunner):
    """Replays a window without the generator state its prepass started from."""

    @contextlib.contextmanager
    def _replayed_rng(self):
        if self._prepass_rng:
            self._prepass_rng.popleft()
        yield


def _as(runner_class):
    def make_runner(program, spec):
        reference = reference_runner(program, spec)
        return runner_class(reference.program, reference.backend, spec)

    return make_runner


class KitCanFailTest(unittest.TestCase):
    """A check that cannot fail checks nothing."""

    def failures(self, runner_class, job):
        class Probe(RunnerChecks, unittest.TestCase):
            make_runner = staticmethod(_as(runner_class))

        result = unittest.TestResult()
        Probe(job).run(result)
        self.assertEqual(result.errors, [])
        return len(result.failures)

    def test_missing_rescale_fails_the_reduction_jobs(self):
        self.assertEqual(self.failures(_Unscaled, "test_window_reduction"), 1)
        # Under the microbatch reduction the rescale is by one.
        self.assertEqual(self.failures(_Unscaled, "test_microbatch_reduction"), 0)

    def test_replay_on_another_generator_fails_the_replay_job(self):
        broken = _ReplayOnAnotherGenerator
        self.assertEqual(self.failures(broken, "test_replay_statistic"), 1)
        self.assertEqual(self.failures(broken, "test_window_reduction"), 0)

    def test_lost_resume_state_fails_the_resume_job(self):
        class WithoutMasters(DataParallelRunner):
            def load_resume(self, state):
                optimizer = dict(copy.deepcopy(state["optimizer"]))
                del optimizer["fp32_params"]
                super().load_resume({**state, "optimizer": optimizer})

        class WithoutSchedule(DataParallelRunner):
            def load_resume(self, state):
                # Everything but the schedule's position is restored.
                scheduler = self.backend.optimizer.scheduler
                with patch.object(scheduler, "load_state_dict"):
                    super().load_resume(state)

        for broken in (WithoutMasters, WithoutSchedule):
            with self.subTest(runner=broken.__name__):
                self.assertEqual(
                    self.failures(broken, "test_resume_continues_the_same_trajectory"),
                    1,
                )

    def test_masters_rounded_to_the_weights_fail_the_bfloat16_job(self):
        class RoundedMasters(DataParallelRunner):
            """Copies the rounded draft back over its fp32 masters every update."""

            def apply_step(self, pending):
                metrics = super().apply_step(pending)
                optimizer = self.backend.optimizer
                with torch.no_grad():
                    for master, weight in zip(
                        optimizer.fp32_params, optimizer.model_params
                    ):
                        master.copy_(weight)
                return metrics

        self.assertEqual(
            self.failures(RoundedMasters, "test_window_reduction_in_bfloat16"), 1
        )
        # In fp32 the weights are the masters: nothing is lost there.
        self.assertEqual(self.failures(RoundedMasters, "test_window_reduction"), 0)

    def test_prepass_outside_training_mode_fails_the_replay_job(self):
        class EvaluatedPrepass(DataParallelRunner):
            def prepass(self, batch, ctx):
                self.program.eval()
                try:
                    return super().prepass(batch, ctx)
                finally:
                    self.program.train()

            def _train_mode(self):
                pass

        self.assertEqual(self.failures(EvaluatedPrepass, "test_replay_statistic"), 1)

    def test_dropped_schedule_or_detail_flag_fails_the_window_job(self):
        class StepZero(DataParallelRunner):
            def _forward(self, batch, ctx, **arguments):
                return super()._forward(
                    batch, dataclasses.replace(ctx, global_step=0), **arguments
                )

        class NoDetail(DataParallelRunner):
            def _forward(self, batch, ctx, **arguments):
                ctx = dataclasses.replace(ctx, collect_detailed_metrics=False)
                return super()._forward(batch, ctx, **arguments)

        for broken in (StepZero, NoDetail):
            with self.subTest(runner=broken.__name__):
                self.assertEqual(self.failures(broken, "test_window_reduction"), 1)

    def test_checkpoint_that_does_not_wait_for_commits_fails_the_protocol(self):
        class Probe(ProtocolChecks, unittest.TestCase):
            pass

        result = unittest.TestResult()
        with patch.object(TrainerController, "_flush_durable_ack", lambda self: None):
            Probe("test_commits_are_flushed_before_a_checkpoint").run(result)
        self.assertEqual((len(result.failures), result.errors), (1, []))

    def test_ledger_that_breaks_the_commit_contract_is_reported(self):
        first, second = ("s0.0", "s0.1"), ("s1.0", "s1.1")
        window = first + second
        good = [
            ("batch", first),
            ("forward", first),
            ("batch", second),
            ("forward", second),
            ("update",),
            ("commit", window, 1),
            ("snapshot", 0),
            ("checkpoint", 1),
        ]
        self.assertEqual(protocol_violations(good), [])
        read_1, hand_1, read_2, hand_2, update, commit, snapshot, checkpoint = good
        for broken, messages in (
            (
                [read_1, hand_1, read_2, hand_2, commit, update],
                ["step 1 was committed after 0 updates"],
            ),
            (
                [*good[:5], snapshot, checkpoint, commit],
                [
                    "a checkpoint's state was collected after 0 commits of 1 "
                    "updates, 0 in flight",
                    "checkpoint 1 was written after 0 commits",
                ],
            ),
            (
                [*good[:6], ("snapshot", 1), checkpoint],
                [
                    "a checkpoint's state was collected after 1 commits of 1 "
                    "updates, 1 in flight"
                ],
            ),
            (
                # A batch read and acknowledged that the runner never saw.
                [read_1, hand_1, read_2, hand_1, update, ("commit", first + first, 1)],
                ["the runner was handed ('s0.0', 's0.1'), not the stream's next batch"],
            ),
            (
                [read_1, hand_1, update, ("commit", first, 1)],
                ["update 1 consumed 1 micro-batches, not 2"],
            ),
            (
                # One sample of each micro-batch is not the window.
                [*good[:5], ("commit", ("s0.0", "s1.0"), 1)],
                ["step 1 committed ('s0.0', 's1.0'); its update consumed " f"{window}"],
            ),
            (
                [*good[:5], ("raised", "IncompleteWindow"), commit],
                ["step 1 was committed after the run raised"],
            ),
            ([*good[:5]], ["1 updates were applied and 0 committed"]),
            (
                [*good[:6], commit],
                [
                    "commit of step 1 follows 1 commits",
                    "1 updates were applied and 2 committed",
                ],
            ),
        ):
            with self.subTest(expected=messages[0]):
                self.assertEqual(protocol_violations(broken), messages)

    def test_runner_that_ignores_the_terms_cannot_train_the_microbatch_job(self):
        class _WindowOnly(DataParallelRunner):
            def _loss_terms(self, result):
                return result.numerator, result.denominator

        program = KitProgram(reduction="microbatch")
        runner = _as(_WindowOnly)(program, program.spec())
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            runner.forward_backward(batch(0), StepContext(), is_boundary=True)


class DeclarationTest(unittest.TestCase):
    def test_each_architecture_makes_the_declaration_it_stands_for(self):
        for declaration, architecture in DECLARATIONS.items():
            with self.subTest(declaration=declaration):
                training = resolve_objective(architecture).registration.training
                self.assertIn(declaration, declared(training))

    def test_every_declaration_in_the_registry_has_a_case(self):
        """A new reduction or statistic needs its case before a runner can claim it."""
        for name, registration in protocol_registrations().items():
            with self.subTest(protocol=name):
                self.assertLessEqual(declared(registration.training), set(DECLARATIONS))


class MutationTest(unittest.TestCase):
    """Three broken references, each of which the replay has to notice."""

    @classmethod
    def setUpClass(cls):
        recorded = {draft.architecture for draft in drafts()}
        unknown = [
            case.name
            for case in declaration_cases()
            if case.architecture not in recorded
        ]
        # A case the recorder has no draft for is a mistake, not a skip.
        assert not unknown, unknown
        placeable = {case.name for case in cases()}
        cls.cases = [case for case in declaration_cases() if case.name in placeable]
        if len(cls.cases) != len(declaration_cases()):
            # The cases need two ranks; a host with one accelerator has one.
            raise unittest.SkipTest("this host cannot place two training ranks")
        cls.reference = record(cls.cases)
        cls.mutated = {
            name: record(cls.cases, around_training=mutation)
            for name, mutation in MUTATIONS.items()
        }

    def changed(self, mutation):
        """Per case, the recorded values that the mutation changed."""
        return {
            case.name: differences(
                {case.name: self.mutated[mutation][case.name]},
                {case.name: self.reference[case.name]},
                # Beyond what two accelerator runs of one revision differ by.
                atol=recomputed_tolerance(),
            )
            for case in self.cases
        }

    def test_every_mutation_changes_every_case(self):
        for mutation in MUTATIONS:
            for name, found in self.changed(mutation).items():
                with self.subTest(mutation=mutation, case=name):
                    self.assertTrue(found)

    def test_world_size_scaling_shows_in_the_first_gradient_norm(self):
        found = self.changed("gradients scaled by the world size")
        for name, lines in found.items():
            with self.subTest(case=name):
                where = f"recording.{name}[0].logged[0].metrics.grad_norm: "
                self.assertTrue(
                    any(line.startswith(where) for line in lines),
                    f"{where} not among {lines}",
                )

    def test_shifted_schedule_shows_in_the_first_applied_rate(self):
        found = self.changed("learning-rate schedule shifted by one step")
        for name, lines in found.items():
            with self.subTest(case=name):
                self.assertTrue(
                    any(
                        line.startswith(f"recording.{name}[0].applied_lr[0]")
                        for line in lines
                    )
                )


if __name__ == "__main__":
    unittest.main()
