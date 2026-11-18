"""The conformance kit holds for the reference runner, and it can fail."""

import contextlib
import unittest
from unittest.mock import patch

import torch

from specforge.algorithms.objectives.registry import (
    protocol_registrations,
    resolve_objective,
)
from specforge.training.backends.plain import PlainRunner
from specforge.training.contracts import StepContext
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

    def test_no_collective_is_issued_inside_a_process_group(self):
        """Not even in a group of several ranks that it was wrongly built in."""
        program = KitProgram(reduction="microbatch")
        runner = plain_runner(program, program.spec())
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.all_reduce") as reduce,
            patch("torch.distributed.all_gather_object") as gather,
        ):
            run_window(runner, [batch(0), batch(1)], StepContext(accumulation_steps=2))
            runner.reduce_sum(torch.ones(1))
            runner.reduce_max(torch.ones(1))
            runner.agree(None, value="kernels")
            with self.assertRaisesRegex(RuntimeError, "rank 0: disk full"):
                runner.agree("disk full")
        reduce.assert_not_called()
        gather.assert_not_called()


class PlainTrajectoryTest(unittest.TestCase):
    """One rank through ``specforge train``: the plain runner is the reference, bit for bit."""

    def test_plain_runner_reproduces_the_reference_at_one_rank(self):
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

    def test_ledger_that_breaks_the_commit_contract_is_reported(self):
        good = [
            ("batch", "s0"),
            ("batch", "s1"),
            ("update",),
            ("commit", ("s0", "s1"), 1),
            ("checkpoint", 1),
        ]
        self.assertEqual(protocol_violations(good), [])
        early = [good[0], good[1], good[3], good[2], good[4]]
        self.assertEqual(
            protocol_violations(early), ["step 1 was committed after 0 updates"]
        )
        unflushed = [good[0], good[1], good[2], good[4], good[3]]
        self.assertEqual(
            protocol_violations(unflushed),
            ["checkpoint 1 was written after 0 commits"],
        )
        partial = [good[0], ("update",), ("commit", ("s0",), 1)]
        self.assertEqual(
            protocol_violations(partial),
            ["step 1 committed ('s0',), not the window ('s0',)"],
        )
        reordered = good[:3] + [("commit", ("s1", "s0"), 1)]
        self.assertEqual(
            protocol_violations(reordered),
            ["step 1 committed ('s1', 's0'), not the window ('s0', 's1')"],
        )


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
        placeable = {case.name for case in cases()}
        cls.cases = [case for case in declaration_cases() if case.name in placeable]
        # An accelerator host with one device would leave nothing to compare.
        assert cls.cases
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
                self.assertIn(
                    f"recording.{name}[0].logged[0].metrics.grad_norm: values differ",
                    lines,
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

    def test_skipped_clipping_leaves_the_first_gradient_norm_alone(self):
        """It changes the update, not what the update measured."""
        found = self.changed("clipping skipped")
        for name, lines in found.items():
            with self.subTest(case=name):
                first_norm = (
                    f"recording.{name}[0].logged[0].metrics.grad_norm: values differ"
                )
                self.assertNotIn(first_norm, lines)
                self.assertTrue(
                    any(
                        line.startswith(f"recording.{name}[0].shared.draft_state_dict")
                        for line in lines
                    )
                )


if __name__ == "__main__":
    unittest.main()
