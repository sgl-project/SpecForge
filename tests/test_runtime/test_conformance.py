"""The conformance kit holds for the reference runner, and it can fail."""

import contextlib
import copy
import unittest
from unittest.mock import patch

from specforge.algorithms.objectives.registry import (
    protocol_registrations,
    resolve_objective,
)
from specforge.training.contracts import StepContext
from specforge.training.controller import TrainerController
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
    protocol_violations,
    reference_runner,
)
from tests.test_runtime.trajectories import cases, differences, record


class ReferenceProtocolTest(ProtocolChecks, unittest.TestCase):
    """The run loop over the DDP / FSDP runner."""


class ReferenceRunnerTest(RunnerChecks, unittest.TestCase):
    """The DDP / FSDP runner against itself: the jobs are deterministic."""


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
        def losing(key):
            class Runner(DataParallelRunner):
                def load_resume(self, state):
                    optimizer = dict(copy.deepcopy(state["optimizer"]))
                    if key == "scheduler_state_dict":
                        # A schedule that starts over.
                        optimizer[key] = self.backend.optimizer.scheduler.state_dict()
                    else:
                        del optimizer[key]
                    super().load_resume({**state, "optimizer": optimizer})

            return Runner

        for key in ("fp32_params", "scheduler_state_dict"):
            with self.subTest(lost=key):
                self.assertEqual(
                    self.failures(
                        losing(key), "test_resume_continues_the_same_trajectory"
                    ),
                    1,
                )

    def test_checkpoint_that_does_not_wait_for_commits_fails_the_protocol(self):
        class Probe(ProtocolChecks, unittest.TestCase):
            pass

        result = unittest.TestResult()
        with patch.object(TrainerController, "_flush_durable_ack", lambda self: None):
            Probe("test_commits_are_flushed_before_a_checkpoint").run(result)
        self.assertEqual((len(result.failures), result.errors), (1, []))

    def test_ledger_that_breaks_the_commit_contract_is_reported(self):
        good = [
            ("batch", "s0"),
            ("forward", "s0"),
            ("batch", "s1"),
            ("forward", "s1"),
            ("update",),
            ("commit", ("s0", "s1"), 1),
            ("checkpoint", 1),
        ]
        self.assertEqual(protocol_violations(good), [])
        batch_0, forward_0, batch_1, forward_1, update, commit, checkpoint = good
        for broken, message in (
            (
                [batch_0, forward_0, batch_1, forward_1, commit, update, checkpoint],
                "step 1 was committed after 0 updates",
            ),
            (
                [batch_0, forward_0, batch_1, forward_1, update, checkpoint, commit],
                "checkpoint 1 was written after 0 commits",
            ),
            (
                # Read and acknowledged, never handed to the runner.
                [batch_0, forward_0, forward_0, update, batch_1, commit],
                "step 1 committed ('s0', 's1'); its update consumed ('s0', 's0')",
            ),
            (
                [batch_0, forward_0, update, ("commit", ("s0",), 1)],
                "update 1 consumed ('s0',), not a window of 2",
            ),
            (
                good[:5] + [("commit", ("s1", "s0"), 1)],
                "step 1 committed ('s1', 's0'); its update consumed ('s0', 's1')",
            ),
            (
                good[:5] + [("raised", "IncompleteWindow"), commit],
                "step 1 was committed after the run raised",
            ),
            (good[:5], "1 updates were applied and 0 committed"),
        ):
            with self.subTest(expected=message):
                self.assertEqual(protocol_violations(broken), [message])
        self.assertEqual(
            protocol_violations(good[:6] + [commit]),
            [
                "commit of step 1 follows 1 commits",
                "1 updates were applied and 2 committed",
            ],
        )

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


if __name__ == "__main__":
    unittest.main()
