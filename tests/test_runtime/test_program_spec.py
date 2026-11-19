"""What an objective model declares to a runner, read from its registration."""

import unittest

from specforge.training.contracts import ProgramSpec
from specforge.training.program import program_spec
from tests.test_algorithms.test_objective_contracts import build_runtime
from tests.test_algorithms.test_objectives import configurations


class ProgramSpecTest(unittest.TestCase):
    def test_declarations_come_from_the_registration(self):
        for config in configurations():
            with self.subTest(architecture=config.architecture):
                _, runtime = build_runtime(config.architecture)
                training = runtime.registration.training
                statistic = training.window_statistic
                self.assertEqual(
                    program_spec(runtime),
                    ProgramSpec(
                        name=runtime.objective_name,
                        inputs=tuple(runtime.registration.inputs),
                        replay_statistics=(
                            () if statistic is None else (statistic.argument_name,)
                        ),
                        evaluation_diagnostics=training.evaluation_diagnostics,
                        reduction=training.reduction,
                    ),
                )

    def test_declarations_that_differ_between_objectives(self):
        specs = {
            architecture: program_spec(build_runtime(architecture)[1])
            for architecture in (
                "DFlashDraftModel",
                "DSparkDraftModel",
                "DSpineDraftModel",
                "Eagle3DraftModel",
            )
        }
        self.assertEqual(
            {name: spec.reduction for name, spec in specs.items()},
            {
                "DFlashDraftModel": "window",
                "DSparkDraftModel": "microbatch",
                "DSpineDraftModel": "window",
                "Eagle3DraftModel": "window",
            },
        )
        self.assertEqual(specs["DSpineDraftModel"].replay_statistics, ("alignment_ce",))
        self.assertEqual(specs["DFlashDraftModel"].replay_statistics, ())
        self.assertTrue(specs["Eagle3DraftModel"].evaluation_diagnostics)
        self.assertFalse(specs["DFlashDraftModel"].evaluation_diagnostics)
        self.assertIn("hidden_states", specs["DFlashDraftModel"].inputs)

    def test_model_without_a_registration_declares_nothing(self):
        _, runtime = build_runtime("DFlashDraftModel")
        runtime.registration = None
        with self.assertRaisesRegex(ValueError, "without a registration"):
            program_spec(runtime)


if __name__ == "__main__":
    unittest.main()
