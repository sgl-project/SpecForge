"""What an objective model declares to a runner, read from its registration."""

import unittest

from torch import nn

from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.contracts import PartitionHints, ProgramSpec
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
                        partition=PartitionHints(
                            blocks=tuple(runtime.draft_model._no_split_modules)
                        ),
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
        self.assertFalse(specs["DSparkDraftModel"].evaluation_diagnostics)
        # A block draft is sharded per decoder layer; EAGLE is one unit.
        self.assertEqual(specs["DFlashDraftModel"].partition.blocks, ("DecoderLayer",))
        self.assertEqual(specs["Eagle3DraftModel"].partition.blocks, ())

    def test_model_without_a_registration_declares_nothing(self):
        _, runtime = build_runtime("DFlashDraftModel")
        runtime.registration = None
        with self.assertRaisesRegex(ValueError, "without a registration"):
            program_spec(runtime)


class FrozenModulesTest(unittest.TestCase):
    def test_bound_tables_and_the_teacher_head_are_frozen(self):
        for config in configurations():
            with self.subTest(architecture=config.architecture):
                _, runtime = build_runtime(config.architecture)
                draft = runtime.draft_model
                bound = [draft.get_submodule(name) for name in config.target_bindings]
                self.assertEqual(
                    [id(module) for module in runtime.frozen],
                    [id(module) for module in bound + [runtime.target_head]],
                )
                self.assertFalse(
                    any(
                        parameter.requires_grad
                        for module in runtime.frozen
                        for parameter in module.parameters()
                    )
                )

    def test_head_shared_with_the_draft_is_listed_once(self):
        # DSpine binds its embedding and its LM head to the target's tables.
        config, runtime = build_runtime("DSpineDraftModel")
        self.assertEqual(config.target_bindings, ("embed_tokens", "lm_head"))
        runtime.target_head = runtime.draft_model.lm_head
        self.assertEqual(
            [id(module) for module in runtime.frozen],
            [id(runtime.draft_model.embed_tokens), id(runtime.draft_model.lm_head)],
        )

    def test_backend_refuses_a_trainable_module_as_frozen(self):
        model = nn.Linear(2, 2)
        backend = FSDPTrainingBackend(ParallelConfig())
        with self.assertRaisesRegex(ValueError, "only frozen parameters"):
            backend.prepare_model(model, wrap=False, frozen=(model,))
        model.requires_grad_(False)
        self.assertIs(backend.prepare_model(model, wrap=False, frozen=(model,)), model)


if __name__ == "__main__":
    unittest.main()
