"""What an objective model declares to a runner, read from its registration."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from torch import nn
from torch.nn.parallel import DistributedDataParallel

from specforge.algorithms.objectives.base import ObjectiveModel
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.contracts import PartitionHints, ProgramSpec
from specforge.training.program import program_spec
from tests.test_algorithms.test_objective_contracts import build_runtime
from tests.test_algorithms.test_objectives import configurations
from tests.test_runtime.test_canonical_end_to_end import (
    build,
    drafts,
    process_group,
    write_run,
)


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
                            blocks=tuple(sorted(runtime.draft_model._no_split_modules))
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
        # In one order on every rank, whatever order the draft holds them in.
        _, peagle = build_runtime("PEagleDraftModel")
        for held in (
            ["EagleDecoderStage", "DecoderLayer"],
            {"DecoderLayer", "EagleDecoderStage"},
        ):
            peagle.draft_model._no_split_modules = held
            self.assertEqual(
                program_spec(peagle).partition.blocks,
                ("DecoderLayer", "EagleDecoderStage"),
            )

    def test_draft_that_names_no_block_is_one_unit(self):
        _, runtime = build_runtime("DFlashDraftModel")
        for held in (None, [], ()):
            with self.subTest(held=held):
                runtime.draft_model._no_split_modules = held
                self.assertEqual(program_spec(runtime).partition.blocks, ())
        # A draft that is not a canonical draft class has no such attribute.
        plain = nn.Linear(2, 2)
        runtime.draft_model = plain
        self.assertFalse(hasattr(plain, "_no_split_modules"))
        self.assertEqual(program_spec(runtime).partition.blocks, ())

    def test_model_without_a_registration_declares_nothing(self):
        _, runtime = build_runtime("DFlashDraftModel")
        runtime.registration = None
        with self.assertRaisesRegex(ValueError, "without a registration"):
            program_spec(runtime)


class FrozenModulesTest(unittest.TestCase):
    def test_modules_are_frozen_where_the_builder_declares_them(self):
        """A model built without the declaration shards as it always did."""
        _, runtime = build_runtime("DSpineDraftModel")
        self.assertEqual(runtime.frozen, ())
        table = nn.Embedding(4, 2)
        declared = ObjectiveModel(
            runtime.draft_model,
            runtime.objective,
            loss_configuration=runtime.loss_configuration,
            frozen_modules=[table, runtime.draft_model.lm_head, table],
        )
        # Each once, in the order given.
        self.assertEqual(
            [id(module) for module in declared.frozen],
            [id(table), id(runtime.draft_model.lm_head)],
        )

    def test_assembled_run_keeps_bound_tables_and_the_teacher_replicated(self):
        for draft in drafts():
            with (
                self.subTest(architecture=draft.architecture),
                TemporaryDirectory() as temp,
                process_group(Path(temp)),
            ):
                trainer = build(write_run(Path(temp), draft)).trainer
                program = trainer.runner.program
                model = (
                    program.module
                    if isinstance(program, DistributedDataParallel)
                    else program
                )
                bound = [
                    model.draft_model.get_submodule(name)
                    for name in draft.target_bindings
                ]
                teacher = [] if model.target_head is None else [model.target_head]
                expected = list(dict.fromkeys(bound + teacher))
                self.assertEqual(
                    [id(module) for module in model.frozen],
                    [id(module) for module in expected],
                )
                # What the program declares is what the backend left alone.
                self.assertEqual(trainer.backend.ignored_frozen_modules, model.frozen)
                self.assertFalse(
                    any(
                        parameter.requires_grad
                        for module in model.frozen
                        for parameter in module.parameters()
                    )
                )

    def test_sharding_wrapper_is_given_the_frozen_modules_and_the_blocks(self):
        """What the backend is told is what reaches FSDP's constructor."""

        class Block(nn.Linear):
            pass

        model = nn.Sequential(Block(2, 2), nn.Linear(2, 2), nn.Embedding(4, 2))
        table = model[2]
        table.requires_grad_(False)
        backend = FSDPTrainingBackend(ParallelConfig(sharding_strategy="SHARD_GRAD_OP"))
        with patch(
            "torch.distributed.fsdp.FullyShardedDataParallel", return_value=model
        ) as wrapper:
            backend.prepare_model(model, frozen=(table,), blocks=("Block",))
        ((wrapped,), arguments) = wrapper.call_args
        self.assertIs(wrapped, model)
        self.assertEqual(arguments["ignored_modules"], (table,))
        self.assertEqual(
            arguments["auto_wrap_policy"].keywords["transformer_layer_cls"], {Block}
        )
        # Without declarations: nothing ignored, one root unit.
        with patch(
            "torch.distributed.fsdp.FullyShardedDataParallel", return_value=model
        ) as wrapper:
            FSDPTrainingBackend(
                ParallelConfig(sharding_strategy="SHARD_GRAD_OP")
            ).prepare_model(model)
        self.assertNotIn("ignored_modules", wrapper.call_args.kwargs)
        self.assertNotIn("auto_wrap_policy", wrapper.call_args.kwargs)

    def test_backend_refuses_a_trainable_module_as_frozen(self):
        model = nn.Linear(2, 2)
        backend = FSDPTrainingBackend(ParallelConfig())
        with self.assertRaisesRegex(ValueError, "only frozen parameters"):
            backend.prepare_model(model, wrap=False, frozen=(model,))
        model.requires_grad_(False)
        self.assertIs(backend.prepare_model(model, wrap=False, frozen=(model,)), model)


if __name__ == "__main__":
    unittest.main()
