"""What an objective model declares to a runner, read from its registration."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.nn.parallel import DistributedDataParallel

from specforge.algorithms.objectives import build_objective_model
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
        self.assertEqual(specs["Eagle3DraftModel"].axes, ())
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

    def test_draft_without_the_block_declaration_is_refused(self):
        """Every canonical draft class has it; a missing one is not "no blocks"."""
        _, runtime = build_runtime("DFlashDraftModel")
        runtime.draft_model._no_split_modules = None
        with self.assertRaises(TypeError):
            program_spec(runtime)
        runtime.draft_model = nn.Linear(2, 2)
        with self.assertRaises(AttributeError):
            program_spec(runtime)

    def test_model_without_a_registration_declares_nothing(self):
        _, runtime = build_runtime("DFlashDraftModel")
        runtime.registration = None
        with self.assertRaisesRegex(ValueError, "without a registration"):
            program_spec(runtime)

    def test_sequence_parallel_program_declares_part_axis(self):
        from types import SimpleNamespace

        from specforge.training.sequence_parallel import SequenceParallel

        group = SimpleNamespace(size=lambda: 2)
        _, runtime = build_runtime(
            "Eagle3DraftModel",
            attention_backend="usp",
            sequence_parallel=SequenceParallel(group, group, group),
        )
        spec = program_spec(runtime)
        self.assertEqual(spec.axes, (("sequence", "part"),))
        # A value: it hashes, as every other declaration does.
        self.assertEqual(hash(spec), hash(program_spec(runtime)))


class FrozenModulesTest(unittest.TestCase):
    def test_modules_are_frozen_where_the_builder_declares_them(self):
        """A model built without the declaration shards as it always did."""
        _, runtime = build_runtime("DSpineDraftModel")
        self.assertEqual(runtime.frozen_modules, ())
        draft = runtime.draft_model
        declared = ObjectiveModel(
            draft,
            runtime.objective,
            loss_configuration=runtime.loss_configuration,
            frozen_modules=[draft.embed_tokens, draft.lm_head, draft.embed_tokens],
        )
        # Each once, in the order given.
        self.assertEqual(
            [id(module) for module in declared.frozen_modules],
            [id(draft.embed_tokens), id(draft.lm_head)],
        )

    def test_module_the_model_does_not_hold_is_refused(self):
        _, runtime = build_runtime("DSpineDraftModel")
        with self.assertRaisesRegex(ValueError, r"does not: \['Embedding'\]"):
            ObjectiveModel(
                runtime.draft_model,
                runtime.objective,
                loss_configuration=runtime.loss_configuration,
                frozen_modules=[runtime.draft_model.lm_head, nn.Embedding(4, 2)],
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
                    [id(module) for module in model.frozen_modules],
                    [id(module) for module in expected],
                )
                self.assertFalse(
                    any(
                        parameter.requires_grad
                        for module in model.frozen_modules
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
            backend.prepare_model(model, frozen_modules=(table,), blocks=("Block",))
        ((wrapped,), arguments) = wrapper.call_args
        self.assertIs(wrapped, model)
        self.assertEqual(arguments["ignored_modules"], (table,))
        self.assertEqual(
            arguments["auto_wrap_policy"].keywords["transformer_layer_cls"], {Block}
        )
        self.assertEqual(backend.ignored_frozen_modules, (table,))
        self.assertEqual(backend.auto_wrap_block_classes, {Block})
        # The program receives its inputs as placed, as under DDP: the wrapper
        # casts parameters, never the program's tensors.
        precision = arguments["mixed_precision"]
        self.assertFalse(precision.cast_root_forward_inputs)
        self.assertFalse(precision.cast_forward_inputs)
        # Without declarations: nothing ignored, one root unit.
        with patch(
            "torch.distributed.fsdp.FullyShardedDataParallel", return_value=model
        ) as wrapper:
            FSDPTrainingBackend(
                ParallelConfig(sharding_strategy="SHARD_GRAD_OP")
            ).prepare_model(model)
        self.assertNotIn("ignored_modules", wrapper.call_args.kwargs)
        self.assertNotIn("auto_wrap_policy", wrapper.call_args.kwargs)

    def test_replicating_wrapper_records_nothing_as_ignored_or_wrapped(self):
        """DDP takes neither declaration, so the backend reports neither."""

        class Block(nn.Linear):
            pass

        model = nn.Sequential(Block(2, 2), nn.Embedding(4, 2))
        model[1].requires_grad_(False)
        backend = FSDPTrainingBackend(ParallelConfig(sharding_strategy="NO_SHARD"))
        with patch(
            "torch.nn.parallel.DistributedDataParallel", return_value=model
        ) as wrapper:
            backend.prepare_model(model, frozen_modules=(model[1],), blocks=("Block",))
        wrapper.assert_called_once()
        self.assertEqual(backend.ignored_frozen_modules, ())
        self.assertEqual(backend.auto_wrap_block_classes, set())

    def test_blocks_that_name_no_module_class_are_refused(self):
        """A misspelt block would otherwise train as one root unit."""

        class Block(nn.Linear):
            pass

        model = nn.Sequential(Block(2, 2), nn.Linear(2, 2))
        backend = FSDPTrainingBackend(ParallelConfig(sharding_strategy="SHARD_GRAD_OP"))
        with self.assertRaisesRegex(ValueError, r"\['Blocks'\] name no module class"):
            backend.prepare_model(model, blocks=("Blocks",))
        # A draft class may name a block that only some of its models hold.
        with patch(
            "torch.distributed.fsdp.FullyShardedDataParallel", return_value=model
        ) as wrapper:
            backend.prepare_model(model, blocks=("Block", "Stage"))
        self.assertEqual(
            wrapper.call_args.kwargs["auto_wrap_policy"].keywords[
                "transformer_layer_cls"
            ],
            {Block},
        )

    def test_backend_refuses_a_trainable_module_as_frozen(self):
        model = nn.Linear(2, 2)
        backend = FSDPTrainingBackend(ParallelConfig())
        with self.assertRaisesRegex(ValueError, "only frozen parameters"):
            backend.prepare_model(model, wrap=False, frozen_modules=(model,))
        model.requires_grad_(False)
        self.assertIs(
            backend.prepare_model(model, wrap=False, frozen_modules=(model,)), model
        )


def sharded_rank(rank, directory):
    """Wrap a draft with bound tables on one of two ranks; save what it holds."""
    torch.set_num_threads(1)
    accelerated = torch.cuda.is_available()
    if accelerated:
        torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl" if accelerated else "gloo",
        init_method=f"file://{directory}/rendezvous",
        rank=rank,
        world_size=2,
    )
    try:
        config, built = build_runtime("DSpineDraftModel")
        draft, teacher = built.draft_model, built.target_head
        program = build_objective_model(
            draft,
            target_head=teacher,
            frozen_modules=[
                draft.get_submodule(name) for name in config.target_bindings
            ]
            + [teacher],
        )
        if accelerated:
            program.cuda()

        def elements(modules):
            return sum(p.numel() for module in modules for p in module.parameters())

        blocks = list(draft.backbone.layers)
        whole = {
            "frozen": elements(program.frozen_modules),
            "blocks": elements(blocks),
        }
        backend = FSDPTrainingBackend(
            ParallelConfig(world_size=2, sharding_strategy="SHARD_GRAD_OP")
        )
        wrapped = backend.prepare_model(
            program,
            optimizer_target=draft,
            frozen_modules=program.frozen_modules,
            blocks=program_spec(program).partition.blocks,
        )
        units = [
            module.module
            for module in wrapped.modules()
            if isinstance(module, FullyShardedDataParallel) and module is not wrapped
        ]
        torch.save(
            {
                "declared": len(program.frozen_modules),
                "units_are_the_blocks": len(units) == len(blocks)
                and all(unit is block for unit, block in zip(units, blocks)),
                "whole": whole,
                # Outside a forward a rank holds its shard of a sharded parameter.
                "held": {
                    "frozen": elements(program.frozen_modules),
                    "blocks": elements(blocks),
                },
            },
            Path(directory) / f"held-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.device_count() >= 2,
    "FSDP shards on accelerators; two ranks need two",
)
class ShardedFrozenModulesTest(unittest.TestCase):
    def test_two_ranks_shard_the_blocks_and_keep_frozen_modules_whole(self):
        with TemporaryDirectory() as directory:
            mp.spawn(sharded_rank, args=(directory,), nprocs=2, join=True)
            ranks = [
                torch.load(Path(directory) / f"held-{rank}.pt") for rank in range(2)
            ]
        for held in ranks:
            # The bound tables and the teacher, and each decoder layer a unit.
            self.assertEqual(held["declared"], 3)
            self.assertTrue(held["units_are_the_blocks"])
            self.assertEqual(held["held"]["frozen"], held["whole"]["frozen"])
            self.assertLess(held["held"]["blocks"], held["whole"]["blocks"])
        self.assertEqual(
            sum(held["held"]["blocks"] for held in ranks), ranks[0]["whole"]["blocks"]
        )


if __name__ == "__main__":
    unittest.main()
