# coding=utf-8
"""CPU/mock checks for Domino-only FSDP memory options."""

import ast
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch.nn as nn


def _module(name, **members):
    module = types.ModuleType(name)
    module.__dict__.update(members)
    return module


def _load_trainer_module():
    """Load trainer.py against light seam fakes, without GPU/model imports."""
    repo = Path(__file__).resolve().parents[2]
    module_names = (
        "specforge",
        "specforge.runtime",
        "specforge.runtime.data_plane",
        "specforge.training",
        "specforge.training.backend",
        "specforge.training.checkpoint",
        "specforge.training.controller",
        "specforge.training.strategies",
        "specforge.training.strategies.registry",
    )
    saved = {name: sys.modules.get(name) for name in module_names}
    try:
        for name in (
            "specforge",
            "specforge.runtime",
            "specforge.training",
            "specforge.training.strategies",
        ):
            package = types.ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package

        class Placeholder:
            pass

        sys.modules["specforge.runtime.data_plane"] = _module(
            "specforge.runtime.data_plane",
            FeatureDataLoader=Placeholder,
            FeatureStore=Placeholder,
        )
        sys.modules["specforge.training.backend"] = _module(
            "specforge.training.backend",
            FSDPTrainingBackend=Placeholder,
            ParallelConfig=Placeholder,
        )
        sys.modules["specforge.training.checkpoint"] = _module(
            "specforge.training.checkpoint", CheckpointManager=Placeholder
        )
        sys.modules["specforge.training.controller"] = _module(
            "specforge.training.controller",
            TrainerController=Placeholder,
            TrainerCore=Placeholder,
        )
        sys.modules["specforge.training.strategies.registry"] = _module(
            "specforge.training.strategies.registry",
            StrategySpec=Placeholder,
            resolve_strategy=lambda name: name,
        )

        spec = importlib.util.spec_from_file_location(
            "_domino_test_trainer", repo / "specforge" / "training" / "trainer.py"
        )
        trainer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer_module)
        return trainer_module
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class _TransformerBlock(nn.Linear):
    pass


class _Draft(nn.Module):
    _no_split_modules = ["_TransformerBlock"]

    def __init__(self):
        super().__init__()
        self.block = _TransformerBlock(3, 3, bias=False)


class _Composite(nn.Module):
    def __init__(self):
        super().__init__()
        self.draft_model = _Draft()
        self.lm_head = nn.Linear(3, 5, bias=False).requires_grad_(False)
        self.embed_tokens = nn.Embedding(5, 3).requires_grad_(False)


class TestDominoFSDPWiring(unittest.TestCase):
    def _build(self, strategy_name, *, auto_wrap=False):
        trainer_module = _load_trainer_module()
        captured = {}

        class FakeLoader:
            def __init__(self, *args, **kwargs):
                pass

        class FakeParallel:
            @classmethod
            def from_distributed(cls, **kwargs):
                captured["parallel_options"] = kwargs
                return object()

        class FakeBackend:
            def __init__(self, parallel, *, optimizer_factory):
                pass

            def prepare_model(self, model, **kwargs):
                captured.update(kwargs)
                return model

        class FakeCore:
            def __init__(self, *args, **kwargs):
                pass

        class FakeControllerRuntime:
            def __init__(self, *args, **kwargs):
                self.global_step = 0

        class FakeCheckpointManager:
            def __init__(self, *args, **kwargs):
                pass

        trainer_module.FeatureDataLoader = FakeLoader
        trainer_module.ParallelConfig = FakeParallel
        trainer_module.FSDPTrainingBackend = FakeBackend
        trainer_module.TrainerCore = FakeCore
        trainer_module.TrainerController = FakeControllerRuntime
        trainer_module.CheckpointManager = FakeCheckpointManager

        class ControlPlane:
            def register_trainer(self, metadata):
                return "trainer-0"

        class Spec:
            name = strategy_name

            @staticmethod
            def make_strategy(wrapped, *, target_head=None, **kwargs):
                return object()

        model = _Composite()
        with mock.patch.dict(os.environ, {"FSDP_AUTO_WRAP": "1" if auto_wrap else "0"}):
            trainer_module.Trainer(
                spec=Spec(),
                controller=ControlPlane(),
                store=object(),
                ref_source={"queue": object()},
                model=model,
                target_head=None,
                optimizer_factory=None,
                run_id="run",
                output_dir="/tmp/out",
                batch_size=1,
                accumulation_steps=1,
                num_epochs=1,
                max_steps=1,
                save_interval=0,
                eval_interval=0,
                tp_size=1,
                sp_ulysses_size=1,
                sp_ring_size=1,
                logger=None,
                log_interval=1,
                collate_fn=lambda features: features,
            )
        return model, captured

    def test_domino_keeps_checkpoint_compatible_root_wrap_by_default(self):
        model, options = self._build("domino")
        self.assertEqual(options["optimizer_target"], model.draft_model)
        self.assertNotIn("auto_wrap_policy", options)
        self.assertNotIn("ignored_modules", options)
        self.assertEqual(
            options["parallel_options"]["sharding_strategy"], "SHARD_GRAD_OP"
        )

    def test_domino_auto_wrap_is_explicit_opt_in(self):
        model, options = self._build("domino", auto_wrap=True)
        self.assertEqual(options["optimizer_target"], model.draft_model)
        self.assertEqual(
            options["ignored_modules"], [model.lm_head, model.embed_tokens]
        )
        policy = options["auto_wrap_policy"]
        self.assertIsNotNone(policy)
        self.assertEqual(policy.keywords["transformer_layer_cls"], {_TransformerBlock})
        self.assertEqual(
            options["parallel_options"]["sharding_strategy"], "SHARD_GRAD_OP"
        )

    def test_non_domino_strategies_do_not_receive_domino_fsdp_options(self):
        model, options = self._build("dflash", auto_wrap=True)
        self.assertEqual(options["optimizer_target"], model.draft_model)
        self.assertNotIn("auto_wrap_policy", options)
        self.assertNotIn("ignored_modules", options)
        self.assertEqual(
            options["parallel_options"]["sharding_strategy"], "SHARD_GRAD_OP"
        )

    def test_standalone_checkpoint_offloads_full_state_to_rank0_cpu(self):
        repo = Path(__file__).resolve().parents[2]
        tree = ast.parse((repo / "scripts" / "train_domino.py").read_text())
        configs = []
        state_dict_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name) and node.value.func.id == (
                    "FullStateDictConfig"
                ):
                    configs.append(node)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "state_dict_type":
                    state_dict_calls.append(node)

        self.assertEqual(len(configs), 1)
        config = configs[0]
        keywords = {keyword.arg: keyword.value for keyword in config.value.keywords}
        self.assertTrue(ast.literal_eval(keywords["offload_to_cpu"]))
        self.assertTrue(ast.literal_eval(keywords["rank0_only"]))
        config_name = config.targets[0].id
        self.assertTrue(
            any(
                any(
                    isinstance(arg, ast.Name) and arg.id == config_name
                    for arg in call.args
                )
                for call in state_dict_calls
            )
        )


if __name__ == "__main__":
    unittest.main()
