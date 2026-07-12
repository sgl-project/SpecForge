# coding=utf-8
"""CPU/mock checks for Domino-only FSDP memory options."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

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
    def _build(self, strategy_name):
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

    def test_domino_excludes_frozen_target_modules_and_wraps_draft_blocks(self):
        model, options = self._build("domino")
        self.assertEqual(options["optimizer_target"], model.draft_model)
        self.assertEqual(
            options["ignored_modules"], [model.lm_head, model.embed_tokens]
        )
        policy = options["auto_wrap_policy"]
        self.assertIsNotNone(policy)
        self.assertEqual(policy.keywords["transformer_layer_cls"], {_TransformerBlock})
        self.assertEqual(options["parallel_options"]["sharding_strategy"], "FULL_SHARD")

    def test_non_domino_strategies_do_not_receive_domino_fsdp_options(self):
        model, options = self._build("dflash")
        self.assertEqual(options["optimizer_target"], model.draft_model)
        self.assertNotIn("auto_wrap_policy", options)
        self.assertNotIn("ignored_modules", options)
        self.assertEqual(
            options["parallel_options"]["sharding_strategy"], "SHARD_GRAD_OP"
        )


if __name__ == "__main__":
    unittest.main()
