"""Configuration registration is usable independently of model implementations."""

import subprocess
import sys
import textwrap
import unittest
from dataclasses import dataclass
from unittest import mock

from transformers import LlamaConfig, Qwen3Config

from specforge.config.draft import AutoDraftConfig, DraftConfig, register_draft_config
from specforge.config.draft.registry import DRAFT_CONFIG_REGISTRY


class DraftConfigRegistryTest(unittest.TestCase):
    def test_training_and_checkpoint_validation_do_not_import_modeling(self):
        code = textwrap.dedent(
            """
            import importlib.abc
            import sys
            from pathlib import Path

            class RejectModelImports(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "specforge.modeling" or fullname.startswith("specforge.modeling."):
                        raise AssertionError("configuration imported model implementation: " + fullname)

            sys.meta_path.insert(0, RejectModelImports())
            from specforge.config import load_config
            from specforge.config.draft import AutoDraftConfig
            from specforge.config.draft.contract import draft_config_contract
            from specforge.algorithms.builtin import builtin_algorithm_registry
            from specforge.training.model_loading import resolve_draft_config

            registry = builtin_algorithm_registry()
            for path in Path("examples/configs").rglob("*.yaml"):
                run = load_config(str(path))
                provider = registry.resolve(run.training.strategy).providers.model.draft_config
                config = resolve_draft_config(run, provider=provider)
                saved = AutoDraftConfig.from_dict(config.to_dict())
                assert draft_config_contract(config) == draft_config_contract(saved), str(path)
            assert not any(name.startswith("specforge.modeling") for name in sys.modules)
        """
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_custom_config_parses_and_validates_without_model_registration(self):
        @dataclass(frozen=True, kw_only=True)
        class Options(DraftConfig):
            width: int

            @classmethod
            def from_model_config(cls, config):
                if config.width <= 0:
                    raise ValueError("width must be positive")
                return cls(width=config.width)

            @classmethod
            def validate_training_config(cls, config):
                if config.width < 2:
                    raise ValueError("training requires width >= 2")

        with mock.patch.dict(DRAFT_CONFIG_REGISTRY):
            register_draft_config("ConfigOnlyDraft", LlamaConfig, Options)
            config = AutoDraftConfig.from_dict(
                {"architectures": ["ConfigOnlyDraft"], "width": 4}
            )
            finalized = AutoDraftConfig.finalize(config)
            AutoDraftConfig.validate_training_config(finalized)
            self.assertEqual(finalized.width, 4)
            self.assertIsNot(finalized, config)
            config.width = 0
            with self.assertRaisesRegex(ValueError, "width must be positive"):
                AutoDraftConfig.finalize(config)
            config.width = 1
            with self.assertRaisesRegex(ValueError, "training requires"):
                AutoDraftConfig.validate_training_config(config)

    def test_registration_is_idempotent_and_conflicts_are_atomic(self):
        with mock.patch.dict(DRAFT_CONFIG_REGISTRY):
            register_draft_config("ConfigOnlyDraft", LlamaConfig)
            registration = DRAFT_CONFIG_REGISTRY["ConfigOnlyDraft"]
            register_draft_config("ConfigOnlyDraft", LlamaConfig)
            with self.assertRaisesRegex(ValueError, "different registration"):
                register_draft_config("ConfigOnlyDraft", Qwen3Config)
            self.assertEqual(DRAFT_CONFIG_REGISTRY["ConfigOnlyDraft"], registration)

    def test_invalid_registration_is_rejected(self):
        for name, config_type, options_type in (
            ("", LlamaConfig, None),
            ("test", object, None),
            ("test", LlamaConfig, object),
        ):
            with self.subTest(
                name=name, config_type=config_type, options_type=options_type
            ):
                with self.assertRaises((TypeError, ValueError)):
                    register_draft_config(name, config_type, options_type)

    def test_model_registration_must_match_pre_registered_config(self):
        from specforge.modeling.draft.registry import DRAFT_REGISTRY, register_draft

        class WrongModel:
            config_class = Qwen3Config

        with mock.patch.dict(DRAFT_CONFIG_REGISTRY), mock.patch.dict(DRAFT_REGISTRY):
            register_draft_config("ConfigOnlyDraft", LlamaConfig)
            with self.assertRaisesRegex(ValueError, "different registration"):
                register_draft(WrongModel, name="ConfigOnlyDraft")
            self.assertNotIn("ConfigOnlyDraft", DRAFT_REGISTRY)
