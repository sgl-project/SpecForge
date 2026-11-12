"""A new native target needs no changes to the public initialization/export paths."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file

from specforge.checkpoint.formats.sglang.native import write_native
from specforge.checkpoint.native import (
    _TARGETS,
    register_native_target,
    validate_target_initialization,
)
from specforge.checkpoint.tensors import TensorStore
from specforge.export.to_sglang import export_to_sglang


class NativeRegistryTest(unittest.TestCase):
    def test_registered_native_uses_its_own_validation_and_layout(self):
        config = SimpleNamespace(architecture="ExternalNative", target_bindings=())
        state = {"weight": torch.ones(2)}
        validate = Mock()

        def export(config, state, source, payload, output, *, copy_target):
            write_native(
                config,
                {"draft_block.weight": state["weight"]},
                source,
                payload,
                output,
                prefix="draft_block.",
                binding_names={},
                copy_target=copy_target,
            )

        with patch.dict(_TARGETS), TemporaryDirectory() as tmp:
            register_native_target(
                "external_target",
                config.architecture,
                source_prefix="draft_block",
                validate=validate,
                export=export,
            )
            target = Path(tmp) / "target"
            target.mkdir()
            payload = {"model_type": "external_target"}
            (target / "config.json").write_text(json.dumps(payload))
            save_file(
                {"draft_block.weight": torch.zeros(2), "target.weight": torch.ones(3)},
                target / "model.safetensors",
            )
            validate_target_initialization(config, payload, {"": "draft_block"})
            validate.assert_called_once_with(config, payload)
            checkpoint = SimpleNamespace(config=config)
            with (
                patch(
                    "specforge.export.to_sglang.read_export",
                    return_value=(checkpoint, state, None),
                ),
                patch(
                    "specforge.export.to_sglang.convert_config",
                    side_effect=AssertionError("generic converter"),
                ),
            ):
                output = Path(tmp) / "output"
                export_to_sglang(
                    "checkpoint", None, output, target_model=target, copy_target=True
                )
            self.assertEqual(validate.call_count, 2)
            store = TensorStore(output)
            torch.testing.assert_close(
                store.read("draft_block.weight"), state["weight"]
            )
            torch.testing.assert_close(store.read("target.weight"), torch.ones(3))
