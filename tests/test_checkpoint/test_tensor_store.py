"""Selective checkpoint loading rejects quantized copies without reading payloads."""

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from specforge.checkpoint import TensorStore, make_weight_plan
from specforge.modeling import build_draft_model
from specforge.modeling.target.checkpoint import list_checkpoint_keys
from tests.test_config.draft_fixtures import draft_config


class TensorStoreTest(unittest.TestCase):
    def test_selected_quantized_weights_fail_in_both_formats(self):
        model = build_draft_model(draft_config(), dtype=torch.bfloat16)
        for suffix in ("safetensors", "pt"):
            with self.subTest(format=suffix), TemporaryDirectory() as tmp:
                path = Path(tmp) / ("weights." + suffix)
                state = {
                    "embed.weight": torch.ones(64, 32, dtype=torch.bfloat16),
                    "quant.weight": torch.zeros(64, 32).to(torch.float8_e4m3fn),
                    "scaled.weight": torch.ones(64, 32, dtype=torch.bfloat16),
                    "scaled.weight_scale_inv": torch.ones(1),
                }
                if suffix == "pt":
                    torch.save(state, path)
                else:
                    save_file(state, path)
                store = TensorStore(path)
                refs = store.refs()
                bindings = {
                    name: refs["embed.weight"]
                    for name in model.config.draft.target_bindings
                }
                with patch.object(
                    TensorStore, "read", side_effect=AssertionError("preflight read")
                ):
                    make_weight_plan(
                        model, {}, bindings=bindings, allow_new=model.state_dict()
                    )
                    for key in ("quant.weight", "scaled.weight"):
                        with self.assertRaisesRegex(ValueError, key):
                            make_weight_plan(
                                model, {}, bindings=bindings | {"lm_head": refs[key]}
                            )
                        # Diagnose quantization before reporting unrelated/scale keys.
                        with self.assertRaisesRegex(ValueError, key):
                            make_weight_plan(
                                model,
                                {
                                    "lm_head.weight": refs[key],
                                    "scale": refs["scaled.weight_scale_inv"],
                                },
                            )
                    store.metadata["quant.weight"] = replace(
                        store.metadata["quant.weight"], dtype="F4"
                    )
                    with self.assertRaisesRegex(ValueError, "quant.weight"):
                        make_weight_plan(
                            model,
                            {},
                            bindings=bindings | {"lm_head": refs["quant.weight"]},
                        )

    def test_pt_payload_is_loaded_once_per_file(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_state.pt"
            torch.save(
                {
                    "draft_state_dict": {"a": torch.ones(2), "b": torch.zeros(2)},
                    "step": 7,
                },
                path,
            )
            store = TensorStore(tmp)
            with patch("torch.load", wraps=torch.load) as load:
                torch.testing.assert_close(store.read("a"), torch.ones(2))
                torch.testing.assert_close(store.read("b", slice(0, 1)), torch.zeros(1))
                store.read("a")
                self.assertEqual(load.call_count, 1)

    def test_readers_share_single_file_precedence_and_fallback(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / "model.safetensors"
            save_file({"weight": torch.ones(2)}, file)
            index = root / "model.safetensors.index.json"
            index.write_text(
                json.dumps({"weight_map": {"missing": "absent.safetensors"}})
            )
            self.assertEqual(
                list(TensorStore(root).metadata), list_checkpoint_keys(tmp)
            )
            index.unlink()
            file.rename(root / "custom.safetensors")
            self.assertEqual(
                list(TensorStore(root).metadata), list_checkpoint_keys(tmp)
            )
