"""Selective checkpoint loading rejects quantized copies without reading payloads."""

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
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
            with patch("torch.load", wraps=torch.load) as load:
                store = TensorStore(tmp)
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
            self.assertEqual(list_checkpoint_keys(tmp), ["weight"])
            with self.assertRaisesRegex(ValueError, "model.safetensors.*index"):
                TensorStore(root)
            index.write_text(json.dumps({"weight_map": {"weight": file.name}}))
            self.assertEqual(list(TensorStore(root).metadata), ["weight"])
            index.unlink()
            file.rename(root / "custom.safetensors")
            self.assertEqual(
                list(TensorStore(root).metadata), list_checkpoint_keys(tmp)
            )

    def test_training_state_precedes_stray_weights(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.save(
                {"draft_state_dict": {"weight": torch.ones(2)}},
                root / "training_state.pt",
            )
            save_file({"weight": torch.zeros(2)}, root / "ema.safetensors")
            torch.testing.assert_close(TensorStore(root).read("weight"), torch.ones(2))

    def test_metadata_only_refs_and_storage_less_sources(self):
        model = build_draft_model(draft_config())
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.pt"
            torch.save({"weight": torch.ones(64, 32)}, path)
            store = TensorStore(path)
            ref = SimpleNamespace(key="weight", info=store.metadata["weight"])
            bindings = {name: ref for name in model.config.draft.target_bindings}
            make_weight_plan(model, {}, bindings=bindings, allow_new=model.state_dict())
            for change, message in (
                ({"scales": ("weight_scale",)}, "scaled"),
                ({"is_meta": True}, "storage"),
            ):
                ref.info = replace(store.metadata["weight"], **change)
                with self.assertRaisesRegex(ValueError, message):
                    make_weight_plan(
                        model, {}, bindings=bindings, allow_new=model.state_dict()
                    )
            torch.save({"weight": torch.empty(64, 32, device="meta")}, path)
            refs = TensorStore(path).refs()
            with self.assertRaisesRegex(ValueError, "storage"):
                make_weight_plan(
                    model,
                    {},
                    bindings={name: refs["weight"] for name in bindings},
                    allow_new=model.state_dict(),
                )

    def test_unsigned_safetensors_and_index_consistency(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = {
                str(dtype): torch.ones(2, dtype=dtype)
                for dtype in (torch.uint16, torch.uint32, torch.uint64)
            }
            save_file(state, root / "unsigned.safetensors")
            store = TensorStore(root)
            for key, value in state.items():
                self.assertEqual(store.metadata[key].dtype, value.dtype)
                torch.testing.assert_close(store.read(key), value)
            (root / "unsigned.safetensors").unlink()
            save_file({"a": torch.ones(2)}, root / "one.safetensors")
            save_file({"b": torch.ones(2)}, root / "two.safetensors")
            index = root / "model.safetensors.index.json"
            for mapping, message in (
                ({"a": "two.safetensors", "b": "one.safetensors"}, "disagrees"),
                ({"a": "one.safetensors", "missing": "one.safetensors"}, "missing"),
            ):
                index.write_text(json.dumps({"weight_map": mapping}))
                with self.assertRaisesRegex(ValueError, message):
                    TensorStore(root)
            save_file(
                {"a": torch.ones(2), "b": torch.ones(2)}, root / "two.safetensors"
            )
            index.write_text(
                json.dumps(
                    {"weight_map": {"a": "one.safetensors", "b": "two.safetensors"}}
                )
            )
            with self.assertRaisesRegex(ValueError, "duplicate.*a"):
                TensorStore(root)
