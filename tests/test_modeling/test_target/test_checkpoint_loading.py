import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from specforge.modeling.target.checkpoint import load_checkpoint_tensors


class CheckpointLoadingTest(unittest.TestCase):
    def test_formats_and_selection(self):
        tensors = {
            "embed.weight": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
            "head.weight": torch.ones(2, 3, dtype=torch.bfloat16),
        }
        for extension in ("safetensors", "bin"):
            for sharded in (False, True):
                with self.subTest(extension=extension, sharded=sharded):
                    with tempfile.TemporaryDirectory() as tmp:
                        root = Path(tmp)
                        stem = (
                            "model" if extension == "safetensors" else "pytorch_model"
                        )
                        save = save_file if extension == "safetensors" else torch.save
                        if sharded:
                            weight_map = {}
                            for i, (key, tensor) in enumerate(tensors.items()):
                                filename = f"{stem}-{i:05d}.{extension}"
                                save({key: tensor}, str(root / filename))
                                weight_map[key] = filename
                            (root / f"{stem}.{extension}.index.json").write_text(
                                json.dumps({"weight_map": weight_map})
                            )
                        else:
                            save(tensors, str(root / f"{stem}.{extension}"))

                        selections = (
                            ({}, tensors),
                            (
                                {"keys": ["embed.weight"]},
                                {"embed.weight": tensors["embed.weight"]},
                            ),
                            (
                                {"key_filter": lambda key: key.startswith("head.")},
                                {"head.weight": tensors["head.weight"]},
                            ),
                        )
                        for options, expected in selections:
                            with self.subTest(selection=next(iter(options), "all")):
                                loaded = load_checkpoint_tensors(tmp, **options)
                                self.assertEqual(set(loaded), set(expected))
                                for key, tensor in expected.items():
                                    torch.testing.assert_close(loaded[key], tensor)
                                    self.assertEqual(loaded[key].device.type, "cpu")

    def test_exact_keys_are_required_but_filters_allow_no_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_file({"present": torch.ones(1)}, str(Path(tmp, "model.safetensors")))
            with self.assertRaisesRegex(KeyError, "missing"):
                load_checkpoint_tensors(tmp, keys=["present", "missing"])
            self.assertEqual(
                {},
                load_checkpoint_tensors(tmp, key_filter=lambda key: key == "missing"),
            )

    def test_only_selected_shards_must_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tensor = torch.ones(1)
            save_file({"present": tensor}, str(root / "part.safetensors"))
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "present": "part.safetensors",
                            "missing": "absent.safetensors",
                        }
                    }
                )
            )
            loaded = load_checkpoint_tensors(tmp, keys=["present"])
            torch.testing.assert_close(loaded["present"], tensor)
            with self.assertRaisesRegex(FileNotFoundError, "absent.safetensors"):
                load_checkpoint_tensors(tmp, keys=["missing"])

    def test_canonical_safetensors_precedes_index_and_bin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = torch.ones(1)
            save_file({"weight": expected}, str(root / "model.safetensors"))
            save_file({"weight": torch.zeros(1)}, str(root / "part.safetensors"))
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "part.safetensors"}})
            )
            torch.save({"weight": torch.zeros(1)}, root / "pytorch_model.bin")
            loaded = load_checkpoint_tensors(tmp, keys=["weight"])
            torch.testing.assert_close(loaded["weight"], expected)

    def test_hub_source_uses_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = torch.ones(1)
            save_file({"weight": expected}, str(Path(tmp, "model.safetensors")))
            with patch(
                "huggingface_hub.snapshot_download", return_value=tmp
            ) as download:
                loaded = load_checkpoint_tensors(
                    "test-org/checkpoint", keys=["weight"], cache_dir="/tmp/test-cache"
                )
            self.assertEqual(
                download.call_args.kwargs["repo_id"], "test-org/checkpoint"
            )
            self.assertEqual(download.call_args.kwargs["cache_dir"], "/tmp/test-cache")
            torch.testing.assert_close(loaded["weight"], expected)

    def test_empty_keys_does_not_resolve_or_download(self):
        with patch(
            "specforge.modeling.target.checkpoint.resolve_checkpoint_dir"
        ) as resolve:
            self.assertEqual({}, load_checkpoint_tensors("unused", keys=[]))
            resolve.assert_not_called()

    def test_invalid_selectors_are_rejected(self):
        with self.assertRaises(ValueError):
            load_checkpoint_tensors(
                "unused", keys=["weight"], key_filter=lambda key: True
            )
        with self.assertRaises(TypeError):
            load_checkpoint_tensors("unused", keys="weight")


if __name__ == "__main__":
    unittest.main()
