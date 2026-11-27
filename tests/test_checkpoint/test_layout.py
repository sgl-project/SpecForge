"""Checkpoint layout resolution and weight-map containment tests."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from specforge.checkpoint import TensorStore
from specforge.checkpoint.layout import resolve_weight_layout


class CheckpointLayoutTest(unittest.TestCase):
    def _symlink(self, target: str, link: Path) -> None:
        try:
            os.symlink(target, link, target_is_directory=os.path.isdir(target))
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"symlinks are unavailable here: {error}")

    def test_canonical_precedence_and_non_canonical_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                resolve_weight_layout(tmp)

            save_file({"weight": torch.ones(2)}, str(root / "custom.safetensors"))
            self.assertEqual(
                resolve_weight_layout(tmp),
                ("single", str(root / "custom.safetensors")),
            )

            save_file({"weight": torch.ones(2)}, str(root / "extra.safetensors"))
            with self.assertRaisesRegex(FileNotFoundError, "Multiple unindexed"):
                resolve_weight_layout(tmp)

            torch.save({"weight": torch.zeros(2)}, root / "pytorch_model.bin")
            self.assertEqual(
                resolve_weight_layout(tmp),
                ("single", str(root / "pytorch_model.bin")),
            )

            save_file({"weight": torch.ones(2)}, str(root / "model.safetensors"))
            self.assertEqual(
                resolve_weight_layout(tmp),
                ("single", str(root / "model.safetensors")),
            )

    def test_weight_map_shard_names_stay_inside_the_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "package"
            outside = root / "outside"
            package.mkdir()
            outside.mkdir()
            marker = torch.full((2, 2), 7.3125)
            save_file(
                {"model.embed_tokens.weight": marker},
                str(outside / "licensed.safetensors"),
            )
            save_file(
                {"model.embed_tokens.weight": torch.ones(2, 2)},
                str(package / "model-00001-of-00001.safetensors"),
            )

            escapes = (
                "../outside/licensed.safetensors",
                "nested/../../outside/licensed.safetensors",
                str(outside / "licensed.safetensors"),
            )
            index_path = package / "model.safetensors.index.json"
            for shard in escapes:
                with self.subTest(shard=shard):
                    index_path.write_text(
                        json.dumps({"weight_map": {"model.embed_tokens.weight": shard}})
                    )
                    with self.assertRaisesRegex(
                        ValueError, "weight_map values must be shard file names"
                    ):
                        TensorStore(package)

            index_path.write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.embed_tokens.weight": (
                                "model-00001-of-00001.safetensors"
                            )
                        }
                    }
                )
            )
            torch.testing.assert_close(
                TensorStore(package).read("model.embed_tokens.weight"),
                torch.ones(2, 2),
            )

    def test_weight_map_rejects_symlink_that_leaves_a_plain_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "package"
            outside = root / "outside"
            package.mkdir()
            outside.mkdir()
            save_file(
                {"model.embed_tokens.weight": torch.full((2, 2), 7.3125)},
                str(outside / "licensed.safetensors"),
            )
            shard = "model-00001-of-00001.safetensors"
            self._symlink(
                os.path.join("..", "outside", "licensed.safetensors"),
                package / shard,
            )
            (package / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"model.embed_tokens.weight": shard}})
            )
            with self.assertRaisesRegex(
                ValueError, "weight_map values must be shard file names"
            ):
                TensorStore(package)

    def test_shard_parent_is_resolved_after_directory_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            version = root / "versions" / "v1"
            version.mkdir(parents=True)
            self._symlink(str(version), root / "current")
            # Only the shard beside v1 is used. The lexical shortcut is a dangling link.
            self._symlink(str(Path(tmp) / "absent"), root / "part.safetensors")
            weights = {"embed.weight": torch.ones(2, 2)}
            save_file(weights, str(version.parent / "part.safetensors"))
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {"weight_map": {"embed.weight": "current/../part.safetensors"}}
                )
            )
            torch.testing.assert_close(
                TensorStore(root).read("embed.weight"),
                weights["embed.weight"],
            )

    def test_weight_map_accepts_huggingface_snapshot_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "hub" / "models--org--name"
            snapshot = repo / "snapshots" / "9f1c0b7"
            blobs = repo / "blobs"
            snapshot.mkdir(parents=True)
            blobs.mkdir()
            outside = root / "outside"
            outside.mkdir()

            shard = "model-00001-of-00001.safetensors"
            payload = {"model.embed_tokens.weight": torch.full((2, 2), 1.5078125)}
            save_file(payload, str(root / "tmp.safetensors"))
            shard_blob = blobs / ("a" * 40)
            (root / "tmp.safetensors").replace(shard_blob)
            outside_shard = outside / "licensed.safetensors"
            save_file(
                {"model.embed_tokens.weight": torch.full((2, 2), 7.3125)},
                str(outside_shard),
            )
            index_blob = blobs / ("b" * 40)
            index_blob.write_text(
                json.dumps({"weight_map": {"model.embed_tokens.weight": shard}})
            )
            self._symlink(
                os.path.join("..", "..", "blobs", index_blob.name),
                snapshot / "model.safetensors.index.json",
            )
            self._symlink(
                os.path.join("..", "..", "blobs", shard_blob.name), snapshot / shard
            )

            torch.testing.assert_close(
                TensorStore(snapshot).read("model.embed_tokens.weight"),
                payload["model.embed_tokens.weight"],
            )

            # A symlink that leaves ``models--org--name`` is still refused.
            escaping = "model-00001-of-00001-link.safetensors"
            self._symlink(
                os.path.relpath(outside_shard, snapshot),
                snapshot / escaping,
            )
            index_blob.write_text(
                json.dumps({"weight_map": {"model.embed_tokens.weight": escaping}})
            )
            with self.assertRaisesRegex(
                ValueError, "weight_map values must be shard file names"
            ):
                TensorStore(snapshot)

    def test_glob_metacharacters_in_checkpoint_dir_stay_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bracket_dir, sibling_dir = root / "run[1]", root / "run1"
            bracket_dir.mkdir()
            sibling_dir.mkdir()
            save_file({"w": torch.ones(2, 2)}, str(sibling_dir / "other.safetensors"))
            (sibling_dir / "custom.index.json").write_text(
                json.dumps({"weight_map": {"w": "other.safetensors"}})
            )
            with self.assertRaises(FileNotFoundError):
                resolve_weight_layout(str(bracket_dir))

            save_file(
                {"w": torch.full((2, 2), 5.0)}, str(bracket_dir / "own.safetensors")
            )
            self.assertEqual(
                resolve_weight_layout(str(bracket_dir)),
                ("single", str(bracket_dir / "own.safetensors")),
            )
            (bracket_dir / "own.index.json").write_text(
                json.dumps({"weight_map": {"w": "own.safetensors"}})
            )
            self.assertEqual(
                resolve_weight_layout(str(bracket_dir)),
                ("index", str(bracket_dir / "own.index.json")),
            )

    def test_dangling_index_symlink_fails_instead_of_loading_partial_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_file(
                {"w": torch.ones(2, 2)},
                str(root / "model-00001-of-00002.safetensors"),
            )
            self._symlink(
                "missing-index-target.json", root / "model.safetensors.index.json"
            )
            self.assertEqual(
                resolve_weight_layout(str(root)),
                ("index", str(root / "model.safetensors.index.json")),
            )
            with self.assertRaises(FileNotFoundError):
                TensorStore(root)


if __name__ == "__main__":
    unittest.main()
