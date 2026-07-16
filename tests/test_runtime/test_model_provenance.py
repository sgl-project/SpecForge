from __future__ import annotations

import hashlib
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import torch.distributed as dist

from specforge.training import provenance


def _config(target_model_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            target_model_path=target_model_path,
            draft_model_config="remote/draft",
            vocab_mapping_path="",
            embedding_key="model.embed_tokens.weight",
            lm_head_key="lm_head.weight",
            load_target_embedding=True,
            input_modality="text",
            torch_dtype="bfloat16",
        )
    )


def _contract(mapping: dict) -> tuple:
    return tuple((key, mapping[key]) for key in sorted(mapping))


class ModelProvenanceTest(unittest.TestCase):
    def test_fresh_identity_does_not_hash_weight_shards(self):
        with TemporaryDirectory(prefix="model-provenance-") as directory:
            root = Path(directory)
            descriptor = root / "model.safetensors.index.json"
            descriptor.write_text(
                json.dumps({"weight_map": {"x": "model-00001.safetensors"}}),
                encoding="utf-8",
            )
            shard = root / "model-00001.safetensors"
            with shard.open("wb") as stream:
                stream.seek(64 * 1024 * 1024)
                stream.write(b"x")

            original_hash = provenance._sha256_file
            with mock.patch.object(
                provenance,
                "_sha256_file",
                wraps=original_hash,
            ) as sha256_file:
                identity = provenance.model_source_identity(directory)

        self.assertEqual(identity[0], "directory-stat-v1")
        self.assertEqual(
            [Path(call.args[0]).name for call in sha256_file.call_args_list],
            [descriptor.name],
        )

    def test_nonzero_rank_uses_broadcast_without_touching_sources(self):
        expected = {"rank_agreed": True}

        def receive_rank_zero(payload, *, src):
            self.assertEqual(src, 0)
            payload[0] = ("ok", expected)

        with (
            mock.patch.object(dist, "is_available", return_value=True),
            mock.patch.object(dist, "is_initialized", return_value=True),
            mock.patch.object(dist, "get_world_size", return_value=8),
            mock.patch.object(dist, "get_rank", return_value=3),
            mock.patch.object(
                dist,
                "broadcast_object_list",
                side_effect=receive_rank_zero,
            ) as broadcast,
            mock.patch.object(
                provenance,
                "_compute_model_resume_provenance",
            ) as compute,
        ):
            result = provenance.model_resume_provenance(
                object(),
                object(),
                object(),
                capture_layers=None,
            )

        self.assertEqual(result, expected)
        compute.assert_not_called()
        broadcast.assert_called_once()

    def test_same_size_preserved_mtime_replacement_changes_shard_identity(self):
        with TemporaryDirectory(prefix="model-provenance-replace-") as directory:
            shard = Path(directory) / "model.safetensors"
            shard.write_bytes(b"old-weights")
            original_stat = shard.stat()
            original = provenance.model_source_identity(directory)

            replacement = Path(directory) / "replacement.safetensors"
            replacement.write_bytes(b"new-weights")
            os.utime(
                replacement,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            os.replace(replacement, shard)
            os.utime(
                shard,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            replaced = provenance.model_source_identity(directory)

        self.assertNotEqual(original, replaced)

    def test_legacy_resume_hashes_once_then_reuses_stat_keyed_cache(self):
        with TemporaryDirectory(prefix="legacy-provenance-") as directory:
            target = Path(directory) / "target"
            target.mkdir()
            (target / "config.json").write_text("{}", encoding="utf-8")
            (target / "model.safetensors").write_bytes(b"legacy-weights")

            current_mapping = provenance._compute_model_resume_provenance(
                _config(str(target)),
                SimpleNamespace(_commit_hash=None),
                SimpleNamespace(_commit_hash=None),
                capture_layers=[1, 3],
            )
            current = _contract(current_mapping)
            saved_mapping = dict(current_mapping)
            saved_mapping.pop(provenance.MODEL_SOURCE_IDENTITY_FORMAT_FIELD)
            saved_mapping["target_model"] = (
                "directory",
                str(target),
                (
                    (
                        "config.json",
                        2,
                        hashlib.sha256(b"{}").hexdigest(),
                    ),
                    (
                        "model.safetensors",
                        len(b"legacy-weights"),
                        hashlib.sha256(b"legacy-weights").hexdigest(),
                    ),
                ),
            )
            saved = _contract(saved_mapping)
            cache_root = str(Path(directory) / "output")

            original_hash = provenance._sha256_file
            with mock.patch.object(
                provenance,
                "_sha256_file",
                wraps=original_hash,
            ) as sha256_file:
                first = provenance.model_provenance_for_resume_comparison(
                    current,
                    saved,
                    cache_root=cache_root,
                )
                first_hash_count = sha256_file.call_count
                second = provenance.model_provenance_for_resume_comparison(
                    current,
                    saved,
                    cache_root=cache_root,
                )

            self.assertEqual(first, saved)
            self.assertEqual(second, saved)
            self.assertEqual(first_hash_count, 2)
            self.assertEqual(sha256_file.call_count, first_hash_count)
            cache_files = list(
                (Path(cache_root) / ".specforge-provenance-cache").glob("*.json")
            )
            self.assertEqual(len(cache_files), 1)

    def test_current_resume_never_builds_legacy_content_identity(self):
        current = _contract(
            {
                provenance.MODEL_SOURCE_IDENTITY_FORMAT_FIELD: (
                    provenance.MODEL_SOURCE_IDENTITY_FORMAT
                ),
                "target_model": ("reference", "remote/target"),
            }
        )

        with mock.patch.object(
            provenance,
            "legacy_model_source_identity",
        ) as legacy_identity:
            comparison = provenance.model_provenance_for_resume_comparison(
                current,
                current,
                cache_root="unused",
            )

        self.assertIs(comparison, current)
        legacy_identity.assert_not_called()

    def test_unknown_saved_identity_format_fails_loudly(self):
        current_mapping = {
            provenance.MODEL_SOURCE_IDENTITY_FORMAT_FIELD: (
                provenance.MODEL_SOURCE_IDENTITY_FORMAT
            ),
            "target_model": ("reference", "remote/target"),
        }
        saved_mapping = dict(current_mapping)
        saved_mapping[provenance.MODEL_SOURCE_IDENTITY_FORMAT_FIELD] = "future-v2"

        with self.assertRaisesRegex(ValueError, "unsupported identity format"):
            provenance.model_provenance_for_resume_comparison(
                _contract(current_mapping),
                _contract(saved_mapping),
                cache_root=os.devnull,
            )


if __name__ == "__main__":
    unittest.main()
