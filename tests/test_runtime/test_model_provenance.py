from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import torch.distributed as dist

from specforge.training import provenance


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

        self.assertEqual(identity[0], "directory-stat-v2")
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

    def test_shard_identity_survives_inode_and_ctime_churn(self):
        # A remount (NFS/FUSE/overlay) or pod reschedule presents byte-identical
        # shards under new inode/device numbers and a new ctime. The identity
        # must be a pure function of (name, size, mtime) so those resumes work.
        with TemporaryDirectory(prefix="model-provenance-remount-") as directory:
            shard = Path(directory) / "model.safetensors"
            shard.write_bytes(b"same-weights")
            original_stat = shard.stat()
            original = provenance.model_source_identity(directory)

            replacement = Path(directory) / "replacement.safetensors"
            replacement.write_bytes(b"same-weights")
            os.replace(replacement, shard)
            os.utime(
                shard,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            remounted = provenance.model_source_identity(directory)

        self.assertEqual(original, remounted)

    def test_shard_identity_changes_with_mtime_or_size(self):
        with TemporaryDirectory(prefix="model-provenance-replace-") as directory:
            shard = Path(directory) / "model.safetensors"
            shard.write_bytes(b"old-weights")
            original_stat = shard.stat()
            original = provenance.model_source_identity(directory)

            # Same size, later mtime. Set it explicitly: a back-to-back
            # rewrite can land inside the filesystem's mtime-update tick and
            # legitimately keep the same timestamp.
            shard.write_bytes(b"new-weights")
            os.utime(
                shard,
                ns=(
                    original_stat.st_atime_ns,
                    original_stat.st_mtime_ns + 1_000_000_000,
                ),
            )
            self.assertNotEqual(original, provenance.model_source_identity(directory))

            shard.write_bytes(b"different-size")
            os.utime(
                shard,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            self.assertNotEqual(original, provenance.model_source_identity(directory))

    def test_resume_requires_current_provenance_format(self):
        current = {
            provenance.MODEL_SOURCE_IDENTITY_FORMAT_FIELD: provenance.MODEL_SOURCE_IDENTITY_FORMAT
        }
        self.assertEqual(
            provenance.model_provenance_for_resume_comparison(current, current),
            (current, current),
        )
        for saved in ({}, {provenance.MODEL_SOURCE_IDENTITY_FORMAT_FIELD: "stat-v1"}):
            with (
                self.subTest(saved=saved),
                self.assertRaisesRegex(ValueError, "unsupported identity format"),
            ):
                provenance.model_provenance_for_resume_comparison(current, saved)

    def test_unknown_saved_identity_format_fails_loudly(self):
        current_mapping = {
            provenance.MODEL_SOURCE_IDENTITY_FORMAT_FIELD: (
                provenance.MODEL_SOURCE_IDENTITY_FORMAT
            ),
            "target_model": ("reference", "remote/target"),
        }
        saved_mapping = dict(current_mapping)
        saved_mapping[provenance.MODEL_SOURCE_IDENTITY_FORMAT_FIELD] = "future-v9"

        with self.assertRaisesRegex(ValueError, "unsupported identity format"):
            provenance.model_provenance_for_resume_comparison(
                _contract(current_mapping),
                _contract(saved_mapping),
            )


if __name__ == "__main__":
    unittest.main()
