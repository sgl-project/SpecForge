import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from specforge.algorithms.common.target_kv_data import (
    CAPTURE_FORMAT,
    build_collator,
    normalize_sample,
    validate_manifest,
)
from specforge.modeling.draft.target_kv import capture_geometry
from tests.test_modeling.test_dspark_target_kv import config, kv_registration


def sample(length):
    return {
        "input_ids": torch.arange(length),
        "loss_mask": torch.ones(length),
        "target_kv": torch.randn(length, 2, 2, 2, 4),
        "target_last_hidden_states": torch.randn(length, 16),
    }


class OfflineKVTest(unittest.TestCase):
    def test_manifest_allows_ablation_reuse_but_rejects_capture_mismatches(self):
        target = SimpleNamespace(
            model_type="qwen3_5_text",
            num_key_value_heads=2,
            head_dim=4,
            hidden_size=16,
            vocab_size=32,
            _commit_hash="public-revision",
            layer_types=["full_attention", "linear_attention", "full_attention"],
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 1e7,
                "partial_rotary_factor": 0.5,
            },
        )
        metadata = {
            "format": CAPTURE_FORMAT,
            "samples": 2,
            "state": "post_qk_norm_post_rope",
            "target_model": "example/target",
            "target_revision": "public-revision",
            "padding": False,
            "position_origin": 0,
            "target_kv_geometry": capture_geometry(config(), target),
        }
        with tempfile.TemporaryDirectory() as path:
            manifest = Path(path) / "capture-manifest.json"
            manifest.write_text(json.dumps(metadata))
            for mode, norm in (
                ("cached", False),
                ("derope_reproject", False),
                ("derope_reproject", True),
            ):
                validate_manifest(path, config(mode, norm), target, "example/target")
            for alias in ("target_kv_v1", "dspark_target_kv_v1"):
                legacy = dict(metadata, format=alias)
                manifest.write_text(json.dumps(legacy))
                validate_manifest(path, config(), target, "example/target")
            for key, value in (
                ("format", "unknown_capture"),
                ("format", None),
                ("state", "pre_rope"),
                ("position_origin", 7),
                ("padding", True),
                ("target_revision", "different"),
                ("target_revision", None),
            ):
                altered = copy.deepcopy(metadata)
                altered[key] = value
                manifest.write_text(json.dumps(altered))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_manifest(path, config(), target, "example/target")
            missing_revision = dict(metadata)
            del missing_revision["target_revision"]
            manifest.write_text(json.dumps(missing_revision))
            with self.assertRaisesRegex(ValueError, "target revision"):
                validate_manifest(path, config(), target, "example/target")
            for key, value in (
                ("layer_ids", [2, 0]),
                ("rope_theta", 1e4),
                ("rotary_dim", 4),
            ):
                altered = copy.deepcopy(metadata)
                altered["target_kv_geometry"][key] = value
                manifest.write_text(json.dumps(altered))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_manifest(path, config(), target, "example/target")

    def test_offline_contract_and_sequence_padding(self):
        registration = kv_registration()
        self.assertFalse(registration.spec.supports_online)
        collated = build_collator()([normalize_sample(sample(n), 8) for n in (4, 7)])
        self.assertEqual(collated["target_kv"].shape, (2, 7, 2, 2, 2, 4))
        self.assertEqual(collated["loss_mask"][0, 4:].count_nonzero(), 0)
        self.assertEqual(collated["target_kv"][0, 4:].count_nonzero(), 0)

    def test_misaligned_or_wrong_layout_features_rejected_before_truncation(self):
        for name, tensor in (
            ("target_kv", torch.zeros(8, 2, 2, 2, 4)),
            ("target_kv", torch.zeros(7, 2, 3, 2, 4)),
            ("input_ids", torch.zeros(7)),
            ("target_last_hidden_states", torch.zeros(8, 16)),
            ("target_last_hidden_states", torch.zeros(6, 16)),
        ):
            raw = sample(7)
            raw[name] = tensor
            with self.subTest(name=name), self.assertRaises(ValueError):
                normalize_sample(raw, 4)

    def test_reader_provides_the_required_tensor_names(self):
        with tempfile.TemporaryDirectory() as path:
            torch.save(sample(7), Path(path) / "example.ckpt")
            refs = (
                kv_registration()
                .providers.offline_for("text")
                .build_reader(path, run_id="test", ttt_length=1, max_len=7)
                .read()
            )
            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0].strategy, "dspark")
            self.assertEqual(set(refs[0].feature_keys), set(sample(7)))


if __name__ == "__main__":
    unittest.main()
