"""Native computation must agree before any target weight is read."""

import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import yaml

from specforge.checkpoint.formats.native.writer import export_native
from specforge.checkpoint.native import validate_target_initialization
from specforge.config.draft.codec import parse_draft
from specforge.loading import initialize_draft
from specforge.training.canonical import capture_layers, resolve_run
from tests.test_checkpoint.test_native_presets import NATIVE_MAPPING, PRESETS
from tests.test_runtime.test_canonical_training import run_config

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/native/qwen3_5_4b"


def native_fixture():
    target = json.loads((FIXTURE / "config.json").read_text())
    payload = yaml.safe_load((PRESETS / "qwen3_5_4b_mtp.yaml").read_text())
    return parse_draft(payload, target_config=target["text_config"]), target


class NativeContractTest(unittest.TestCase):
    def test_native_preflight_and_capture(self):
        draft, target = native_fixture()
        validate_target_initialization(draft, target, NATIVE_MAPPING)
        cfg = run_config(FIXTURE, draft, initialization={"from_target": NATIVE_MAPPING})
        resolve_run(cfg, target_config=target["text_config"])
        layers = capture_layers(cfg, draft, target["text_config"])
        self.assertEqual(layers, [0])
        self.assertLessEqual(
            max(layers), target["text_config"]["num_hidden_layers"] - 2
        )

    def test_mismatches_fail_before_loading_in_both_paths(self):
        draft, target = native_fixture()
        variants = [
            replace(draft, rope_parameters=replace(draft.rope_parameters, **values))
            for values in (
                {"rope_theta": 1e6},
                {"partial_rotary_factor": 0.5, "mrope_section": (22, 22, 20)},
                {"mrope_interleaved": False},
                {"mrope_section": (10, 12, 10)},
            )
        ] + [replace(draft, norm_config=replace(draft.norm_config, eps=1e-5))]
        for changed in variants:
            with self.subTest(rope=changed.rope_parameters, norm=changed.norm_config):
                cfg = run_config(
                    FIXTURE, changed, initialization={"from_target": NATIVE_MAPPING}
                )
                with self.assertRaisesRegex(ValueError, "disagrees with draft"):
                    resolve_run(cfg, target_config=target["text_config"])
                with patch(
                    "specforge.loading.build_draft_model", side_effect=AssertionError
                ):
                    with self.assertRaisesRegex(ValueError, "disagrees with draft"):
                        initialize_draft(
                            changed, target=FIXTURE, from_target=NATIVE_MAPPING
                        )
                with self.assertRaisesRegex(ValueError, "disagrees with draft"):
                    export_native(changed, {}, FIXTURE, "unused")
                # An embedding copy is not a claim to reproduce the native decoder.
                validate_target_initialization(
                    changed,
                    target,
                    {"embed_tokens.weight": "model.language_model.embed_tokens.weight"},
                )
