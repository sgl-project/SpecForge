"""Recipes copied from configs/ at 83ec0c3 retain their projector-selected heads."""

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.gates.normalize_dflash_export import normalize_export
from specforge.config.draft.codec import decode_legacy_config
from tests.test_config.test_draft_legacy_dflash import target_for

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/legacy_projectors"


class LegacyDispatchTest(unittest.TestCase):
    def test_historical_projectors_select_architecture_and_preserve_heads(self):
        for name, architecture in (
            ("qwen3-8b-domino.json", "DominoDraftModel"),
            ("qwen3-4b-dspark.json", "DSparkDraftModel"),
        ):
            with self.subTest(name=name):
                payload = json.loads((FIXTURES / name).read_text())
                before = copy.deepcopy(payload)
                config = decode_legacy_config(
                    payload, target_config=target_for(payload)
                )
                self.assertEqual(config.architecture, architecture)
                spec = config.spec_config
                if architecture == "DominoDraftModel":
                    self.assertEqual(
                        (spec.gru.hidden_size, spec.gru.projection_size), (1024, 256)
                    )
                    self.assertEqual(spec.pure_draft_prefix_len, 1)
                    self.assertTrue(spec.block.shift_label)
                else:
                    self.assertEqual(
                        (spec.markov.type, spec.markov.rank), ("vanilla", 256)
                    )
                    self.assertTrue(spec.confidence.with_markov)
                self.assertEqual(payload, before)

    def test_normalized_domino_export_keeps_its_architecture(self):
        payload = json.loads((FIXTURES / "qwen3-8b-domino.json").read_text())
        expected = decode_legacy_config(payload, target_config=target_for(payload))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(payload))
            normalized = normalize_export(str(path), payload["block_size"])
        self.assertEqual(normalized["architectures"], ["DFlashDraftModel"])
        self.assertEqual(
            decode_legacy_config(normalized, target_config=target_for(payload)),
            expected,
        )

    def test_unknown_architectures_use_the_same_rejection(self):
        for architecture in ("Qwen3DSparkModel", "UnknownDraftModel"):
            with self.subTest(architecture=architecture):
                with self.assertRaisesRegex(
                    ValueError, "unsupported legacy architectures"
                ):
                    decode_legacy_config(
                        {"architectures": [architecture]}, target_config={}
                    )
