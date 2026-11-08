"""Every bundled draft recipe has a canonical migration, irrespective of its name."""

import json
import unittest

from specforge.config.draft.codec import (
    decode_config,
    decode_legacy_config,
    encode_config,
)
from tests.test_config.test_draft_legacy_dflash import CONFIGS


class LegacyCatalogTest(unittest.TestCase):
    def test_every_bundled_configuration_round_trips(self):
        paths = sorted(CONFIGS.glob("*.json"))
        self.assertGreaterEqual(len(paths), 51)
        for path in paths:
            with self.subTest(path=path.name):
                payload = json.loads(path.read_text())
                # A structural oracle only; no real target checkpoint is loaded.
                target = {
                    "hidden_size": payload["hidden_size"],
                    "vocab_size": payload["vocab_size"],
                    "num_hidden_layers": payload.get("num_target_layers", 128),
                }
                node = decode_legacy_config(payload, target_config=target)
                restored = decode_config(
                    json.loads(json.dumps(encode_config(node))), target_config=target
                )
                self.assertEqual(restored, node)


if __name__ == "__main__":
    unittest.main()
