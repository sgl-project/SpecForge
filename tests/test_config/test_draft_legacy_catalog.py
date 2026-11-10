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
        self.assertGreaterEqual(len(paths), 52)
        for path in paths:
            with self.subTest(path=path.name):
                payload = json.loads(path.read_text())
                options = (
                    {"mask_token_id": 1}
                    if payload["architectures"] == ["PEagleDraftModel"]
                    else {}
                )
                # A structural oracle only; no real target checkpoint is loaded.
                target = {
                    "hidden_size": payload["hidden_size"],
                    "vocab_size": payload["vocab_size"],
                    "num_hidden_layers": payload.get("num_target_layers", 128),
                }
                node = decode_legacy_config(payload, **options, target_config=target)
                restored = decode_config(
                    json.loads(json.dumps(encode_config(node))),
                    target_config=target,
                )
                self.assertEqual(restored, node)
                inherited = decode_legacy_config(
                    payload | {"draft_vocab_size": None},
                    **options,
                    target_config=target,
                )
                self.assertEqual(inherited.draft_vocab_size, payload["vocab_size"])
                for name in (
                    "hidden_size",
                    "num_hidden_layers",
                    "num_attention_heads",
                    "intermediate_size",
                    "vocab_size",
                ):
                    with (
                        self.subTest(field=name),
                        self.assertRaisesRegex(ValueError, name),
                    ):
                        decode_legacy_config(
                            {
                                key: value
                                for key, value in payload.items()
                                if key != name
                            },
                            **options,
                            target_config=target,
                        )


if __name__ == "__main__":
    unittest.main()
