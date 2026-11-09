"""Nested configuration decoding is independent of model architecture schemas."""

import unittest
from dataclasses import dataclass

from specforge.config.draft.codec.nodes import decode_node, encode_node


@dataclass(frozen=True)
class Router:
    top_k: int


@dataclass(frozen=True)
class NestedSettings:
    router: Router
    scales: tuple[float, ...]
    enabled: bool
    label: str | None


class NodeCodecTest(unittest.TestCase):
    def test_nested_round_trip(self):
        payload = dict(router={"top_k": 2}, scales=[0.5, 1.0], enabled=True, label=None)
        node = decode_node(NestedSettings, payload, "settings")
        self.assertEqual(node, NestedSettings(Router(2), (0.5, 1.0), True, None))
        self.assertEqual(encode_node(node), payload)

    def test_nested_errors_keep_the_field_path(self):
        for router, message in (
            ({}, "settings.router: missing fields"),
            ({"top_k": 2, "extra": 1}, "settings.router: unknown fields"),
            ({"top_k": True}, "settings.router.top_k: expected"),
        ):
            with (
                self.subTest(router=router),
                self.assertRaisesRegex(ValueError, message),
            ):
                decode_node(
                    NestedSettings,
                    dict(router=router, scales=[1.0], enabled=True, label="test"),
                    "settings",
                )


if __name__ == "__main__":
    unittest.main()
