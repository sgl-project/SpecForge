"""A canonical file declares the packages its out-of-tree kinds come from."""

import dataclasses
import re
import unittest
from typing import ClassVar
from unittest.mock import patch

from specforge.config.draft import registry
from specforge.config.draft.attention import GQASpec
from specforge.config.draft.codec import decode_config, encode_config
from tests.test_config.draft_fixtures import draft_config


@dataclasses.dataclass(frozen=True, kw_only=True)
class ExtensionAttention(GQASpec):
    type: ClassVar[str] = "test_extension_attention"


def extension_config():
    base = draft_config()
    node = ExtensionAttention(**dataclasses.asdict(base.attention_configs["full"]))
    return dataclasses.replace(base, attention_configs={"full": node})


class ExtensionsTest(unittest.TestCase):
    def test_built_in_kinds_declare_no_extensions(self):
        self.assertEqual(encode_config(draft_config())["extensions"], [])

    def test_out_of_tree_kinds_are_declared_and_checked(self):
        with patch.dict(registry._SCHEMAS):
            registry.register_component_schema("attention", ExtensionAttention)
            config = extension_config()
            payload = encode_config(config)
            package = ExtensionAttention.__module__.split(".")[0]
            self.assertEqual(payload["extensions"], [package])
            self.assertEqual(decode_config(payload), config)
            for declared in ([], ["other"], ["other", "tests"]):
                with self.assertRaisesRegex(ValueError, "extensions: declared"):
                    decode_config({**payload, "extensions": declared})
        with self.assertRaisesRegex(
            ValueError, re.escape(f"requires extensions ['{package}']")
        ):
            decode_config(payload)

    def test_declarations_are_a_sorted_list_of_names(self):
        payload = encode_config(draft_config())
        for value in (None, "tests", [""], [1], ["b", "a"], ["a", "a"]):
            with self.assertRaisesRegex(ValueError, "extensions"):
                decode_config({**payload, "extensions": value})


if __name__ == "__main__":
    unittest.main()
