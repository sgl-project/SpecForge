import importlib
import sys
import unittest
import warnings


def _fresh_import(name):
    for key in [
        key for key in sys.modules if key == name or key.startswith(name + ".")
    ]:
        del sys.modules[key]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module(name)
    return module, caught


class MovedImportPathTest(unittest.TestCase):
    def assert_alias(self, old, new):
        module, caught = _fresh_import(old)
        self.assertIs(module, importlib.import_module(new))
        self.assertTrue(
            any(issubclass(w.category, DeprecationWarning) for w in caught), old
        )

    def test_module_aliases_resolve_to_the_moved_objects(self):
        self.assert_alias(
            "specforge.modeling.draft.llama3_eagle", "specforge.modeling.draft.eagle3"
        )
        self.assert_alias(
            "specforge.modeling.draft.flex_attention_backend",
            "specforge.modeling.draft.layers.attention.backends.flex_attention_backend",
        )
        for name in ("embedding", "linear", "lm_head"):
            self.assert_alias(
                f"specforge.layers.{name}", f"specforge.modeling.draft.layers.{name}"
            )

    def test_package_reexports_keep_class_identity(self):
        from specforge.modeling.draft import layers
        from specforge.modeling.draft.eagle3_mask import generate_eagle3_mask
        from specforge.modeling.draft.layers.attention.backends import flex_attention

        legacy_layers, caught = _fresh_import("specforge.layers")
        self.assertTrue(any(w.category is DeprecationWarning for w in caught))
        for name in legacy_layers.__all__:
            self.assertIs(getattr(legacy_layers, name), getattr(layers, name))

        legacy_flex, caught = _fresh_import("specforge.modeling.draft.flex_attention")
        self.assertTrue(any(w.category is DeprecationWarning for w in caught))
        self.assertIs(legacy_flex.generate_eagle3_mask, generate_eagle3_mask)
        self.assertIs(
            legacy_flex.compile_friendly_flex_attention,
            flex_attention.compile_friendly_flex_attention,
        )


if __name__ == "__main__":
    unittest.main()
