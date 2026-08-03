import unittest

from specforge.algorithms.model_providers import _transformers_attention_implementation


class TransformersAttentionImplementationTest(unittest.TestCase):
    def test_usp_uses_sdpa_for_transformers_model_construction(self):
        self.assertEqual("sdpa", _transformers_attention_implementation("usp"))

    def test_concrete_backend_is_preserved(self):
        self.assertEqual(
            "flex_attention",
            _transformers_attention_implementation("flex_attention"),
        )


if __name__ == "__main__":
    unittest.main()
