import unittest

from specforge.modeling.auto import AutoDraftModel


class TestAutoModelForCausalLM(unittest.TestCase):
    def test_automodel_requires_user_config(self):
        with self.assertRaisesRegex(ValueError, "user's YAML"):
            AutoDraftModel.from_pretrained("org/draft")
