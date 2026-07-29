import unittest

from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.llama3_eagle import LlamaForCausalLMEagle3


class TestAutoModelForCausalLM(unittest.TestCase):

    def test_automodel(self):
        """init"""
        model = AutoDraftModel.from_pretrained(
            "jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B"
        )
        self.assertIsInstance(model, LlamaForCausalLMEagle3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
