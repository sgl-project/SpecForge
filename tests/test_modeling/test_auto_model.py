import unittest

from specforge.modeling.auto import AutoEagle3DraftModel, LlamaForCausalLMEagle3


class TestAutoModelForCausalLM(unittest.TestCase):

    def test_automodel(self):
        """init"""
        model = AutoEagle3DraftModel.from_pretrained(
            "jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B"
        )
        self.assertIsInstance(model, LlamaForCausalLMEagle3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
