import unittest

from specforge.modeling.draft.llama3_eagle import LlamaForCausalLMEagle3
from specforge.modeling.target.llama4 import Llama4ForCausalLM


class TestAutoModelForCausalLM(unittest.TestCase):

    def test_automodel(self):
        """init"""
        model = Llama4ForCausalLM.from_pretrained(
            "nvidia/Llama-4-Maverick-17B-128E-Eagle3"
        )
        self.assertIsInstance(model, LlamaForCausalLMEagle3)

        model = Llama4ForCausalLM.from_pretrained(
            "jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B"
        )
        self.assertIsInstance(model, LlamaForCausalLMEagle3)


if __name__ == "__main__":
    suite = unittest.TestSuite()

    suite.addTest(unittest.makeSuite(TestAutoModelForCausalLM))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
