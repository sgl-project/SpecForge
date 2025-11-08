import os
import unittest

import torch

from specforge.distributed import init_distributed
from specforge.model.draft import AutoEagle3DraftModel, LlamaForCausalLMEagle3


class TestAutoModelForCausalLM(unittest.TestCase):

    def test_automodel(self):
        """init"""
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        if not torch.distributed.is_initialized():
            init_distributed(timeout=10, target_tp_size=1, draft_tp_size=1)
        model = AutoEagle3DraftModel.from_pretrained(
            "jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B"
        )
        self.assertIsInstance(model, LlamaForCausalLMEagle3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
