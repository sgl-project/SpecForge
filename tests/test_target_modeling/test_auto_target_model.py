import os
import unittest

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from transformers import LlamaForCausalLM as HFLlamaForCausalLM

from specforge.distributed import init_distributed
from specforge.modeling.target.custom_backend.llama import (
    LlamaForCausalLM as SFLlamaForCausalLM,
)
from tests.utils import get_available_port


class TestAutoTargetModel(unittest.TestCase):

    def test_auto_target_model(self):
        port = get_available_port()
        os.environ["RANK"] = str(0)
        os.environ["WORLD_SIZE"] = str(1)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)

        init_distributed(tp_size=1)

        set_seed(42)
        hf_model = HFLlamaForCausalLM.from_pretrained(
            "unsloth/Llama-3.2-1B", torch_dtype=torch.bfloat16
        ).cuda()
        sf_model = SFLlamaForCausalLM.from_pretrained(
            "unsloth/Llama-3.2-1B", torch_dtype=torch.bfloat16
        ).cuda()

        # create data
        input_ids = torch.randint(0, 1000, (1, 256)).cuda()
        attention_mask = torch.ones_like(input_ids).cuda()

        expected_logits = hf_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        dist_logits = sf_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits

        assert torch.allclose(
            expected_logits,
            dist_logits,
            rtol=1e-5,
            atol=1e-5,
        ), f"Logits are not close, {expected_logits} vs {dist_logits}"

        dist.destroy_process_group()


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestAutoTargetModel))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
