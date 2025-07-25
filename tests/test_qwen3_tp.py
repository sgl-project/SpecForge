import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate.utils import set_seed
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

from specforge.distributed import init_distributed


def test_qwen3_moe_tp(rank, world_size, temp_dir):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    init_distributed(tp_size=4)
    set_seed(42)
    config = Qwen3MoeConfig(
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=6144,
        moe_intermediate_size=1536,
        num_hidden_layers=2,
        max_position_embeddings=2048,
        num_attention_heads=64,
        num_key_value_heads=4,
        num_experts=128,
        num_experts_per_tok=8,
        hidden_act="silu",
        rms_norm_eps=1e-6,
    )

    # create a simple single-gpu model
    model = Qwen3MoeForCausalLM(config).cuda()

    from specforge.modeling.target.qwen3_moe import (
        Qwen3MoeForCausalLM as DistQwen3MoeForCausalLM,
    )

    dist_model = DistQwen3MoeForCausalLM(config).cuda()

    # save the model weights to a temp directory
    if dist.get_rank() == 0:
        model.save_pretrained(temp_dir)
        print(f"Saved model to {temp_dir}")
    dist.barrier()

    # load the model weights to the distributed model
    print(f"Loading model from {temp_dir}")
    dist_model.load_checkpoint(temp_dir)
    dist.barrier()

    # create data
    input_ids = torch.randint(0, 1000, (1, 256)).cuda()
    attention_mask = torch.ones_like(input_ids).cuda()

    expected_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    dist_logits = dist_model(input_ids=input_ids, attention_mask=attention_mask).logits

    print(expected_logits, dist_logits)
    assert torch.allclose(
        expected_logits,
        dist_logits,
        rtol=1e-4,
        atol=1e-4,
    ), f"Logits are not close, {expected_logits} vs {dist_logits}"


class TestQwen3MoeTP(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_qwen3_moe_tp(self):
        mp.spawn(test_qwen3_moe_tp, nprocs=4, args=(4, self.temp_dir.name))


if __name__ == "__main__":
    suite = unittest.TestSuite()

    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestQwen3MoeTP))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
