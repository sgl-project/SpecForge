import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate.utils import set_seed
from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM

from specforge.distributed import init_distributed


def test_phi3_forward(rank, world_size, temp_dir):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29503"

    init_distributed(tp_size=2)
    set_seed(42)
    config = Phi3Config(
        vocab_size=32064,
        hidden_size=3072,
        intermediate_size=8192,
        num_hidden_layers=2,
        max_position_embeddings=4096,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        resid_pdrop=0.0,
    )

    from specforge.modeling.target.phi3 import (
        Phi3ForCausalLM as DistPhi3ForCausalLM,
    )

    dist_model = DistPhi3ForCausalLM(config).cuda()

    # create data
    input_ids = torch.randint(0, 1000, (1, 256)).cuda()
    attention_mask = torch.ones_like(input_ids).cuda()

    try:
        print(f"[Rank {rank}] Running forward pass...")
        dist_logits = dist_model(input_ids=input_ids, attention_mask=attention_mask).logits
        print(f"[Rank {rank}] Forward pass successful, logits shape: {dist_logits.shape}")
    except Exception as e:
        print(f"[Rank {rank}] Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    dist.barrier()
    print(f"[Rank {rank}] Test completed successfully")


class TestPhi3Forward(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_phi3_forward(self):
        mp.spawn(test_phi3_forward, nprocs=2, args=(2, self.temp_dir.name))


if __name__ == "__main__":
    suite = unittest.TestSuite()

    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestPhi3Forward))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
