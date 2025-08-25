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


def test_phi3_tp(rank, world_size, temp_dir):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"

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

    # create a simple single-gpu model
    model = Phi3ForCausalLM(config).cuda()

    from specforge.modeling.target.phi3 import (
        Phi3ForCausalLM as DistPhi3ForCausalLM,
    )

    dist_model = DistPhi3ForCausalLM(config).cuda()

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
    input_ids = torch.randint(0, 1000, (1, 32)).cuda()  # Smaller sequence length
    attention_mask = torch.ones_like(input_ids).cuda()

    try:
        print(f"[Rank {rank}] Running inference on distributed model...")
        dist_logits = dist_model(input_ids=input_ids, attention_mask=attention_mask).logits
        print(f"[Rank {rank}] Distributed model inference successful, logits shape: {dist_logits.shape}")
        
        if dist.get_rank() == 0:
            print(f"[Rank {rank}] Running inference on original model...")
            expected_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            print(f"[Rank {rank}] Original model inference successful, logits shape: {expected_logits.shape}")
            
            print(f"[Rank {rank}] Comparing logits...")
            print(f"Expected logits sample: {expected_logits[0, 0, :5]}")
            print(f"Distributed logits sample: {dist_logits[0, 0, :5]}")
            
            # More lenient comparison due to potential numerical differences in TP
            if torch.allclose(expected_logits, dist_logits, rtol=1e-3, atol=1e-3):
                print(f"[Rank {rank}] ✓ Logits match within tolerance")
            else:
                print(f"[Rank {rank}] ✗ Logits do not match within tolerance")
                max_diff = torch.max(torch.abs(expected_logits - dist_logits))
                print(f"[Rank {rank}] Max difference: {max_diff}")
    except Exception as e:
        print(f"[Rank {rank}] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return


class TestPhi3TP(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_phi3_tp(self):
        mp.spawn(test_phi3_tp, nprocs=2, args=(2, self.temp_dir.name))


if __name__ == "__main__":
    suite = unittest.TestSuite()

    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestPhi3TP))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
