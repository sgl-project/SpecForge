import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate.utils import set_seed
from transformers import GptOssConfig, GptOssForCausalLM

from specforge.distributed import init_distributed
from specforge.modeling.target.custom_backend.gpt_oss import (
    GptOssForCausalLM as DistGptOssForCausalLM,
)
from tests.utils import get_available_port


def test_gpt_oss_tp(rank, world_size, temp_dir, port):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    init_distributed(tp_size=2)
    set_seed(42)

    for tie_word_embeddings in [True, False]:
        config = GptOssConfig(
            vocab_size=1000,
            hidden_size=384,
            intermediate_size=512,
            intermediate_size_mlp=512,
            num_hidden_layers=2,
            max_position_embeddings=1024,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=64,
            num_local_experts=4,
            tie_word_embeddings=tie_word_embeddings,
            initializer_range=0.02,
            hidden_act="silu",
            layer_types=[
                "sliding_attention",
                "full_attention",
            ],
        )

        # create the single-gpu
        model = GptOssForCausalLM(config).cuda().eval()

        if dist.get_rank() == 0:
            model.save_pretrained(temp_dir)
            print(f"Saved model to {temp_dir}")
        dist.barrier()

        # # load the model weights to the distributed model
        print(f"Loading model from {temp_dir}")
        dist_model = DistGptOssForCausalLM.from_pretrained(temp_dir).cuda()
        dist.barrier()

        if tie_word_embeddings:
            assert torch.equal(
                model.get_input_embeddings().weight, model.lm_head.weight
            )
            assert torch.equal(
                dist_model.get_input_embeddings().weight, dist_model.lm_head.weight
            )

        # # create data
        input_ids = torch.randint(0, 1000, (1, 256)).cuda()
        attention_mask = torch.ones_like(input_ids).cuda()

        expected_logits = model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        dist_logits = dist_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits

        assert torch.allclose(
            expected_logits,
            dist_logits,
            rtol=1e-5,
            atol=1e-5,
        ), f"Logits are not close, {expected_logits} vs {dist_logits}"

    dist.destroy_process_group()


class TestGptOssTP(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_gpt_oss_tp(self):
        port = get_available_port()
        mp.spawn(test_gpt_oss_tp, nprocs=2, args=(2, self.temp_dir.name, port))


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestGptOssTP))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
