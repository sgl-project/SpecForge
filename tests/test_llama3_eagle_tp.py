import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate.utils import set_seed
from transformers import LlamaConfig, LlamaForCausalLM

from sgl_spec.distributed import init_distributed
from sgl_spec.modeling.draft.llama3_eagle import LlamaForCausalLMEagle3


def test_llama3_eagle_tp(rank, world_size, temp_dir):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"

    init_distributed(tp_size=2)
    set_seed(42)
    config = LlamaConfig(
        vocab_size=1000,
        draft_vocab_size=500,
        hidden_size=384,
        intermediate_size=512,
        num_hidden_layers=2,
        max_position_embeddings=1024,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=48,
        initializer_range=0.02,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        pad_token_id=0,
    )

    model = LlamaForCausalLM(config).cuda()

    dist_model = LlamaForCausalLMEagle3(config).cuda()

    if dist.get_rank() == 0:
        torch.save(model.state_dict(), os.path.join(temp_dir, "model.pt"))
    dist.barrier()

    dist_model.load_state_dict(
        torch.load(os.path.join(temp_dir, "model.pt"), map_location="cuda")
    )
    dist.barrier()

    batch_size, seq_len = 1, 16

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).cuda()
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size * 3).cuda()
    inputs_embeds = dist_model.embed_input_ids(input_ids)

    with torch.no_grad():
        expected = model(hidden_states, inputs_embeds)
        actual = dist_model(hidden_states, inputs_embeds)

    assert torch.allclose(
        expected, actual, rtol=1e-5, atol=1e-5
    ), f"Logits not close: {expected} vs {actual}"


class TestLlama3EagleTP(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_llama3_eagle_tp(self):
        mp.spawn(test_llama3_eagle_tp, nprocs=2, args=(2, self.temp_dir.name))


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestLlama3EagleTP))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
