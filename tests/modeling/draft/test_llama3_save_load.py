# Filename: test_tp_correctness.py (Final version with tests for both MLP and Attention)

import logging
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from accelerate.utils import set_seed
from transformers import LlamaConfig

from specforge.distributed import destroy_distributed, init_distributed
from specforge.modeling.draft import AutoEagle3DraftModel

logging.basicConfig(level=logging.INFO)


def run_save_load_test(rank, world_size, temp_dir_name):
    """This function executes the parallel computation for Attention and compares it with the 'golden standard'."""
    os.environ["RANK"], os.environ["WORLD_SIZE"] = str(rank), str(world_size)
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "localhost", "29504"
    init_distributed(draft_tp_size=world_size)
    torch.cuda.set_device(rank)
    config = LlamaConfig(
        hidden_size=128,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=512,
        num_hidden_layers=1,
        vocab_size=30,
    )
    config.draft_vocab_size = 60
    set_seed(42)
    randn_hidden_states = torch.randn(
        1, 5, config.hidden_size * 3, dtype=torch.bfloat16
    ).cuda()
    randn_input_embeds = torch.randn(
        1, 5, config.hidden_size, dtype=torch.bfloat16
    ).cuda()
    draft_model = (
        AutoEagle3DraftModel.from_config(
            config, torch_dtype=torch.bfloat16, attention_backend="sdpa"
        )
        .cuda()
        .eval()
    )
    # draft_model.sync_state_dict_across_tp()
    output_before_save = draft_model(
        hidden_states=randn_hidden_states, inputs_embeds=randn_input_embeds
    )

    all_outpus = [torch.zeros_like(output_before_save) for _ in range(world_size)]
    dist.all_gather(all_outpus, output_before_save)
    if rank == 0:
        for i in range(1, world_size):
            assert torch.allclose(
                all_outpus[0], all_outpus[i], rtol=1e-3, atol=1e-3
            ), f"mismatch between rank {i} and rank 0 {torch.max(torch.abs(all_outpus[0] - all_outpus[i]))}"
    dist.barrier()

    draft_model.save_pretrained(temp_dir_name)
    output_after_save = draft_model(
        hidden_states=randn_hidden_states, inputs_embeds=randn_input_embeds
    )
    assert torch.allclose(
        output_before_save, output_after_save, rtol=1e-3, atol=1e-3
    ), f"between save and load! {torch.max(torch.abs(output_before_save - output_after_save))}"
    dist.barrier()
    draft_model_loaded = (
        AutoEagle3DraftModel.from_pretrained(
            temp_dir_name,
            torch_dtype=torch.bfloat16,
            attention_backend="sdpa",
            ignore_mismatched_sizes=False,
        )
        .cuda()
        .eval()
    )
    output_after_load = draft_model_loaded(
        hidden_states=randn_hidden_states, inputs_embeds=randn_input_embeds
    )

    state_dict_before_load = draft_model_loaded.state_dict()
    state_dict_after_load = draft_model.state_dict()
    for name, param_before_load in state_dict_before_load.items():
        param_after_load = state_dict_after_load[name]
        assert torch.allclose(
            param_before_load, param_after_load, rtol=1e-3, atol=1e-3
        ), f"{rank=} state dict mismatch between save and load! {name} {param_before_load=} vs {param_after_load=}"

    assert torch.allclose(
        output_before_save, output_after_load, rtol=1e-3, atol=1e-3
    ), f"{rank=} between save and load {torch.max(torch.abs(output_before_save - output_after_load))}"
    dist.barrier()
    if rank == 0:
        print("✅ LlamaForCausalLMEagle3 save and load correctness test passed!")
    destroy_distributed()


# === unittest Launcher ===
class TestSaveLoadCorrectness(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_load_correctness(self):
        world_size = 2
        temp_dir_path = self.temp_dir.name
        print("\n--- Running Save and Load Test ---")
        mp.spawn(
            run_save_load_test, nprocs=world_size, args=(world_size, temp_dir_path)
        )


if __name__ == "__main__":
    unittest.main()
