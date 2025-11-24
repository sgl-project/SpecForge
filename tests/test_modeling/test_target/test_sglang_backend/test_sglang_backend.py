import os
import unittest

import torch
import torch.multiprocessing as mp
from accelerate.utils import set_seed

from specforge.distributed import init_distributed
from specforge.modeling.target.eagle3_target_model import SGLangEagle3TargetModel
from tests.utils import get_available_port


@torch.no_grad()
def test_dense(rank, world_size, port, tp_size):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    init_distributed(tp_size=tp_size)
    set_seed(42)

    input_ids = torch.randint(0, 1000, (2, 256)).cuda()
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.ones_like(input_ids)

    # test dense model
    sgl_target_model = SGLangEagle3TargetModel.from_pretrained(
        "unsloth/Llama-3.2-1B",
        torch_dtype=torch.float16,
        device="cuda",
        attention_backend="fa3",
        mem_fraction_static=0.4,
        enable_torch_compile=True,
        enable_nccl_nvls=True,
        enable_symm_mem=True,
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        enable_piecewise_cuda_graph=True,
        ep_size=1,
        context_length=256,
    )
    sgl_target_model.set_aux_hidden_states_layers()
    sgl_out = sgl_target_model.generate_eagle3_data(
        input_ids=input_ids, attention_mask=attention_mask, loss_mask=loss_mask
    )


@torch.no_grad()
def test_moe(rank, world_size, port, tp_size):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    init_distributed(tp_size=tp_size)
    set_seed(42)

    input_ids = torch.randint(0, 1000, (2, 256)).cuda()
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.ones_like(input_ids)

    # test moe model
    sgl_target_model = SGLangEagle3TargetModel.from_pretrained(
        "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        torch_dtype=torch.float16,
        device="cuda",
        attention_backend="fa3",
        mem_fraction_static=0.4,
        enable_torch_compile=True,
        enable_nccl_nvls=True,
        enable_symm_mem=True,
        enable_dp_attention=True,
        enable_dp_lm_head=True,
        enable_piecewise_cuda_graph=True,
        ep_size=2,
        context_length=256,
    )
    sgl_target_model.set_aux_hidden_states_layers()
    sgl_out = sgl_target_model.generate_eagle3_data(
        input_ids=input_ids, attention_mask=attention_mask, loss_mask=loss_mask
    )


class TestTargetModelBackend(unittest.TestCase):

    def test_sglang_backend_with_dense(self):
        world_size = 2
        port = get_available_port()
        mp.spawn(test_dense, nprocs=world_size, args=(world_size, port, 2))

    def test_sglang_backend_with_moe(self):
        world_size = 2
        port = get_available_port()
        mp.spawn(test_moe, nprocs=world_size, args=(world_size, port, 2))


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestTargetModelBackend))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
