import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate.utils import set_seed

from specforge.distributed import init_distributed
from specforge.layers import ParallelLMHead, VocabParallelEmbedding
from tests.utils import get_available_port


def run_lm_head(rank, world_size, port):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_distributed(tp_size=world_size)
    set_seed(42)

    # ===============================
    # Case 1: the output vocab size is divisible by the TP size
    # ===============================
    # create data
    data = torch.rand(1, 128, 256).cuda()

    for bias in [True, False]:
        # create layers
        native_lm_head = torch.nn.Linear(256, 512, bias=bias).cuda()
        sf_lm_head = ParallelLMHead(256, 512, bias=bias).cuda()
        sf_lm_head.load_state_dict(native_lm_head.state_dict())

        # forward
        native_output = native_lm_head(data)
        sf_output = sf_lm_head(data, gather_output=True)

        # check
        assert torch.allclose(
            native_output, sf_output, rtol=1e-5, atol=1e-5
        ), f"bias: {bias}, native_output: \n{native_output}, \nsf_output: \n{sf_output}"

        # ===============================
        # Case 2: the output vocab size is not divisible by the TP size
        # ===============================
        # create data
        data = torch.rand(1, 128, 256).cuda()

        # create layers
        native_lm_head = torch.nn.Linear(256, 377, bias=bias).cuda()
        sf_lm_head = ParallelLMHead(256, 377, bias=bias).cuda()
        sf_lm_head.load_state_dict(native_lm_head.state_dict())

        # forward
        native_output = native_lm_head(data)
        sf_output = sf_lm_head(data, gather_output=True)

        # check
        assert torch.allclose(
            native_output, sf_output, rtol=1e-5, atol=1e-5
        ), f"bias: {bias}, native_output: \n{native_output}, \nsf_output: \n{sf_output}"

        # ===============================
        # Case 3: tie word embedding
        # ===============================
        if not bias:
            # there is no bias in the embedding layer so we skip when bias is True
            # create data
            data = torch.rand(128, 256).cuda()

            # create native layers
            native_embedding = torch.nn.Embedding(512, 256).cuda()
            native_lm_head = torch.nn.Linear(256, 512, bias=bias).cuda()
            native_lm_head.weight = native_embedding.weight

            # create specforge layers
            sf_embedding = VocabParallelEmbedding(512, 256).cuda()
            sf_embedding.load_state_dict(native_embedding.state_dict())
            sf_lm_head = ParallelLMHead(256, 512, bias=bias).cuda()
            sf_lm_head.weight = sf_embedding.weight

            # forward
            native_output = native_lm_head(data)
            sf_output = sf_lm_head(data, gather_output=True)

            # check
            assert torch.allclose(
                native_output, sf_output, rtol=1e-5, atol=1e-5
            ), f"bias: {bias}, native_output: \n{native_output}, \nsf_output: \n{sf_output}"

    dist.destroy_process_group()


class TestLMHead(unittest.TestCase):

    def test_lm_head(self):
        port = get_available_port()
        mp.spawn(run_lm_head, nprocs=2, args=(2, port))

        port = get_available_port()
        mp.spawn(run_lm_head, nprocs=1, args=(1, port))


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestLMHead))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
