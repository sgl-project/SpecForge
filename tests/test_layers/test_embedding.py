import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate.utils import set_seed

from specforge.distributed import init_distributed
from specforge.layers import VocabParallelEmbedding
from tests.utils import get_available_port


def run_embedding(rank, world_size, port):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_distributed(tp_size=world_size)
    set_seed(42)

    # ===============================
    # Case 1: vocab size is divisible by the TP size
    # ===============================
    # create layers
    data = torch.randint(0, 512, (1, 128)).long().cuda()
    native_embedding = torch.nn.Embedding(512, 256, padding_idx=314).cuda()
    sf_embedding = VocabParallelEmbedding(512, 256, padding_idx=314).cuda()
    sf_embedding.load_state_dict(native_embedding.state_dict())

    # forward
    native_output = native_embedding(data)
    sf_output = sf_embedding(data)

    # check
    assert torch.allclose(
        native_output, sf_output, rtol=1e-5, atol=1e-5
    ), f"native_output: \n{native_output}, \nsf_output: \n{sf_output}"

    # ===============================
    # Case 2: vocab size is NOT divisible by the TP size
    # ===============================
    # create layers
    data = torch.randint(0, 355, (1, 128)).long().cuda()
    native_embedding = torch.nn.Embedding(355, 256, padding_idx=314).cuda()
    sf_embedding = VocabParallelEmbedding(355, 256, padding_idx=314).cuda()
    sf_embedding.load_state_dict(native_embedding.state_dict())

    # forward
    native_output = native_embedding(data)
    sf_output = sf_embedding(data)

    # check
    assert torch.allclose(
        native_output, sf_output, rtol=1e-5, atol=1e-5
    ), f"native_output: \n{native_output}, \nsf_output: \n{sf_output}"

    dist.destroy_process_group()


class TestEmbedding(unittest.TestCase):

    def test_embedding(self):
        port = get_available_port()
        mp.spawn(run_embedding, nprocs=2, args=(2, port))

        port = get_available_port()
        mp.spawn(run_embedding, nprocs=1, args=(1, port))


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestEmbedding))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
