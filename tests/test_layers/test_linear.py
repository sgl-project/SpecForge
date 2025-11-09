import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate.utils import set_seed

from specforge.distributed import gather_tensor, get_tp_group, init_distributed
from specforge.layers import ColumnParallelLinear, RowParallelLinear
from tests.utils import get_available_port


def run_column_parallel_linear(rank, world_size, port):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_distributed(tp_size=world_size)
    set_seed(42)

    # ===============================
    # Case 1: normal layout
    # ===============================
    # create data
    data = torch.rand(1, 256).cuda()

    # create layers
    native_linear = torch.nn.Linear(256, 512).cuda()
    sf_linear = ColumnParallelLinear(256, 512, layout_type="normal").cuda()
    sf_linear.load_state_dict(native_linear.state_dict())

    # forward
    native_output = native_linear(data)
    sf_output = sf_linear(data)
    full_sf_output = gather_tensor(sf_output, get_tp_group())

    # check
    assert torch.allclose(
        native_output, full_sf_output, rtol=1e-5, atol=1e-5
    ), f"native_output: \n{native_output}, \nsf_output: \n{sf_output}"

    # ===============================
    # Case 2: merged QKV layout
    # ===============================
    # create data
    data = torch.rand(1, 256 * 3).cuda()

    # create layers
    native_linear = torch.nn.Linear(256 * 3, 512).cuda()
    sf_linear = ColumnParallelLinear(256 * 3, 512, layout_type="merged_qkv").cuda()
    sf_linear.load_state_dict(native_linear.state_dict())

    # forward
    q, k, v = native_linear(data).chunk(3, dim=1)
    sf_q, sf_k, sf_v = sf_linear(data).chunk(3, dim=1)
    full_sf_q = gather_tensor(sf_q, get_tp_group())
    full_sf_k = gather_tensor(sf_k, get_tp_group())
    full_sf_v = gather_tensor(sf_v, get_tp_group())

    # check
    assert torch.allclose(
        q, full_sf_q, rtol=1e-5, atol=1e-5
    ), f"q: \n{q}, \nfull_sf_q: \n{full_sf_q}"
    assert torch.allclose(
        k, full_sf_k, rtol=1e-5, atol=1e-5
    ), f"k: \n{k}, \nfull_sf_k: \n{full_sf_k}"
    assert torch.allclose(
        v, full_sf_v, rtol=1e-5, atol=1e-5
    ), f"v: \n{v}, \nfull_sf_v: \n{full_sf_v}"

    # ===============================
    # Case 3: gate_up layout
    # ===============================
    # create data
    data = torch.rand(1, 256 * 2).cuda()

    # create layers
    native_linear = torch.nn.Linear(256 * 2, 512).cuda()
    sf_linear = ColumnParallelLinear(256 * 2, 512, layout_type="gate_up").cuda()
    sf_linear.load_state_dict(native_linear.state_dict())

    # forward
    gate, up = native_linear(data).chunk(2, dim=1)
    sf_gate, sf_up = sf_linear(data).chunk(2, dim=1)
    full_sf_gate = gather_tensor(sf_gate, get_tp_group())
    full_sf_up = gather_tensor(sf_up, get_tp_group())

    # check
    assert torch.allclose(
        gate, full_sf_gate, rtol=1e-5, atol=1e-5
    ), f"gate: \n{gate}, \nfull_sf_gate: \n{full_sf_gate}"
    assert torch.allclose(
        up, full_sf_up, rtol=1e-5, atol=1e-5
    ), f"up: \n{up}, \nfull_sf_up: \n{full_sf_up}"

    dist.destroy_process_group()


def run_row_parallel_linear(rank, world_size, port):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_distributed(tp_size=world_size)
    set_seed(42)

    # ===============================
    # Case 1: normal layout
    # the data in an parallel input, i.g.
    # Y = AllReduce(X_i * W_i)
    # ===============================
    # create data
    data = torch.rand(1, 256).cuda()

    # create layers
    native_linear = torch.nn.Linear(256, 512).cuda()
    sf_linear = RowParallelLinear(256, 512, layout_type="normal").cuda()
    sf_linear.load_state_dict(native_linear.state_dict())

    # forward
    native_output = native_linear(data)
    sf_output = sf_linear(data.chunk(world_size, dim=0)[rank])
    dist.all_reduce(sf_output, op=dist.ReduceOp.SUM, group=get_tp_group())

    # check
    assert torch.allclose(
        native_output, sf_output, rtol=1e-5, atol=1e-5
    ), f"native_output: \n{native_output}, \nfull_sf_output: \n{full_sf_output}"


class TestLinear(unittest.TestCase):

    def test_column_parallel_linear(self):
        port = get_available_port()
        mp.spawn(run_column_parallel_linear, nprocs=2, args=(2, port))

    def test_column_parallel_linear(self):
        port = get_available_port()
        mp.spawn(run_column_parallel_linear, nprocs=1, args=(1, port))


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestLinear))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
