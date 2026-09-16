# coding=utf-8
"""Expert parallelism over the grouped routed experts.

The contract is that a sharded layer is numerically the unsharded layer: same
output, same input gradient, same router gradient, and each rank's expert
gradients equal the corresponding slice of the reference's.
"""

import os
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.testing import assert_close

from specforge.modeling.draft.moe import MoELayer, resolve_moe_config
from specforge.modeling.draft.moe.grouped_experts import (
    stack_grouped_expert_state_dict,
    unstack_grouped_expert_state_dict,
)

N_EXPERTS = 8
HIDDEN = 32


def _config():
    return resolve_moe_config(
        dict(
            moe_preset="deepseek_v4",
            n_routed_experts=N_EXPERTS,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            dflash_config={"moe_bias_update_rate": 1e-3},
        )
    )


def _layer() -> MoELayer:
    torch.manual_seed(0)
    layer = MoELayer(_config(), HIDDEN)
    layer.reset_parameters(std=0.05)
    for parameter in layer.shared_experts.parameters():
        nn.init.normal_(parameter, std=0.05)
    return layer


def _expert_parallel_worker(rank, init_file, starve_a_rank=False):
    if os.uname().sysname == "Darwin":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")

    reference = _layer()
    if starve_a_rank:
        # Force every token onto experts 0..3, so the rank owning 4..7 runs a
        # microbatch with no selected expert at all. Through the selection-only
        # balancing bias, so the scores themselves are untouched: driving the
        # gate WEIGHT instead would push logits into softplus underflow and
        # produce NaN gradients, a different problem than the one under test.
        with torch.no_grad():
            reference.gate.balance.bias[N_EXPERTS // 2 :].fill_(-1e9)
    initial_state = reference.state_dict()
    reference_input = torch.randn(6, HIDDEN, requires_grad=True)

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    with torch.no_grad():
        dist.broadcast(reference_input, src=0)
    reference_output = reference(reference_input)

    # The production layout comes from the draft device mesh; driving the seam
    # directly keeps this test independent of how the group is provisioned.
    patch = mock.patch(
        "specforge.modeling.draft.moe.grouped_experts.draft_ep_layout",
        return_value=(dist.group.WORLD, rank, 2),
    )
    patch.start()
    try:
        sharded = _layer()
        local = N_EXPERTS // 2
        assert sharded.experts.ep_size == 2
        assert sharded.experts.w1.shape[0] == local
        # The full stacked checkpoint is sliced down to this rank on load.
        sharded.load_state_dict(initial_state, strict=False)
        expected_slice = slice(rank * local, (rank + 1) * local)
        assert_close(
            sharded.experts.w1, reference.experts.w1[expected_slice], rtol=0, atol=0
        )

        sharded_input = reference_input.detach().clone().requires_grad_(True)
        sharded_output = sharded(sharded_input)

        cotangent = torch.randn_like(reference_output)
        dist.broadcast(cotangent, src=0)
        reference_output.backward(cotangent)
        sharded_output.backward(cotangent)

        assert_close(sharded_output, reference_output, rtol=1e-5, atol=1e-6)
        assert_close(sharded_input.grad, reference_input.grad, rtol=1e-5, atol=1e-6)
        assert_close(
            sharded.gate.weight.grad, reference.gate.weight.grad, rtol=1e-5, atol=1e-6
        )
        for name in ("w1", "w2", "w3"):
            assert_close(
                getattr(sharded.experts, name).grad,
                getattr(reference.experts, name).grad[expected_slice],
                rtol=1e-5,
                atol=1e-6,
            )

        # Checkpoint naming carries global expert indices, and round-trips.
        checkpoint = unstack_grouped_expert_state_dict(sharded.state_dict())
        assert f"experts.{rank * local}.w1.weight" in checkpoint
        assert "experts.expert_offset" not in checkpoint
        restacked = stack_grouped_expert_state_dict(checkpoint)
        assert_close(restacked["experts.w1"], sharded.experts.w1, rtol=0, atol=0)
        if rank:
            assert int(restacked["experts.expert_offset"]) == rank * local
    finally:
        patch.stop()
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available(), "torch.distributed is unavailable")
class ExpertParallelMoETest(unittest.TestCase):
    def _spawn(self, *args):
        if dist.is_initialized():
            self.skipTest("requires ownership of the singleton process group")
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _expert_parallel_worker,
                args=(os.path.join(directory, "gloo-init"), *args),
                nprocs=2,
                join=True,
            )

    def test_two_rank_sharding_matches_the_unsharded_layer(self):
        self._spawn()

    def test_a_rank_with_no_selected_expert_still_reaches_the_collectives(self):
        self._spawn(True)

    def test_without_a_group_the_layer_is_unsharded(self):
        layer = _layer()
        self.assertEqual(layer.experts.ep_size, 1)
        self.assertEqual(layer.experts.w1.shape[0], N_EXPERTS)
        self.assertFalse(
            getattr(layer.experts, "_specforge_rank_local_parameters", False)
        )
        checkpoint = unstack_grouped_expert_state_dict(layer.state_dict())
        self.assertIn("experts.0.w1.weight", checkpoint)
        self.assertNotIn("experts.expert_offset", checkpoint)


if __name__ == "__main__":
    unittest.main(verbosity=2)
