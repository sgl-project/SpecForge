# coding=utf-8
"""Expert parallelism over the grouped routed experts.

The contract is that a sharded layer is numerically the unsharded layer: same
output, same input gradient, same gate gradient, and each rank's expert
gradients equal the corresponding slice of the reference's.
"""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.testing import assert_close

from specforge.modeling.draft.moe import (
    SparseMoE,
    stack_grouped_expert_state_dict,
    unstack_grouped_expert_state_dict,
)

N_EXPERTS = 8
HIDDEN = 32


def _moe_config():
    return SimpleNamespace(
        hidden_size=HIDDEN,
        n_routed_experts=N_EXPERTS,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        dflash_config={},
    )


def _moe() -> SparseMoE:
    torch.manual_seed(0)
    return SparseMoE(_moe_config())


def _expert_parallel_worker(rank, init_file, starve_a_rank=False):
    if os.uname().sysname == "Darwin":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")

    reference = _moe()
    if starve_a_rank:
        # Force every token onto experts 0..3, so the rank owning 4..7 runs a
        # microbatch with no selected expert at all. Its routed_x and gate
        # outputs then have no differentiable consumer, and without the zero
        # terms in SparseMoE.forward autograd never reaches the expert-parallel
        # seam: its all-reduce is never issued and the peer that did issue one
        # hangs. This is the case an ordinary seed only sometimes produces.
        # Through the selection-only balancing bias, so the scores themselves
        # are untouched: driving the gate WEIGHT instead would push logits into
        # softplus underflow and produce NaN gradients, which is a different
        # problem than the one under test.
        with torch.no_grad():
            reference.gate.bias[N_EXPERTS // 2 :].fill_(-1e9)
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
        "specforge.modeling.draft.moe.draft_ep_layout",
        return_value=(dist.group.WORLD, rank, 2),
    )
    patch.start()
    try:
        sharded = _moe()
        local = N_EXPERTS // 2
        assert sharded.experts.ep_size == 2
        assert sharded.experts.w1.shape[0] == local
        sharded.load_state_dict(initial_state, strict=False)
        expert_slice = slice(rank * local, (rank + 1) * local)
        assert_close(
            sharded.experts.w1, reference.experts.w1[expert_slice], rtol=0, atol=0
        )
        # Everything replicated must be bit-identical, or the comparison below
        # is measuring initialization noise rather than the sharding.
        for name, tensor in reference.state_dict().items():
            if name.startswith("experts."):
                continue
            assert_close(sharded.state_dict()[name], tensor, rtol=0, atol=0)

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
                getattr(reference.experts, name).grad[expert_slice],
                rtol=1e-5,
                atol=1e-6,
            )

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
    def _spawn(self, **kwargs):
        if dist.is_initialized():
            self.skipTest("requires ownership of the singleton process group")
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _expert_parallel_worker,
                args=(os.path.join(directory, "gloo-init"), *kwargs.values()),
                nprocs=2,
                join=True,
            )

    def test_two_rank_sharding_matches_the_unsharded_layer(self):
        self._spawn()

    def test_a_rank_with_no_selected_expert_still_reaches_the_collectives(self):
        self._spawn(starve_a_rank=True)

    def test_without_a_group_the_layer_is_unsharded(self):
        moe = _moe()
        self.assertEqual(moe.experts.ep_size, 1)
        self.assertEqual(moe.experts.w1.shape[0], N_EXPERTS)
        self.assertFalse(
            getattr(moe.experts, "_specforge_rank_local_parameters", False)
        )
        checkpoint = unstack_grouped_expert_state_dict(moe.state_dict())
        self.assertIn("experts.0.w1.weight", checkpoint)
        self.assertNotIn("experts.expert_offset", checkpoint)


if __name__ == "__main__":
    unittest.main(verbosity=2)
