# coding=utf-8
"""Expert parallelism over the grouped routed experts, and on the FSDP2 backend.

The layer contract: an EP-sharded layer is numerically the unsharded layer run
on the EP group's tokens. Each rank gets its own tokens' outputs and input
gradients; expert gradients equal the reference's slice (the owner sees the
whole group's tokens); the replicated router and shared expert hold partial
gradients that sum over the group to the reference's. A rank whose experts
receive no token must still complete every collective.

The backend contract: under ``FSDP2TrainingBackend`` with
``expert_parallel_size > 1`` every gradient matches a data-parallel reference
(dense ones through FSDP averaging, expert ones after the ``1 / ep`` rescale),
the full checkpoint keeps the official per-expert naming, and it reloads.

CPU gloo with 2 and 4 processes; no CUDA needed.
"""

import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.testing import assert_close

from specforge.modeling.draft.moe import (
    MoELayer,
    iter_moe_layers,
    resolve_moe_config,
    to_checkpoint_state_dict,
)

N_EXPERTS = 8
TOPK = 2
HIDDEN = 32
TOKENS = 6


def _config():
    return resolve_moe_config(
        dict(
            moe_preset="deepseek_v4",
            n_routed_experts=N_EXPERTS,
            num_experts_per_tok=TOPK,
            moe_intermediate_size=16,
            dflash_config={"moe_bias_update_rate": 1e-3},
        )
    )


def _init_layer(layer: MoELayer) -> MoELayer:
    layer.reset_parameters(std=0.05)
    for parameter in layer.shared_experts.parameters():
        nn.init.normal_(parameter, std=0.05)
    return layer


def _layer(seed: int = 0) -> MoELayer:
    torch.manual_seed(seed)
    return _init_layer(MoELayer(_config(), HIDDEN))


def _init_process_group(rank: int, world: int, init_file: str) -> None:
    if os.uname().sysname == "Darwin":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world
    )


def _gather_all(tensor: torch.Tensor, world: int) -> torch.Tensor:
    parts = [torch.empty_like(tensor) for _ in range(world)]
    dist.all_gather(parts, tensor)
    return torch.cat(parts)


def _full(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


# --------------------------------------------------------------------------
# Layer level: a 1-D ``ep`` mesh over two ranks, no FSDP.
# --------------------------------------------------------------------------
def _layer_worker(rank, init_file, starve_a_rank):
    world = 2
    _init_process_group(rank, world, init_file)
    try:
        reference = _layer()
        if starve_a_rank:
            # Force every token onto experts 0..3 so the rank owning 4..7 sees a
            # micro-batch with no selected expert at all. Through the
            # selection-only balancing bias, so the combine scores are untouched.
            with torch.no_grad():
                reference.gate.balance.bias[N_EXPERTS // 2 :].fill_(-1e9)
        sharded = _layer()
        sharded.load_state_dict(reference.state_dict())
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("ep",))
        layout = sharded.apply_expert_parallel(mesh)
        local = N_EXPERTS // world
        assert layout.size == world and layout.expert_start == rank * local
        assert isinstance(sharded.experts.w1, DTensor)
        assert sharded.experts.w1.to_local().shape[0] == local
        expert_slice = slice(rank * local, (rank + 1) * local)
        for name in ("w1", "w2", "w3"):
            assert_close(
                getattr(sharded.experts, name).to_local(),
                getattr(reference.experts, name)[expert_slice],
                rtol=0,
                atol=0,
            )

        torch.manual_seed(100 + rank)
        x_local = torch.randn(TOKENS, HIDDEN)
        cotangent_local = torch.randn(TOKENS, HIDDEN)
        x_all = _gather_all(x_local, world).requires_grad_(True)
        cotangent_all = _gather_all(cotangent_local, world)
        token_slice = slice(rank * TOKENS, (rank + 1) * TOKENS)

        # Reference: the unsharded layer over the whole group's tokens, with the
        # sum of every rank's loss.
        reference_output = reference(x_all)
        (reference_output * cotangent_all).sum().backward()

        sharded_input = x_local.clone().requires_grad_(True)
        sharded_output = sharded(sharded_input)
        (sharded_output * cotangent_local).sum().backward()

        assert_close(
            sharded_output, reference_output[token_slice], rtol=1e-5, atol=1e-6
        )
        assert_close(sharded_input.grad, x_all.grad[token_slice], rtol=1e-5, atol=1e-6)
        # The owner's expert gradient covers the whole group's tokens.
        for name in ("w1", "w2", "w3"):
            sharded_grad = getattr(sharded.experts, name).grad
            assert sharded_grad is not None, f"{name} received no gradient"
            assert_close(
                sharded_grad.to_local(),
                getattr(reference.experts, name).grad[expert_slice],
                rtol=1e-5,
                atol=1e-6,
            )
        # Replicated parameters hold partial gradients that sum over the group.
        replicated = [(sharded.gate.weight, reference.gate.weight)]
        replicated += list(
            zip(
                sharded.shared_experts.parameters(),
                reference.shared_experts.parameters(),
            )
        )
        for sharded_param, reference_param in replicated:
            total = sharded_param.grad.clone()
            dist.all_reduce(total)
            assert_close(total, reference_param.grad, rtol=1e-5, atol=1e-6)
        # Routing statistics describe the gathered tokens.
        assert int(sharded.last_counts.sum()) == world * TOKENS * TOPK
        # The full state keeps the stacked layout and converts to official naming.
        full = {key: _full(value) for key, value in sharded.state_dict().items()}
        assert_close(full["experts.w1"], reference.experts.w1, rtol=0, atol=0)
        official = to_checkpoint_state_dict(full)
        assert "experts.0.w1.weight" in official
        assert f"experts.{N_EXPERTS - 1}.w3.weight" in official
        assert "gate.bias" in official
    finally:
        dist.destroy_process_group()


# --------------------------------------------------------------------------
# Backend level: FSDP2 over four ranks with a 2-D ``(efsdp, ep)`` mesh.
# --------------------------------------------------------------------------
class _Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.mlp = MoELayer(cfg, HIDDEN)

    def forward(self, x):
        return x + self.mlp(self.proj(x))


class _TinyMoEDraft(nn.Module):
    _no_split_modules = ["_Block"]

    def __init__(self, cfg, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([_Block(cfg) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(HIDDEN)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


def _tiny(seed: int = 0) -> _TinyMoEDraft:
    torch.manual_seed(seed)
    model = _TinyMoEDraft(_config())
    for layer in iter_moe_layers(model):
        _init_layer(layer)
    return model


def _backend_worker(rank, init_file, ep):
    world = 4
    _init_process_group(rank, world, init_file)
    try:
        from specforge.training.backend import ParallelConfig
        from specforge.training.fsdp2 import FSDP2TrainingBackend

        reference = _tiny()
        model = _tiny()
        model.load_state_dict(reference.state_dict())
        ep_mesh = init_device_mesh(
            "cpu", (world // ep, ep), mesh_dim_names=("efsdp", "ep")
        )
        parallel = ParallelConfig(
            world_size=world,
            expert_parallel_size=ep,
            sharding_strategy="FULL_SHARD",
            param_dtype=torch.float32,
            fsdp_process_group=dist.group.WORLD,
            draft_ep_mesh=ep_mesh,
        )
        backend = FSDP2TrainingBackend(parallel)
        wrapped = backend.prepare_model(model, optimizer_target=model)
        assert backend.auto_wrap_block_classes == {_Block}
        for layer in iter_moe_layers(wrapped):
            assert layer.ep is not None and layer.ep.size == ep
            assert isinstance(layer.experts.w1, DTensor)
            assert "ep" in layer.experts.w1.device_mesh.mesh_dim_names

        torch.manual_seed(200 + rank)
        x_local = torch.randn(TOKENS, HIDDEN)
        cotangent_local = torch.randn(TOKENS, HIDDEN)
        x_all = _gather_all(x_local, world)
        cotangent_all = _gather_all(cotangent_local, world)
        token_slice = slice(rank * TOKENS, (rank + 1) * TOKENS)

        # Data-parallel reference: FSDP averages the per-rank losses.
        reference_output = reference(x_all)
        ((reference_output * cotangent_all).sum() / world).backward()

        output = wrapped(x_local)
        backend.backward((output * cotangent_local).sum(), is_boundary=True)
        assert_close(output, reference_output[token_slice], rtol=1e-5, atol=1e-6)

        reference_params = dict(reference.named_parameters())
        for name, parameter in wrapped.named_parameters():
            assert parameter.grad is not None, name
            assert_close(
                _full(parameter.grad),
                reference_params[name].grad,
                rtol=1e-4,
                atol=1e-6,
                msg=lambda m, n=name: f"{n}: {m}",
            )

        # Full checkpoint in the official naming, gathered on rank zero.
        state = backend.state_dict()
        model_state = state["model"]
        if rank == 0:
            assert "layers.0.mlp.experts.0.w1.weight" in model_state
            assert f"layers.1.mlp.experts.{N_EXPERTS - 1}.w2.weight" in model_state
            assert "layers.0.mlp.experts.w1" not in model_state
            assert "layers.0.mlp.gate.bias" in model_state
            for name, value in model_state.items():
                assert not isinstance(value, DTensor), name
            assert_close(
                model_state["layers.1.mlp.experts.5.w2.weight"],
                reference.layers[1].mlp.experts.w2[5],
                rtol=0,
                atol=0,
            )
        else:
            assert model_state == {}
        # Every rank reads the same file in production; broadcast the payload
        # here, perturb the live weights and reload.
        payload = [state if rank == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        with torch.no_grad():
            for parameter in wrapped.parameters():
                local = (
                    parameter.to_local()
                    if isinstance(parameter, DTensor)
                    else parameter
                )
                local.zero_()
        backend.load_state_dict(payload[0])
        for name, parameter in wrapped.named_parameters():
            assert_close(_full(parameter), reference_params[name], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available(), "torch.distributed is unavailable")
class ExpertParallelMoETest(unittest.TestCase):
    def _spawn(self, worker, nprocs, *args):
        if dist.is_initialized():
            self.skipTest("requires ownership of the singleton process group")
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                worker,
                args=(os.path.join(directory, "gloo-init"), *args),
                nprocs=nprocs,
                join=True,
            )

    def test_two_rank_sharding_matches_the_unsharded_layer(self):
        self._spawn(_layer_worker, 2, False)

    def test_a_rank_with_no_selected_expert_still_reaches_the_collectives(self):
        self._spawn(_layer_worker, 2, True)

    def test_fsdp2_backend_with_ep2_matches_data_parallel_reference(self):
        self._spawn(_backend_worker, 4, 2)

    def test_fsdp2_backend_with_ep4_matches_data_parallel_reference(self):
        self._spawn(_backend_worker, 4, 4)

    def test_without_expert_parallelism_the_layer_is_unchanged(self):
        layer = _layer()
        self.assertIsNone(layer.ep)
        self.assertEqual(layer.experts.n_local_experts, N_EXPERTS)
        self.assertFalse(isinstance(layer.experts.w1, DTensor))
        output = layer(torch.randn(TOKENS, HIDDEN))
        self.assertEqual(tuple(output.shape), (TOKENS, HIDDEN))
        official = to_checkpoint_state_dict(layer.state_dict())
        self.assertIn("experts.0.w1.weight", official)


if __name__ == "__main__":
    unittest.main(verbosity=2)
