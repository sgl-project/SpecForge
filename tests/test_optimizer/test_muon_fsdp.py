"""Muon over sharded masters: two processes (Gloo on CPU; NCCL on two GPUs).

* Hand-made uneven and empty slices (every Muon matrix split differently,
  some wholly on one rank, one matrix without a gradient on one update) equal
  the unsharded ``torch.optim.Muon`` + ``AdamW`` oracle bit for bit.
* Real FSDP1 through the production backend and optimizer factory
  (``SHARD_GRAD_OP``, ``FULL_SHARD``; ``NO_SHARD`` = DDP, full matrices)
  equals the same oracle bit for bit on every update, and a run resumed from
  a mid-run backend state (weights + per-rank optimizer state) ends bit for
  bit where the uninterrupted run ends.
"""

from __future__ import annotations

import copy
import functools
import os
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _muon_helpers import (  # noqa: E402
    HAS_TORCH_MUON,
    Composite,
    TorchReference,
    constant_warmup_lr,
    expected_muon_names,
    tiny_draft,
    tiny_inputs,
    tree_differences,
)

WORLD = 2
LR, MUON_LR = 1e-3, 3e-2
STEPS, CUT, W = 6, 3, 2


def _init(rank, init_file, device_type="cpu"):
    torch.set_num_threads(1)
    if device_type == "cuda":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl" if device_type == "cuda" else "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=120),
    )
    return torch.device("cuda", rank) if device_type == "cuda" else torch.device("cpu")


def _gather_flat(tensors):
    """Concatenate every rank's local slices (rank order) of each tensor."""
    local = [t.detach().reshape(-1).cpu().clone() for t in tensors]
    everyone = [None] * WORLD
    dist.all_gather_object(everyone, local)
    return [
        torch.cat([everyone[r][i] for r in range(WORLD)]) for i in range(len(tensors))
    ]


def _momentum(optimizer):
    muon = optimizer.muon_optimizer
    return [
        muon.state[p]["momentum_buffer"] if p in muon.state else p.new_empty(0)
        for p in muon.param_groups[0]["params"]
    ]


def _check_against_reference(optimizer, reference, *, sharded, where, adamw_exact=True):
    masters = (
        _gather_flat(optimizer.fp32_params)
        if sharded
        else [t.cpu() for t in optimizer.fp32_params]
    )
    muon_names = set(optimizer.muon_optimizer.names)
    for name, master in zip(optimizer._parameter_names, masters):
        expected = reference.master(name).reshape(-1).cpu()
        if adamw_exact or name in muon_names:
            assert torch.equal(master.reshape(-1), expected), (where, name)
        else:
            torch.testing.assert_close(
                master.reshape(-1), expected, rtol=1e-6, atol=0, msg=(where, name)
            )
    momenta = (
        _gather_flat(_momentum(optimizer))
        if sharded
        else [t.cpu() for t in _momentum(optimizer)]
    )
    for name, buf in zip(optimizer.muon_optimizer.names, momenta):
        expected = reference.momentum(name).reshape(-1).cpu()
        assert torch.equal(buf.reshape(-1), expected), (where, name)


# --------------------------------------------------------------- hand-made slices
def _split(numel, index):
    """A different cut for every tensor: empty on rank 0 or 1, 1 element, middle, ..."""
    return (0, numel, 1, numel - 1, numel // 3, numel // 2, numel)[index % 7]


def _run_manual_slices(rank, init_file):
    from specforge.muon import capture_parameter_layout
    from specforge.optimizer import BF16Optimizer

    _init(rank, init_file)
    try:
        full = tiny_draft()
        layout = capture_parameter_layout(full)
        sliced = copy.deepcopy(full)
        reference_model = copy.deepcopy(full)
        cuts = {}
        for index, (name, parameter) in enumerate(sliced.named_parameters()):
            if not parameter.requires_grad:
                continue
            first = _split(parameter.numel(), index)
            start, stop = (0, first) if rank == 0 else (first, parameter.numel())
            cuts[name] = (start, stop)
            parameter.data = parameter.data.reshape(-1)[start:stop].clone()
        optimizer = BF16Optimizer(
            sliced,
            lr=LR,
            max_grad_norm=1e9,
            optimizer="muon",
            muon_lr=MUON_LR,
            parameter_layout=layout,
            lr_scheduler="constant",
            warmup_ratio=W / STEPS + 1e-9,
            total_steps=STEPS,
        )
        optimizer.configure_grad_norm_reduction(process_group=None, enabled=True)
        plan = optimizer.muon_optimizer.plan
        assert plan.sharded and plan.world == WORLD
        sizes = list(zip(*plan.sizes))
        assert any(0 in pair for pair in sizes), sizes  # a matrix empty on one rank
        assert any(min(pair) > 1 and pair[0] != pair[1] for pair in sizes), sizes
        reference = TorchReference(
            reference_model, lr=LR, muon_lr=MUON_LR, max_grad_norm=1e9, warmup=W
        )
        skipped = "layers.1.mlp.up_proj.weight"
        for update in range(1, STEPS + 1):
            reference_model(tiny_inputs(update)).backward()
            if update == 4:  # no gradient for one matrix: torch skips it, so must Muon
                reference_model.get_parameter(skipped).grad = None
            for name, parameter in sliced.named_parameters():
                if name not in cuts:
                    continue
                start, stop = cuts[name]
                grad = reference_model.get_parameter(name).grad
                parameter.grad = (
                    None
                    if grad is None or start == stop
                    else grad.reshape(-1)[start:stop].clone()
                )
            optimizer.step()
            reference.step(update)
            # CPU AdamW may round an element of an arbitrarily cut slice's
            # scalar tail 1 ulp off its vectorized full-tensor value; Muon
            # masters and momenta stay exact.
            _check_against_reference(
                optimizer, reference, sharded=True, where=update, adamw_exact=False
            )
    finally:
        dist.destroy_process_group()


# ------------------------------------------------------------------- real FSDP1
def _config():
    from specforge.config import Config

    return Config.model_validate(
        {
            "model": {
                "target_model_path": "target",
                "draft_model_config": "draft.json",
            },
            "data": {"hidden_states_path": "/features"},
            "training": {
                "max_steps": STEPS,
                "learning_rate": LR,
                "max_grad_norm": 1e9,
                "lr_scheduler": "constant",
                "warmup_ratio": W / STEPS + 1e-9,
                "optimizer": "muon",
                "muon_learning_rate": MUON_LR,
            },
        }
    )


def _wrap(model, sharding, device):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from specforge.training.assembly import _ConfiguredOptimizerFactory
    from specforge.training.backend import FSDPTrainingBackend, ParallelConfig

    backend = FSDPTrainingBackend(
        ParallelConfig(world_size=WORLD, sharding_strategy=sharding),
        optimizer_factory=_ConfiguredOptimizerFactory(_config()),
    )
    # Explicit device_id lets real FSDP run on CPU in the unit suite.
    with mock.patch(
        "torch.distributed.fsdp.FullyShardedDataParallel",
        functools.partial(FSDP, device_id=device),
    ):
        wrapped = backend.prepare_model(model, optimizer_target=model.draft_model)
    return backend, wrapped


def _full_weights(backend, wrapped):
    from torch.distributed.fsdp import FullStateDictConfig

    if backend._wrapper_kind == "ddp":
        return copy.deepcopy(wrapped.module.state_dict())
    with backend._full_state_ctx(
        FullStateDictConfig(offload_to_cpu=False, rank0_only=False)
    ):
        return copy.deepcopy(wrapped.state_dict())


def _run_fsdp(rank, init_file, sharding, device_type):
    device = _init(rank, init_file, device_type)
    try:
        sharded = sharding != "NO_SHARD"
        torch.manual_seed(0)
        model = Composite().to(device=device, dtype=torch.bfloat16)
        reference_model = copy.deepcopy(model.draft_model)
        backend, wrapped = _wrap(model, sharding, device)
        optimizer = backend.optimizer
        assert optimizer.optimizer_kind == "muon"
        assert optimizer.muon_optimizer.names == tuple(expected_muon_names())
        plan = optimizer.muon_optimizer.plan
        assert plan.sharded == sharded, (sharding, plan.describe())
        if sharded:
            sizes = list(zip(*plan.sizes))
            assert any(0 in pair for pair in sizes), sizes
            assert any(min(pair) > 0 for pair in sizes), sizes
        reference = TorchReference(
            reference_model, lr=LR, muon_lr=MUON_LR, max_grad_norm=1e9, warmup=W
        )
        if device_type == "cuda":
            # BF16Optimizer runs fused AdamW on CUDA masters.
            reference.adamw = torch.optim.AdamW(
                reference.adamw.param_groups[0]["params"],
                lr=LR,
                weight_decay=0.0,
                fused=True,
            )
        snapshot = None
        for update in range(1, STEPS + 1):
            x = tiny_inputs(update, device=device)
            assert optimizer.learning_rates_by_group() == {
                "adamw": constant_warmup_lr(update, LR, W),
                "muon": constant_warmup_lr(update, MUON_LR, W),
            }
            backend.backward(wrapped(x))
            norm = backend.step()
            reference_model(x).backward()
            reference_norm = reference.step(update)
            torch.testing.assert_close(norm, reference_norm, rtol=1e-6, atol=0)
            _check_against_reference(
                optimizer, reference, sharded=sharded, where=update
            )
            if update == CUT:
                snapshot = (
                    _full_weights(backend, wrapped),
                    copy.deepcopy(optimizer.state_dict()),
                )
        final_optimizer = copy.deepcopy(optimizer.state_dict())
        final_weights = _full_weights(backend, wrapped)

        # Resume: weights load before the wrap (the Trainer's order), the
        # rank's own optimizer state after it; then the same updates again.
        torch.manual_seed(123)
        resumed = Composite().to(device=device, dtype=torch.bfloat16)
        resumed.load_state_dict(snapshot[0])
        backend2, wrapped2 = _wrap(resumed, sharding, device)
        with mock.patch("specforge.optimizer.print_on_rank0"):
            backend2.load_state_dict({"optimizer": snapshot[1]})
        for update in range(CUT + 1, STEPS + 1):
            backend2.backward(wrapped2(tiny_inputs(update, device=device)))
            backend2.step()
        assert tree_differences(backend2.optimizer.state_dict(), final_optimizer) == []
        assert tree_differences(_full_weights(backend2, wrapped2), final_weights) == []
    finally:
        dist.destroy_process_group()


def _spawn(target, *args):
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(
            target,
            args=(os.path.join(directory, "pg"), *args),
            nprocs=WORLD,
            join=True,
        )


@unittest.skipUnless(HAS_TORCH_MUON, "torch.optim.Muon (torch >= 2.9) is the oracle")
class TestShardedMuon(unittest.TestCase):
    def test_uneven_and_empty_slices_match_the_unsharded_oracle(self):
        _spawn(_run_manual_slices)

    def test_fsdp_and_ddp_match_the_oracle_and_resume_bitwise(self):
        for sharding in ("SHARD_GRAD_OP", "FULL_SHARD", "NO_SHARD"):
            with self.subTest(sharding=sharding):
                _spawn(_run_fsdp, sharding, "cpu")

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
    def test_nccl_fsdp_and_ddp_match_the_oracle_and_resume_bitwise(self):
        for sharding in ("SHARD_GRAD_OP", "FULL_SHARD", "NO_SHARD"):
            with self.subTest(sharding=sharding):
                _spawn(_run_fsdp, sharding, "cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
