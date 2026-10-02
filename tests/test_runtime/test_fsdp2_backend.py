"""FSDP2 numerical, local-master, and checkpoint gates on real process groups."""

import copy
import os
import sys
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor

from specforge.optimizer import BF16Optimizer
from specforge.training.backend import ParallelConfig, create_training_backend


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(7, 7)

    def forward(self, x):
        return x + self.linear(x).tanh()


class TinyDraft(nn.Module):
    _no_split_modules = ["TinyBlock"]

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(TinyBlock(), TinyBlock())
        self.head = nn.Linear(7, 5)
        # FSDP2's dim-0 sharding leaves an empty local shard on rank one.
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.layers(x)

    def project(self, x):
        return self.head(x) * self.scale


class TinyComposite(nn.Module):
    def __init__(self):
        super().__init__()
        self.draft_model = TinyDraft()
        self.embed_tokens = nn.Embedding(11, 7).requires_grad_(False)
        self.lm_head = nn.Linear(7, 5, bias=False).requires_grad_(False)

    def forward(self, x):
        x = x + self.embed_tokens(torch.arange(x.shape[0], device=x.device) % 3)
        hidden = self.draft_model(x)
        # Real DFlash-family objectives also call heads outside draft.forward.
        return self.draft_model.project(hidden) + self.lm_head(hidden)


def _optimizer(model, *, offload=False):
    return BF16Optimizer(
        model,
        lr=1e-2,
        weight_decay=0.01,
        total_steps=8,
        warmup_ratio=0,
        max_grad_norm=0.1,
        offload_master=offload,
    )


def _assert_plain_tensors(value):
    if isinstance(value, torch.Tensor):
        assert not isinstance(value, DTensor), "checkpoint leaked a DTensor"
    elif isinstance(value, dict):
        for item in value.values():
            _assert_plain_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_plain_tensors(item)


def _numeric_worker(rank, world_size, port, workdir):
    from tests.test_runtime import _fixtures as fx

    fx.init_rank_distributed(rank, world_size, port=str(port))
    try:
        for dtype in (torch.float32, torch.bfloat16):
            for sharding in ("SHARD_GRAD_OP", "FULL_SHARD", "NO_SHARD"):
                for offload in (False, True):
                    torch.manual_seed(21)
                    template = TinyComposite().cuda().to(dtype)
                    reference_model = copy.deepcopy(template)
                    reference = create_training_backend(
                        "fsdp", ParallelConfig(), optimizer_factory=_optimizer
                    )
                    reference.prepare_model(
                        reference_model,
                        wrap=False,
                        optimizer_target=reference_model.draft_model,
                    )
                    pc = ParallelConfig.from_distributed(
                        sharding_strategy=sharding, param_dtype=dtype
                    )
                    factory = lambda m: _optimizer(m, offload=offload)
                    backends = []
                    for name in ("fsdp", "fsdp2"):
                        model = copy.deepcopy(template)
                        backend = create_training_backend(
                            name, pc, optimizer_factory=factory
                        )
                        backend.prepare_model(model, optimizer_target=model.draft_model)
                        if name == "fsdp2" and sharding != "NO_SHARD":
                            assert isinstance(model.draft_model.scale, DTensor)
                            assert not isinstance(model.lm_head.weight, DTensor)
                            assert not isinstance(model.embed_tokens.weight, DTensor)
                        assert all(
                            not isinstance(master, DTensor)
                            for master in backend.optimizer.fp32_params
                        )
                        backends.append(backend)

                    def update(backend, *, full_batch=False, step=0):
                        for micro in range(2):
                            x = torch.arange(
                                world_size * 21, device="cuda", dtype=torch.float32
                            ).reshape(world_size * 3, 7)
                            x = (x / 31 + micro * 0.2 + step * 0.3).to(dtype)
                            if not full_batch:
                                x = x[rank * 3 : (rank + 1) * 3]
                            loss = backend.module(x).float().square().mean() / 2
                            backend.backward(loss, is_boundary=micro == 1)
                        backend.scale_gradients(torch.tensor(0.75, device="cuda"))
                        return backend.step(
                            loss_denominator=torch.tensor(11.0, device="cuda")
                        )

                    expected_norm = update(reference, full_batch=True)
                    tolerance = 0.03 if dtype == torch.bfloat16 else 2e-5
                    for backend in backends:
                        actual_norm = update(backend)
                        assert actual_norm > 0.1, "the clipping gate must be exercised"
                        torch.testing.assert_close(
                            actual_norm, expected_norm, rtol=tolerance, atol=tolerance
                        )
                        state = backend.state_dict()
                        _assert_plain_tensors(state)
                        if rank == 0:
                            for key, actual in state["model"].items():
                                if sharding != "NO_SHARD" and key.startswith(
                                    "draft_model."
                                ):
                                    assert actual.device.type == "cpu"
                                expected = reference_model.state_dict()[key].cpu()
                                torch.testing.assert_close(
                                    actual.cpu(),
                                    expected,
                                    rtol=tolerance,
                                    atol=2e-3 if dtype == torch.bfloat16 else 2e-6,
                                )
                            torch.save(
                                state["model"], os.path.join(workdir, "model.pt")
                            )
                        else:
                            # FSDP1 may retain ignored, replicated target tables
                            # in nonzero-rank state dicts; no draft may leak.
                            assert not any(
                                key.startswith("draft_model.") for key in state["model"]
                            ), (backend.name, list(state["model"]))
                        del state["model"]
                        rank_path = os.path.join(workdir, f"rank{rank}.pt")
                        torch.save(state, rank_path)
                        dist.barrier()
                        restored = torch.load(
                            rank_path, map_location="cpu", weights_only=False
                        )
                        restored["model"] = torch.load(
                            os.path.join(workdir, "model.pt"),
                            map_location="cpu",
                            weights_only=True,
                        )
                        resumed_model = copy.deepcopy(template)
                        # Also exercise a CPU-master placement change on resume.
                        resumed = create_training_backend(
                            backend.name,
                            pc,
                            optimizer_factory=lambda m: _optimizer(
                                m, offload=not offload
                            ),
                        )
                        resumed.prepare_model(
                            resumed_model, optimizer_target=resumed_model.draft_model
                        )
                        resumed.load_state_dict(restored)
                        torch.testing.assert_close(
                            torch.get_rng_state(), restored["rng"]["torch"]
                        )
                        torch.testing.assert_close(
                            update(resumed, step=1),
                            update(backend, step=1),
                            rtol=2e-5,
                            atol=2e-6,
                        )
                        actual = resumed.state_dict()["model"]
                        expected = backend.state_dict()["model"]
                        for key in actual:
                            torch.testing.assert_close(
                                actual[key], expected[key], rtol=0, atol=2e-6
                            )
                        for a, b in zip(
                            resumed.optimizer.fp32_params, backend.optimizer.fp32_params
                        ):
                            torch.testing.assert_close(
                                a.cpu(), b.cpu(), rtol=2e-5, atol=2e-7
                            )
                        assert (
                            resumed.optimizer.scheduler.last_epoch
                            == backend.optimizer.scheduler.last_epoch
                        )
                        dist.barrier()
    finally:
        from specforge.distributed import destroy_distributed

        destroy_distributed(abort=sys.exc_info()[0] is not None)


class TestBackendSelection(unittest.TestCase):
    def test_default_and_explicit_sharding(self):
        from specforge.config.schema import TrainingConfig

        self.assertEqual(TrainingConfig().backend, "fsdp")
        self.assertEqual(TrainingConfig(backend="fsdp2").backend, "fsdp2")
        with self.assertRaises(ValueError):
            TrainingConfig(backend="unknown")
        with mock.patch.dict(os.environ, {"FSDP_SHARDING": "NO_SHARD"}):
            self.assertEqual(
                ParallelConfig.from_distributed().sharding_strategy, "NO_SHARD"
            )
            self.assertEqual(
                ParallelConfig.from_distributed(
                    sharding_strategy="FULL_SHARD"
                ).sharding_strategy,
                "FULL_SHARD",
            )

    def test_checkpoint_backend_mismatch_fails_before_loading_weights(self):
        def build(name):
            model = nn.Linear(7, 5)
            backend = create_training_backend(
                name, ParallelConfig(), optimizer_factory=_optimizer
            )
            backend.prepare_model(model, wrap=False)
            return backend

        for source, target in (("fsdp", "fsdp2"), ("fsdp2", "fsdp")):
            state = build(source).state_dict()
            backend = build(target)
            before = copy.deepcopy(backend.module.state_dict())
            with self.assertRaisesRegex(ValueError, "optimizer checkpoint backend"):
                backend.load_state_dict(state)
            for name, param in backend.module.state_dict().items():
                torch.testing.assert_close(param, before[name], rtol=0, atol=0)

        legacy = build("fsdp").state_dict()
        del legacy["metadata"]
        with mock.patch("specforge.optimizer.print_on_rank0"):
            build("fsdp").load_state_dict(legacy)
        with self.assertRaisesRegex(ValueError, "optimizer checkpoint backend"):
            build("fsdp2").load_state_dict(legacy)

    def test_unknown_backend_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported training backend"):
            create_training_backend("unknown", ParallelConfig())


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestFSDP2Distributed(unittest.TestCase):
    def test_dense_equivalence_and_resume(self):
        from tests.utils import get_available_port

        with tempfile.TemporaryDirectory(prefix="fsdp2_backend_") as workdir:
            torch.multiprocessing.spawn(
                _numeric_worker,
                args=(2, get_available_port(), workdir),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
