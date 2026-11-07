"""``training.fp8_linear``: torchao Float8Linear blocks under the FSDP2 backend."""

import json
import os
import tempfile
import types
import unittest
from unittest import mock

import torch
import torch.nn as nn

from specforge.training.backend import (
    BackendOptions,
    ParallelConfig,
    create_training_backend,
)

try:
    import torchao  # noqa: F401

    HAS_TORCHAO = True
except ImportError:  # pragma: no cover
    HAS_TORCHAO = False


class WideBlock(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.up = nn.Linear(dim, 4 * dim, bias=False)
        self.down = nn.Linear(4 * dim, dim, bias=False)
        self.odd = nn.Linear(dim, 24)  # not a multiple of 16: must stay BF16

    def forward(self, x):
        return x + self.down(torch.nn.functional.silu(self.up(x))) + self.odd(x).sum(-1, keepdim=True) * 0


class WideDraft(nn.Module):
    _no_split_modules = ["WideBlock"]

    def __init__(self, dim=64):
        super().__init__()
        self.layers = nn.Sequential(WideBlock(dim), WideBlock(dim))
        self.head = nn.Linear(dim, 32, bias=False)

    def forward(self, x):
        return self.head(self.layers(x))


class WideComposite(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.draft_model = WideDraft(dim)
        self.lm_head = nn.Linear(dim, 32, bias=False).requires_grad_(False)

    def forward(self, x):
        return self.draft_model(x)


class TestFp8Selection(unittest.TestCase):
    def test_fsdp1_rejects_fp8(self):
        pc = ParallelConfig(world_size=1)
        backend = create_training_backend("fsdp", pc, options=BackendOptions(fp8_linear=True))
        with self.assertRaisesRegex(ValueError, "fp8_linear"):
            backend.prepare_model(WideComposite(), optimizer_target=None)

    def test_config_requires_fsdp2_and_recommends_static_shapes(self):
        from specforge.config.schema import TrainingConfig

        with self.assertRaisesRegex(ValueError, "fp8_linear"):
            TrainingConfig(fp8_linear=True, static_shapes=True)
        with self.assertWarnsRegex(UserWarning, "static_shapes"):
            TrainingConfig(backend="fsdp2", fp8_linear=True)
        cfg = TrainingConfig(backend="fsdp2", fp8_linear=True, static_shapes=True)
        self.assertTrue(cfg.fp8_linear)

    def test_config_requires_max_length_multiple_of_16(self):
        from specforge.config import Config

        def cfg(max_length):
            return Config.model_validate(
                {
                    "model": {"target_model_path": "t", "draft_model_config": "d"},
                    "data": {"hidden_states_path": "features", "max_length": max_length},
                    "training": {
                        "strategy": "dflash",
                        "backend": "fsdp2",
                        "fp8_linear": True,
                        "static_shapes": True,
                        "max_steps": 1,
                    },
                }
            )

        with self.assertRaisesRegex(ValueError, "multiple of 16"):
            cfg(1000)
        self.assertEqual(cfg(2048).data.max_length, 2048)
        # Without static_shapes the batch length is data-dependent: no hard check.
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            from specforge.config import Config

            loose = Config.model_validate(
                {
                    "model": {"target_model_path": "t", "draft_model_config": "d"},
                    "data": {"hidden_states_path": "features", "max_length": 1000},
                    "training": {"strategy": "dflash", "backend": "fsdp2", "fp8_linear": True, "max_steps": 1},
                }
            )
        self.assertEqual(loose.data.max_length, 1000)

    def test_data_parallel_size(self):
        from specforge.training.fsdp2 import _data_parallel_size

        self.assertEqual(_data_parallel_size(ParallelConfig(world_size=4)), 4)
        self.assertEqual(_data_parallel_size(ParallelConfig(world_size=4, tp_size=2)), 2)
        self.assertEqual(_data_parallel_size(ParallelConfig(world_size=1)), 1)

    def test_uneven_shards_fall_back_to_bf16_all_gather(self):
        try:
            import torchao  # noqa: F401
        except ImportError:
            self.skipTest("torchao not installed")
        from specforge.training.fsdp2 import _convert_blocks_to_float8

        count, float8_all_gather = _convert_blocks_to_float8([WideBlock()], dp_size=2)
        self.assertEqual(count, 2)
        self.assertTrue(float8_all_gather)
        with self.assertLogs("specforge.training.fsdp2", level="WARNING"):
            count, float8_all_gather = _convert_blocks_to_float8([WideBlock()], dp_size=3)
        self.assertEqual(count, 2)
        self.assertFalse(float8_all_gather)

    def test_filter_skips_non_multiple_of_16(self):
        from specforge.training.fsdp2 import _float8_linear_filter

        self.assertTrue(_float8_linear_filter(nn.Linear(64, 256, bias=False), "up"))
        self.assertFalse(_float8_linear_filter(nn.Linear(64, 24), "odd"))
        frozen = nn.Linear(64, 256, bias=False).requires_grad_(False)
        self.assertFalse(_float8_linear_filter(frozen, "lm_head"))


def _worker(rank, world_size, port, results_dir):
    from tests.test_runtime import _fixtures as fx
    from tests.test_runtime.test_fsdp2_backend import _optimizer

    fx.init_rank_distributed(rank, world_size, port=str(port))
    torch.manual_seed(0)
    model = WideComposite().cuda().to(torch.bfloat16)
    # float8 GEMMs need every dimension (tokens included, for grad_weight) % 16 == 0
    x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    pc = ParallelConfig.from_distributed()
    backend = create_training_backend(
        "fsdp2", pc, optimizer_factory=_optimizer, options=BackendOptions(fp8_linear=True)
    )
    wrapped = backend.prepare_model(model, optimizer_target=model.draft_model)
    names = {type(m).__name__ for m in wrapped.modules()}
    before = [p.detach().clone() for p in wrapped.draft_model.parameters()]
    losses = []
    for _ in range(3):
        loss = wrapped(x).float().pow(2).mean()
        backend.backward(loss, is_boundary=True)
        backend.step()
        losses.append(float(loss.detach()))
    changed = any(
        not torch.equal(a.to_local() if hasattr(a, "to_local") else a,
                        b.to_local() if hasattr(b, "to_local") else b)
        for a, b in zip(before, wrapped.draft_model.parameters())
    )
    out = {
        "float8_modules": backend.fp8_linear_modules,
        "has_float8_linear": "Float8Linear" in names,
        "losses": losses,
        "finite": all(torch.isfinite(torch.tensor(losses)).tolist()),
        "changed": changed,
    }
    with open(os.path.join(results_dir, f"rank{rank}.json"), "w") as f:
        json.dump(out, f)


def _fp8_capable():
    if not (HAS_TORCHAO and torch.cuda.device_count() >= 2):
        return False
    return torch.cuda.get_device_capability(0) >= (8, 9)


@unittest.skipUnless(_fp8_capable(), "requires torchao and two sm_89+ CUDA devices")
class TestFp8Distributed(unittest.TestCase):
    def test_fp8_blocks_train(self):
        import torch.multiprocessing as mp

        results_dir = tempfile.mkdtemp(prefix="fp8_linear_")
        mp.spawn(_worker, args=(2, 29617, results_dir), nprocs=2, join=True)
        for rank in range(2):
            with open(os.path.join(results_dir, f"rank{rank}.json")) as f:
                out = json.load(f)
            self.assertEqual(out["float8_modules"], 4)  # up/down in two blocks
            self.assertTrue(out["has_float8_linear"])
            self.assertTrue(out["finite"])
            self.assertTrue(out["changed"])
            self.assertLess(out["losses"][-1], out["losses"][0])


class TestDisaggregatedLaunchForwardsFp8Linear(unittest.TestCase):
    """The disaggregated runtime builds its trainers through its own call sites
    (offline consumer and online consumer); ``training.fp8_linear`` must
    reach them exactly like the single-process path."""

    @staticmethod
    def _training():
        return {
            "strategy": "dflash",
            "role": "consumer",
            "max_steps": 1,
            "backend": "fsdp2",
            "fp8_linear": True,
            "static_shapes": True,
        }

    def test_online_consumer_receives_the_option(self):
        from specforge.algorithms.builtin import builtin_algorithm_registry
        from specforge.config import Config
        from specforge.training.disaggregated import _build_online

        cfg = Config.model_validate(
            {
                "model": {"target_model_path": "t", "draft_model_config": "d"},
                "data": {"prompts_path": "prompts.jsonl"},
                "training": self._training(),
                "deployment": {
                    "mode": "disaggregated",
                    "disaggregated": {
                        "control_dir": "/shared/fp8_linear",
                        "backend": "mooncake",
                        "server_urls": ["http://capture:30000"],
                    },
                },
            }
        )
        bundle = types.SimpleNamespace(model=object(), target_head=None, strategy_kwargs={})
        with (
            mock.patch.dict(os.environ, {"DISAGG_REF_CHANNEL": "/shared/refs"}),
            mock.patch("specforge.runtime.data_plane.streaming_ref_channel.StreamingRefChannel"),
            mock.patch("specforge.training.disaggregated._mooncake_store", return_value=mock.Mock()),
            mock.patch("specforge.launch.build_disagg_online_consumer", return_value=_FakeFitTrainer()) as build,
        ):
            _build_online(
                cfg,
                algorithm=builtin_algorithm_registry().resolve("dflash"),
                build_model_bundle=lambda _cfg: bundle,
                prepare_prompts=mock.Mock(),
                optimizer_factory=mock.Mock(),
                logger=None,
            )
        options = build.call_args.kwargs["backend_options"]
        self.assertIsInstance(options, BackendOptions)
        self.assertTrue(options.fp8_linear)

    def test_offline_consumer_receives_the_option(self):
        from specforge.algorithms.builtin import builtin_algorithm_registry
        from specforge.config import Config
        from specforge.training.disaggregated import _build_offline

        cfg = Config.model_validate(
            {
                "model": {"target_model_path": "t", "draft_model_config": "d"},
                "data": {"hidden_states_path": "features"},
                "training": self._training(),
                "deployment": {
                    "mode": "disaggregated",
                    "disaggregated": {"control_dir": "/shared/fp8_linear", "backend": "mooncake"},
                },
            }
        )
        bundle = types.SimpleNamespace(model=object(), target_head=None, strategy_kwargs={})
        with (
            mock.patch.dict(os.environ, {"DISAGG_MANIFEST": "/shared/manifest.json"}),
            mock.patch("specforge.training.disaggregated._wait_for"),
            mock.patch("specforge.training.disaggregated._offline_store", return_value=mock.Mock()),
            mock.patch("specforge.runtime.data_plane.disagg_ingest.read_ref_manifest", return_value=[]),
            mock.patch("specforge.launch.build_disagg_offline_runtime", return_value=_FakeFitTrainer()) as build,
        ):
            _build_offline(
                cfg,
                algorithm=builtin_algorithm_registry().resolve("dflash"),
                build_model_bundle=lambda _cfg: bundle,
                optimizer_factory=mock.Mock(),
                logger=None,
            )
        options = build.call_args.kwargs["backend_options"]
        self.assertIsInstance(options, BackendOptions)
        self.assertTrue(options.fp8_linear)


class _FakeFitTrainer:
    def fit(self):
        return 1


if __name__ == "__main__":
    unittest.main()
