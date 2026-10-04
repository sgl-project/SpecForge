"""``training.compile_blocks``: in-place block compilation ahead of FSDP2 sharding."""

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


class TestCompileBlocksSelection(unittest.TestCase):
    def test_fsdp1_rejects_compile_blocks(self):
        from tests.test_runtime.test_fsdp2_backend import TinyComposite

        pc = ParallelConfig(world_size=1)
        backend = create_training_backend(
            "fsdp", pc, options=BackendOptions(compile_blocks=True)
        )
        with self.assertRaisesRegex(ValueError, "compile_blocks"):
            backend.prepare_model(TinyComposite(), optimizer_target=None)

    def test_config_requires_fsdp2(self):
        from specforge.config.schema import TrainingConfig

        with self.assertRaisesRegex(ValueError, "compile_blocks"):
            TrainingConfig(compile_blocks=True)
        cfg = TrainingConfig(backend="fsdp2", compile_blocks=True)
        self.assertTrue(cfg.compile_blocks)

    def test_block_targets_fall_back_to_midlayer(self):
        from specforge.training.backend import DistributedTrainingBackend

        class Draft(nn.Module):
            def __init__(self):
                super().__init__()
                self.midlayer = nn.Linear(3, 3)

        class Composite(nn.Module):
            def __init__(self):
                super().__init__()
                self.draft_model = Draft()

        model = Composite()
        targets = DistributedTrainingBackend._block_targets(
            model, set(), model.draft_model
        )
        self.assertEqual(targets, [model.draft_model.midlayer])


def _worker(rank, world_size, port, results_dir):
    from tests.test_runtime import _fixtures as fx
    from tests.test_runtime.test_fsdp2_backend import TinyComposite, _optimizer

    fx.init_rank_distributed(rank, world_size, port=str(port))
    torch.manual_seed(0)
    reference = TinyComposite().cuda()
    compiled = TinyComposite().cuda()
    compiled.load_state_dict(reference.state_dict())
    x = torch.randn(4, 7, device="cuda")

    losses = {}
    for name, model, opts in (
        ("eager", reference, BackendOptions()),
        ("compiled", compiled, BackendOptions(compile_blocks=True)),
    ):
        # fp32 test models: match the backend's compute dtype like PR 915's gates.
        pc = ParallelConfig.from_distributed(param_dtype=torch.float32)
        backend = create_training_backend(
            "fsdp2", pc, optimizer_factory=_optimizer, options=opts
        )
        wrapped = backend.prepare_model(model, optimizer_target=model.draft_model)
        blocks = [m for m in wrapped.modules() if type(m).__name__.endswith("TinyBlock")]
        compiled_flags = [m._compiled_call_impl is not None for m in blocks]
        step_losses = []
        for _ in range(3):
            loss = wrapped(x).float().pow(2).mean()
            backend.backward(loss, is_boundary=True)
            backend.step()
            step_losses.append(float(loss.detach()))
        losses[name] = {"losses": step_losses, "compiled": compiled_flags}
    with open(os.path.join(results_dir, f"rank{rank}.json"), "w") as f:
        json.dump(losses, f)


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestCompileBlocksDistributed(unittest.TestCase):
    def test_compiled_blocks_match_eager(self):
        import torch.multiprocessing as mp

        results_dir = tempfile.mkdtemp(prefix="compile_blocks_")
        mp.spawn(_worker, args=(2, 29613, results_dir), nprocs=2, join=True)
        for rank in range(2):
            with open(os.path.join(results_dir, f"rank{rank}.json")) as f:
                out = json.load(f)
            self.assertTrue(all(out["compiled"]["compiled"]))
            self.assertFalse(any(out["eager"]["compiled"]))
            for a, b in zip(out["eager"]["losses"], out["compiled"]["losses"]):
                self.assertAlmostEqual(a, b, places=4)


class TestDisaggregatedLaunchForwardsCompileBlocks(unittest.TestCase):
    """The disaggregated runtime builds its trainers through its own call sites
    (offline consumer and online consumer); ``training.compile_blocks`` must
    reach them exactly like the single-process path."""

    @staticmethod
    def _training():
        return {
            "strategy": "dflash",
            "role": "consumer",
            "max_steps": 1,
            "backend": "fsdp2",
            "compile_blocks": True,
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
                        "control_dir": "/shared/compile_blocks",
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
        self.assertTrue(options.compile_blocks)

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
                    "disaggregated": {"control_dir": "/shared/compile_blocks", "backend": "mooncake"},
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
        self.assertTrue(options.compile_blocks)


class _FakeFitTrainer:
    def fit(self):
        return 1


if __name__ == "__main__":
    unittest.main()
