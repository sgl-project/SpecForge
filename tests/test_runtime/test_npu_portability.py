# coding=utf-8
"""Static seams keeping the canonical trainer portable to Ascend NPU."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import specforge.distributed as sf_dist
from specforge.training.backend import FSDPTrainingBackend


class _Mesh:
    def __init__(self, name):
        self.name = name

    def get_group(self, dimension):
        return f"{self.name}:{dimension}"


class NPUDistributedTest(unittest.TestCase):
    _GLOBAL_NAMES = (
        "_DEVICE_MESH",
        "_TP_DEVICE_MESH",
        "_TP_GROUP",
        "_DP_DEVICE_MESH",
        "_DP_GROUP",
        "_DRAFT_DP_GROUP",
        "_DRAFT_SP_GROUP",
        "_SP_ULYSSES_GROUP",
        "_SP_RING_GROUP",
    )

    def setUp(self):
        # This suite can run after a CUDA runtime gate in the same interpreter.
        # Preserve its live groups while the NPU initializer is mocked.
        self._saved_globals = {
            name: getattr(sf_dist, name) for name in self._GLOBAL_NAMES
        }

    def tearDown(self):
        for name, value in self._saved_globals.items():
            setattr(sf_dist, name, value)

    def test_backend_mapping_is_device_specific(self):
        self.assertEqual(sf_dist._distributed_backend("cuda"), "nccl")
        self.assertEqual(sf_dist._distributed_backend("npu"), "hccl")
        self.assertEqual(sf_dist._distributed_backend("cpu"), "gloo")

    def test_pre_group_binding_falls_back_to_global_rank(self):
        accelerator = mock.Mock()
        accelerator.is_available.return_value = True
        accelerator.device_count.return_value = 4

        with (
            mock.patch.dict(os.environ, {"RANK": "5"}, clear=True),
            mock.patch.object(sf_dist, "_device_module", return_value=accelerator),
            mock.patch.object(sf_dist.dist, "is_initialized", return_value=False),
        ):
            local_rank = sf_dist._bind_local_device("npu")

        self.assertEqual(local_rank, 1)
        accelerator.set_device.assert_called_once_with(1)

    def test_npu_init_uses_hccl_binding_and_npu_meshes(self):
        accelerator = mock.Mock()
        accelerator.is_available.return_value = True
        accelerator.device_count.return_value = 8
        events = []
        accelerator.set_device.side_effect = lambda rank: events.append(("bind", rank))
        target_mesh = _Mesh("target")
        draft_mesh = _Mesh("draft")
        set_seq_parallel_pg = mock.Mock()
        process_group = SimpleNamespace(ULYSSES_PG="ulysses", RING_PG="ring")

        with (
            mock.patch.dict(os.environ, {"LOCAL_RANK": "3"}),
            mock.patch.object(sf_dist, "get_device_type", return_value="npu"),
            mock.patch.object(sf_dist, "_device_module", return_value=accelerator),
            mock.patch.object(
                sf_dist.dist,
                "init_process_group",
                side_effect=lambda **kwargs: events.append(("init", kwargs["backend"])),
            ) as init_pg,
            mock.patch.object(sf_dist.dist, "is_initialized", return_value=False),
            mock.patch.object(sf_dist.dist, "get_rank", return_value=3),
            mock.patch.object(sf_dist.dist, "get_world_size", return_value=8),
            mock.patch.object(
                sf_dist.dist.device_mesh,
                "init_device_mesh",
                side_effect=(target_mesh, draft_mesh),
            ) as init_mesh,
            mock.patch.object(
                sf_dist.dist.DeviceMesh,
                "from_group",
                side_effect=("tp-mesh", "dp-mesh"),
            ) as from_group,
            mock.patch.object(
                sf_dist,
                "_load_yunchang_globals",
                return_value=(process_group, set_seq_parallel_pg),
            ),
        ):
            sf_dist.init_distributed(
                timeout=7, tp_size=2, sp_ulysses_size=1, sp_ring_size=1
            )

        self.assertEqual(init_pg.call_args.kwargs["backend"], "hccl")
        accelerator.set_device.assert_called_once_with(3)
        self.assertEqual(events[:2], [("bind", 3), ("init", "hccl")])
        self.assertEqual(init_mesh.call_args_list[0].args[:2], ("npu", (4, 2)))
        self.assertEqual(init_mesh.call_args_list[1].args[:2], ("npu", (8, 1)))
        self.assertEqual(
            [call.kwargs["device_type"] for call in from_group.call_args_list],
            ["npu", "npu"],
        )
        set_seq_parallel_pg.assert_called_once_with(1, 1, 3, 8)


class DistributedTeardownTest(unittest.TestCase):
    def test_destroy_clears_every_cached_group_and_mesh(self):
        names = NPUDistributedTest._GLOBAL_NAMES
        saved = {name: getattr(sf_dist, name) for name in names}
        try:
            for name in names:
                setattr(sf_dist, name, object())
            with (
                mock.patch.object(sf_dist.dist, "is_initialized", return_value=True),
                mock.patch.object(sf_dist.dist, "destroy_process_group"),
            ):
                sf_dist.destroy_distributed()
            for name in names:
                self.assertIsNone(getattr(sf_dist, name), name)
        finally:
            for name, value in saved.items():
                setattr(sf_dist, name, value)


class NPURNGTest(unittest.TestCase):
    def test_checkpoint_round_trip_uses_bound_npu_rng(self):
        accelerator = mock.Mock()
        accelerator.is_available.return_value = True
        accelerator.current_device.return_value = 2
        accelerator_state = torch.tensor([4, 5], dtype=torch.uint8)
        accelerator.get_rng_state.return_value = accelerator_state
        cpu_state = torch.tensor([1, 2, 3], dtype=torch.uint8)

        with (
            mock.patch.object(torch, "npu", accelerator, create=True),
            mock.patch("specforge.utils.get_device_type", return_value="npu"),
            mock.patch.object(torch, "get_rng_state", return_value=cpu_state),
        ):
            state = FSDPTrainingBackend._rng_state()

        self.assertEqual(state["device_type"], "npu")
        self.assertIs(state["npu"], accelerator_state)
        self.assertIsNone(state["cuda"])
        accelerator.get_rng_state.assert_called_once_with(2)

        accelerator.reset_mock()
        accelerator.is_available.return_value = True
        accelerator.current_device.return_value = 2
        with (
            mock.patch.object(torch, "npu", accelerator, create=True),
            mock.patch.object(torch, "set_rng_state") as set_cpu_rng,
        ):
            FSDPTrainingBackend._set_rng_state(state)
        set_cpu_rng.assert_called_once_with(cpu_state)
        accelerator.set_rng_state.assert_called_once_with(accelerator_state, 2)

    def test_legacy_cuda_rng_checkpoint_remains_readable(self):
        accelerator = mock.Mock()
        accelerator.is_available.return_value = True
        accelerator.current_device.return_value = 1
        cpu_state = torch.tensor([1], dtype=torch.uint8)
        cuda_state = torch.tensor([2], dtype=torch.uint8)
        legacy = {"torch": cpu_state, "cuda": cuda_state}
        with (
            mock.patch.object(torch, "cuda", accelerator),
            mock.patch.object(torch, "set_rng_state"),
        ):
            FSDPTrainingBackend._set_rng_state(legacy)
        accelerator.set_rng_state.assert_called_once_with(cuda_state, 1)


class NPUTargetAssemblyTest(unittest.TestCase):
    def test_hf_target_uses_the_bound_local_device(self):
        from specforge.training.assembly import _build_target_engine
        from specforge.training.strategies.registry import resolve_strategy

        target = mock.Mock()
        cfg = SimpleNamespace(
            model=SimpleNamespace(
                target_model_path="target",
                target_backend="hf",
                input_modality="text",
                trust_remote_code=True,
                torch_dtype="bfloat16",
                cache_dir=None,
            ),
            training=SimpleNamespace(strategy="dflash"),
            data=SimpleNamespace(max_length=1024),
        )
        with (
            mock.patch(
                "specforge.inference.target_engine.get_target_engine",
                return_value=target,
            ) as get_engine,
            mock.patch("specforge.training.assembly._device", return_value="npu:3"),
        ):
            result = _build_target_engine(cfg, [1, 2], resolve_strategy("dflash"))

        self.assertIs(result, target)
        self.assertEqual(get_engine.call_args.kwargs["device"], "npu:3")
        target.set_capture_layers.assert_called_once_with([1, 2])


class NPUEvaluatorTest(unittest.TestCase):
    def test_hccl_collectives_use_the_bound_npu(self):
        from specforge.eval import Evaluator

        device = SimpleNamespace(type="npu")
        with (
            mock.patch("torch.distributed.get_backend", return_value="hccl"),
            mock.patch("specforge.utils.get_local_device", return_value=device),
        ):
            self.assertIs(Evaluator._comm_device(), device)


if __name__ == "__main__":
    unittest.main(verbosity=2)
