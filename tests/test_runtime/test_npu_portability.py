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
        # Rank 3 of a (dp 4, tp 2) mesh and of a (draft_dp 8, sp 1) mesh.
        group_ranks = {
            None: 3,
            "target:tp": 1,
            "target:dp": 1,
            "draft:draft_dp": 3,
            "draft:sp": 0,
        }

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
            mock.patch.object(
                sf_dist.dist,
                "get_rank",
                side_effect=lambda group=None: group_ranks[group],
            ),
            mock.patch.object(sf_dist.dist, "get_world_size", return_value=8),
            mock.patch.object(sf_dist.dist, "group", SimpleNamespace(WORLD="world")),
            mock.patch.object(
                sf_dist.dist.device_mesh,
                "init_device_mesh",
                side_effect=(target_mesh, draft_mesh),
            ) as init_mesh,
            mock.patch.object(
                sf_dist,
                "_load_yunchang_globals",
                return_value=(process_group, set_seq_parallel_pg),
            ) as load_yunchang,
        ):
            topology = sf_dist.init_distributed(
                timeout=7, tp_size=2, sp_ulysses_size=1, sp_ring_size=1
            )

        self.assertEqual(init_pg.call_args.kwargs["backend"], "hccl")
        accelerator.set_device.assert_called_once_with(3)
        self.assertEqual(events[:2], [("bind", 3), ("init", "hccl")])
        self.assertEqual(init_mesh.call_args_list[0].args[:2], ("npu", (4, 2)))
        self.assertEqual(init_mesh.call_args_list[1].args[:2], ("npu", (8, 1)))
        self.assertEqual(
            topology,
            sf_dist.Topology(
                rank=3,
                world_size=8,
                local_rank=3,
                tp_size=2,
                tp_rank=1,
                dp_rank=1,
                draft_dp_rank=3,
                sp_rank=0,
                device_type="npu",
            ),
        )
        self.assertEqual(topology.world_group, "world")
        self.assertEqual(
            (topology.tp_group, topology.dp_group, topology.draft_dp_group),
            ("target:tp", "target:dp", "draft:draft_dp"),
        )
        # SP sizes of 1 must not touch yunchang: its import probes CUDA and
        # crashes NPU-only torch builds. The SP groups must still be the
        # singleton draft-SP group because PyTorch interprets a None group as
        # the full WORLD group.
        load_yunchang.assert_not_called()
        set_seq_parallel_pg.assert_not_called()
        self.assertEqual(topology.draft_sp_group, "draft:sp")
        self.assertEqual(topology.sp_ulysses_group, "draft:sp")
        self.assertEqual(topology.sp_ring_group, "draft:sp")


class DistributedTeardownTest(unittest.TestCase):
    def test_destroy_tears_down_every_group_the_topology_holds(self):
        groups = {
            name: object()
            for name in (
                "world_group",
                "tp_group",
                "dp_group",
                "draft_dp_group",
                "draft_sp_group",
                "sp_ulysses_group",
                "sp_ring_group",
            )
        }
        with mock.patch.object(sf_dist.dist, "destroy_process_group") as destroy:
            sf_dist.destroy_distributed(sf_dist.Topology(**groups))
        self.assertEqual(
            {id(call.args[0]) for call in destroy.call_args_list},
            {id(group) for group in groups.values()},
        )
        self.assertEqual(destroy.call_count, len(groups))

    def test_failure_teardown_aborts_each_distinct_process_group(self):
        subgroup = mock.Mock()
        world = mock.Mock()
        topology = sf_dist.Topology(
            world_group=world, tp_group=subgroup, dp_group=subgroup
        )
        with mock.patch.object(sf_dist.dist, "destroy_process_group") as destroy:
            sf_dist.destroy_distributed(topology, abort=True)

        subgroup.abort.assert_called_once_with()
        world.abort.assert_called_once_with()
        destroy.assert_not_called()


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

    def test_rng_checkpoint_requires_device_type(self):
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
            with self.assertRaisesRegex(ValueError, "does not record device_type"):
                FSDPTrainingBackend._set_rng_state(legacy)
        accelerator.set_rng_state.assert_not_called()


class NPUReductionTest(unittest.TestCase):
    def test_host_values_are_reduced_on_the_models_device(self):
        """An empty evaluation shard builds its sums on the host; HCCL needs the NPU."""
        from specforge.training.runner import DataParallelRunner
        from tests.test_runtime._fixtures import fake_spec

        device = SimpleNamespace(type="npu")
        program = SimpleNamespace(
            parameters=lambda: iter([SimpleNamespace(device=device)])
        )
        runner = DataParallelRunner(
            program, SimpleNamespace(process_group=None), fake_spec()
        )
        values = mock.Mock()
        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=2),
            mock.patch("torch.distributed.all_reduce") as reduce,
        ):
            reduced = runner.reduce_sum(values)
        values.to.assert_called_once_with(device)
        self.assertIs(reduced, values.to.return_value)
        self.assertIs(reduce.call_args.args[0], reduced)


if __name__ == "__main__":
    unittest.main(verbosity=2)
