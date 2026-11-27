# coding=utf-8
"""The job's one topology record: its sizes, and what ``init_distributed`` reads."""

import dataclasses
import json
import multiprocessing
import os
import socket
import tempfile
import time
import traceback
import unittest

import torch.distributed as dist

from specforge.distributed import Topology

WORLD_SIZE = 2


class TopologyTest(unittest.TestCase):
    def test_default_is_one_process_without_groups(self):
        topology = Topology()
        self.assertEqual(
            (topology.rank, topology.world_size, topology.local_rank), (0, 1, 0)
        )
        self.assertEqual(
            (topology.tp_rank, topology.dp_rank, topology.draft_dp_rank), (0, 0, 0)
        )
        self.assertEqual(topology.sp_rank, 0)
        self.assertEqual(topology.device_type, "cpu")
        self.assertEqual(
            (
                topology.sp_size,
                topology.dp_size,
                topology.draft_dp_size,
                topology.data_parallel_size,
            ),
            (1, 1, 1, 1),
        )
        for name in (
            "world_group",
            "tp_group",
            "dp_group",
            "draft_dp_group",
            "draft_sp_group",
            "sp_ulysses_group",
            "sp_ring_group",
        ):
            self.assertIsNone(getattr(topology, name), name)

    def test_derived_sizes(self):
        topology = Topology(world_size=16, tp_size=2, sp_ulysses_size=2, sp_ring_size=2)
        self.assertEqual(topology.sp_size, 4)
        self.assertEqual(topology.dp_size, 8)
        self.assertEqual(topology.draft_dp_size, 4)
        self.assertEqual(topology.data_parallel_size, 2)

    def test_impossible_shapes_are_refused(self):
        for shape, message in (
            ({"world_size": 0}, r"must be positive: \['world_size'\]"),
            ({"world_size": 2, "sp_ring_size": 0}, r"positive: \['sp_ring_size'\]"),
            ({"world_size": 2, "rank": 2}, "rank 2 is not a rank of a world of 2"),
            ({"rank": -1}, "rank -1 is not a rank"),
            # More model-parallel ranks than the world holds.
            ({"world_size": 2, "tp_size": 2, "sp_ulysses_size": 2}, "is not divided"),
            ({"world_size": 3, "sp_ring_size": 2}, "is not divided"),
        ):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, message):
                    Topology(**shape)

    def test_layout_records_the_sizes(self):
        self.assertEqual(
            Topology(rank=3, world_size=8, sp_ring_size=4).layout(),
            {"world_size": 8, "tp_size": 1, "sp_ulysses_size": 1, "sp_ring_size": 4},
        )

    def test_equality_ignores_groups(self):
        with_groups = Topology(world_size=2, world_group=object(), tp_group=object())
        self.assertEqual(with_groups, Topology(world_size=2))
        self.assertEqual(hash(with_groups), hash(Topology(world_size=2)))
        self.assertNotIn("group", repr(with_groups))
        self.assertNotEqual(Topology(rank=1, world_size=2), Topology(world_size=2))

    def test_is_frozen(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            Topology().rank = 1


def _init_worker(rank: int, port: int, results_dir: str) -> None:
    """Report what ``init_distributed`` and ``torch.distributed`` say on one rank."""
    payload = {}
    try:
        os.environ.update(
            {
                "RANK": str(rank),
                "WORLD_SIZE": str(WORLD_SIZE),
                "LOCAL_RANK": str(rank),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
                "SPECFORGE_DEVICE": "cpu",
            }
        )
        from specforge.distributed import destroy_distributed, init_distributed

        topology = init_distributed(timeout=1, tp_size=WORLD_SIZE)
        payload = {
            "topology": [
                topology.rank,
                topology.world_size,
                topology.tp_rank,
                topology.dp_rank,
                topology.draft_dp_rank,
                topology.sp_rank,
                topology.device_type,
            ],
            "torch": [
                dist.get_rank(),
                dist.get_world_size(),
                dist.get_rank(topology.tp_group),
                dist.get_rank(topology.dp_group),
                dist.get_rank(topology.draft_dp_group),
                dist.get_rank(topology.draft_sp_group),
                "cpu",
            ],
            "group_sizes": [
                dist.get_world_size(topology.tp_group),
                dist.get_world_size(topology.dp_group),
                dist.get_world_size(topology.draft_dp_group),
                dist.get_world_size(topology.draft_sp_group),
            ],
            "sizes": [topology.dp_size, topology.draft_dp_size, topology.sp_size],
            "local_rank": topology.local_rank,
            "world_group": topology.world_group is dist.group.WORLD,
        }
        destroy_distributed(topology)
        payload["initialized_after_destroy"] = dist.is_initialized()
    except BaseException as exc:
        payload = {
            "worker_error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    finally:
        with open(os.path.join(results_dir, f"rank{rank}.json"), "w") as stream:
            json.dump(payload, stream)


def _failed_init_worker(port, results_dir):
    """One rank asks for a tensor-parallel size its world cannot hold, then retries."""
    os.environ.update(
        RANK="0",
        WORLD_SIZE="1",
        LOCAL_RANK="0",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        SPECFORGE_DEVICE="cpu",
    )
    from specforge.distributed import destroy_distributed, init_distributed

    payload = {}
    try:
        try:
            init_distributed(tp_size=2)
            payload["raised"] = False
        except AssertionError:
            payload["raised"] = True
        payload["initialized_after_failure"] = dist.is_initialized()
        topology = init_distributed(tp_size=1)
        payload["retried_world_size"] = topology.world_size
        destroy_distributed(topology)
    except BaseException as exc:
        payload["worker_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        with open(os.path.join(results_dir, "failed.json"), "w") as stream:
            json.dump(payload, stream)


@unittest.skipUnless(
    dist.is_available() and dist.is_gloo_available(), "requires torch.distributed Gloo"
)
class InitDistributedTest(unittest.TestCase):
    def test_a_failed_init_leaves_no_default_group(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        with tempfile.TemporaryDirectory(prefix="topology_failed_") as results_dir:
            process = multiprocessing.get_context("spawn").Process(
                target=_failed_init_worker, args=(port, results_dir)
            )
            process.start()
            process.join(60)
            if process.is_alive():
                process.terminate()
                self.fail("the worker did not finish")
            with open(os.path.join(results_dir, "failed.json")) as stream:
                result = json.load(stream)
        self.assertNotIn("worker_error", result)
        self.assertTrue(result["raised"])
        self.assertFalse(result["initialized_after_failure"])
        self.assertEqual(result["retried_world_size"], 1)

    def test_two_gloo_ranks_read_what_torch_distributed_reports(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        with tempfile.TemporaryDirectory(prefix="topology_init_") as results_dir:
            context = multiprocessing.get_context("spawn")
            processes = [
                context.Process(target=_init_worker, args=(rank, port, results_dir))
                for rank in range(WORLD_SIZE)
            ]
            for process in processes:
                process.start()
            deadline = time.monotonic() + 60
            for process in processes:
                process.join(max(0.0, deadline - time.monotonic()))
            stuck = [
                rank for rank, process in enumerate(processes) if process.is_alive()
            ]
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(5)
            self.assertFalse(stuck, f"ranks {stuck} did not finish")
            results = []
            for rank in range(WORLD_SIZE):
                with open(os.path.join(results_dir, f"rank{rank}.json")) as stream:
                    results.append(json.load(stream))

        for rank, result in enumerate(results):
            with self.subTest(rank=rank):
                self.assertNotIn("worker_error", result, result.get("traceback"))
                self.assertEqual(result["topology"], result["torch"])
                # One (dp 1, tp 2) mesh and one (draft_dp 2, sp 1) mesh.
                self.assertEqual(result["topology"][2:6], [rank, 0, rank, 0])
                self.assertEqual(result["group_sizes"], [2, 1, 2, 1])
                self.assertEqual(result["sizes"], [1, 2, 1])
                # A CPU rank binds no accelerator.
                self.assertEqual(result["local_rank"], 0)
                self.assertTrue(result["world_group"])
                self.assertFalse(result["initialized_after_destroy"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
