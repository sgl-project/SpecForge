"""Rendezvous isolation for the local torchrun workers used by the recorder."""

import os
import unittest
from datetime import timedelta
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch.distributed as dist

from tests.test_runtime.trajectories import _worker_environment


def _join():
    return next(dist.rendezvous("env://", timeout=timedelta(seconds=5)))


class TrajectoryWorkerTest(unittest.TestCase):
    def test_overlapping_recordings_have_distinct_live_stores(self):
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            with _worker_environment(first, rank=0, world_size=1):
                first_port = os.environ["MASTER_PORT"]
                first_store, _, _ = _join()
                first_store.set("probe", "first")
                with _worker_environment(second, rank=0, world_size=1):
                    self.assertNotEqual(first_port, os.environ["MASTER_PORT"])
                    second_store, _, _ = _join()
                    second_store.set("probe", "second")
                    self.assertEqual(first_store.get("probe"), b"first")
                    self.assertEqual(second_store.get("probe"), b"second")

    def test_ranks_join_the_local_node_when_launched_from_another_node(self):
        with (
            TemporaryDirectory() as output,
            patch.dict(os.environ, {"NODE_RANK": "3"}),
            _worker_environment(output, rank=0, world_size=2),
        ):
            store, rank, size = _join()
            self.assertEqual((rank, size, os.environ["NODE_RANK"]), (0, 2, "0"))
            store.set("rank-zero", "ready")
            with _worker_environment(output, rank=1, world_size=2):
                peer, rank, size = _join()
                self.assertEqual((rank, size, os.environ["NODE_RANK"]), (1, 2, "0"))
                self.assertEqual(peer.get("rank-zero"), b"ready")
