# coding=utf-8
"""Two-rank CPU protocol tests for the disaggregated online consumer.

These tests deliberately stop below model/FSDP construction.  They exercise the
distributed setup and acknowledgement collectives with the Gloo backend so the
failure protocol stays covered without GPUs or model fixtures.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sqlite3
import tempfile
import time
import traceback
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import torch.distributed as dist
except ModuleNotFoundError:  # Dependency-light local checks may not install torch.
    dist = None


WORLD_SIZE = 2
_GLOO_AVAILABLE = bool(
    dist is not None
    and dist.is_available()
    and getattr(dist, "is_gloo_available", lambda: False)()
)


class _PathOnlyChannel:
    """The setup-failure case only needs the channel's inbox path default."""

    def __init__(self, path: str) -> None:
        self.path = path


class _RetainingFeatureStore:
    retain_on_release = True

    def __init__(self) -> None:
        self.aborted = []

    def abort(self, sample_id, *, reason="aborted") -> None:
        self.aborted.append((sample_id, reason))


class _FakeRefDistributor:
    """Lifecycle probe used before any model/FSDP construction."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.started = False
        self.stopped = False
        self.kwargs = _kwargs

    @staticmethod
    def inbox_path(inbox_dir: str, dp_rank: int) -> str:
        return os.path.join(inbox_dir, f"inbox-rank{dp_rank}.jsonl")

    def start(self):
        self.started = True
        return self

    def stop(self) -> None:
        self.stopped = True


def _write_result(results_dir: str, rank: int, payload: dict) -> None:
    with open(os.path.join(results_dir, f"rank{rank}.json"), "w") as handle:
        json.dump(payload, handle)


def _init_gloo(rank: int, init_method: str) -> None:
    import torch.distributed as child_dist

    child_dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=timedelta(seconds=20),
    )


def _rank0_setup_error_worker(
    rank: int,
    init_method: str,
    work_dir: str,
    metadata_db_path: str,
    results_dir: str,
) -> None:
    """Enter the real builder and report the rank-0 setup error seen per rank."""
    payload = {}
    initialized = False
    try:
        _init_gloo(rank, init_method)
        initialized = True

        from specforge.launch import build_disagg_online_consumer

        try:
            build_disagg_online_consumer(
                feature_store=_RetainingFeatureStore(),
                channel=_PathOnlyChannel(os.path.join(work_dir, "refs.jsonl")),
                draft_model=None,
                optimizer_factory=None,
                run_id="setup-error-run",
                output_dir=work_dir,
                metadata_db_path=metadata_db_path,
                dp_rank=rank,
                dp_size=WORLD_SIZE,
                inbox_dir=os.path.join(work_dir, "inboxes"),
            )
        except BaseException as exc:
            payload = {"type": type(exc).__name__, "message": str(exc)}
        else:
            payload = {"type": None, "message": "builder unexpectedly succeeded"}
    except BaseException as exc:
        payload = {
            "worker_error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    finally:
        _write_result(results_dir, rank, payload)
        if initialized:
            import torch.distributed as child_dist

            child_dist.destroy_process_group()


def _dp_ack_worker(
    rank: int,
    init_method: str,
    metadata_db_path: str,
    results_dir: str,
) -> None:
    """Run one real DPAck all-gather and expose each rank's local marker."""
    payload = {}
    initialized = False
    store = None
    try:
        _init_gloo(rank, init_method)
        initialized = True

        from specforge.runtime.control_plane.dp_ack import DPAckController
        from specforge.runtime.control_plane.metadata_store import (
            InMemoryMetadataStore,
            SQLiteMetadataStore,
        )

        is_authority = rank == 0
        store = (
            SQLiteMetadataStore(metadata_db_path)
            if is_authority
            else InMemoryMetadataStore()
        )
        controller = DPAckController(
            "ack-run",
            is_authority=is_authority,
            metadata_store=store,
        )
        rank_ids = ["rank-0", "shared"] if rank == 0 else ["shared", "rank-1"]
        controller.ack_train_refs(
            f"trainer-{rank}",
            rank_ids,
            global_step=17 if rank == 0 else 999,
            optimizer_durable=rank == 0,
        )
        marker = store.durable_marker()
        payload = {
            "acked": sorted(marker["acked"]),
            "global_step": marker["global_step"],
            "optimizer_durable": marker["optimizer_durable"],
        }

        # Keep the process group alive until both ranks have returned from the
        # DPAck all_gather_object collective and observed their local marker.
        import torch.distributed as child_dist

        child_dist.barrier()
    except BaseException as exc:
        payload = {
            "worker_error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    finally:
        _write_result(results_dir, rank, payload)
        if store is not None and hasattr(store, "close"):
            store.close()
        if initialized:
            import torch.distributed as child_dist

            child_dist.destroy_process_group()


@unittest.skipUnless(_GLOO_AVAILABLE, "requires torch.distributed with Gloo")
class TestDisaggOnlineDPProtocol(unittest.TestCase):
    def _spawn_two_ranks(self, worker, *args, timeout_s: float = 40.0) -> list[dict]:
        context = multiprocessing.get_context("spawn")
        processes = [
            context.Process(target=worker, args=(rank, *args))
            for rank in range(WORLD_SIZE)
        ]
        for process in processes:
            process.start()

        deadline = time.monotonic() + timeout_s
        for process in processes:
            process.join(max(0.0, deadline - time.monotonic()))

        stuck = [rank for rank, process in enumerate(processes) if process.is_alive()]
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
        self.assertFalse(stuck, f"two-rank protocol deadlocked on ranks {stuck}")
        self.assertEqual(
            [process.exitcode for process in processes],
            [0] * WORLD_SIZE,
        )

        results_dir = args[-1]
        results = []
        for rank in range(WORLD_SIZE):
            path = os.path.join(results_dir, f"rank{rank}.json")
            self.assertTrue(os.path.exists(path), f"rank {rank} wrote no result")
            with open(path) as handle:
                result = json.load(handle)
            self.assertNotIn("worker_error", result, result.get("traceback"))
            results.append(result)
        return results

    @staticmethod
    def _init_method(work_dir: str) -> str:
        # FileStore requires a path that does not exist before rendezvous.
        return Path(os.path.join(work_dir, "gloo-init")).resolve().as_uri()

    def test_rank0_setup_error_is_broadcast_without_deadlock(self):
        with tempfile.TemporaryDirectory(prefix="disagg_setup_broadcast_") as work:
            metadata_db = os.path.join(work, "metadata.sqlite")
            # A non-empty committed table makes only rank 0 reject the ledger.
            # Rank 1 must receive that rejection through broadcast_object_list.
            with sqlite3.connect(metadata_db) as connection:
                connection.execute(
                    "CREATE TABLE committed "
                    "(sample_id TEXT PRIMARY KEY, ref_json TEXT NOT NULL)"
                )
                connection.execute(
                    "INSERT INTO committed (sample_id, ref_json) VALUES (?, ?)",
                    ("already-used", "{}"),
                )

            results = self._spawn_two_ranks(
                _rank0_setup_error_worker,
                self._init_method(work),
                work,
                metadata_db,
                work,
            )

            self.assertEqual(
                [result["type"] for result in results], ["RuntimeError"] * 2
            )
            self.assertEqual(results[0]["message"], results[1]["message"])
            self.assertIn("online consumer rank-0 setup failed", results[0]["message"])
            self.assertIn("already holds 1 committed samples", results[0]["message"])

    def test_dp_ack_gathers_union_and_only_rank0_persists(self):
        with tempfile.TemporaryDirectory(prefix="disagg_dp_ack_") as work:
            metadata_db = os.path.join(work, "metadata.sqlite")
            results = self._spawn_two_ranks(
                _dp_ack_worker,
                self._init_method(work),
                metadata_db,
                work,
            )

            self.assertEqual(results[0]["acked"], ["rank-0", "rank-1", "shared"])
            self.assertEqual(results[0]["global_step"], 17)
            self.assertTrue(results[0]["optimizer_durable"])
            self.assertEqual(
                results[1],
                {"acked": [], "global_step": None, "optimizer_durable": False},
            )

            # The shared durable ledger reflects rank 0's boundary metadata,
            # while its ack set is the real two-rank all-gather union.
            with sqlite3.connect(metadata_db) as connection:
                acked = sorted(
                    row[0]
                    for row in connection.execute(
                        "SELECT sample_id FROM acked ORDER BY sample_id"
                    )
                )
                marker = dict(connection.execute("SELECT k, v FROM marker").fetchall())
            self.assertEqual(acked, ["rank-0", "rank-1", "shared"])
            self.assertEqual(json.loads(marker["global_step"]), 17)
            self.assertTrue(json.loads(marker["optimizer_durable"]))


@unittest.skipUnless(dist is not None and dist.is_available(), "requires torch")
class TestSingleRankConsumerLifecycle(unittest.TestCase):
    def _build(self, work: str, *, assembly_error=None):
        from specforge.launch import build_disagg_online_consumer
        from specforge.runtime.control_plane.metadata_store import InMemoryMetadataStore
        from specforge.runtime.data_plane.streaming_ref_channel import (
            StreamingRefChannel,
        )

        captured = {}

        def assemble(**kwargs):
            captured.update(kwargs)
            if assembly_error is not None:
                raise assembly_error
            return SimpleNamespace()

        channel = StreamingRefChannel(os.path.join(work, "refs.jsonl"))
        with (
            mock.patch.object(dist, "is_initialized", return_value=False),
            mock.patch(
                "specforge.runtime.data_plane.ref_distributor.RefDistributor",
                _FakeRefDistributor,
            ),
            mock.patch("specforge.launch._assemble_trainer", side_effect=assemble),
        ):
            trainer = build_disagg_online_consumer(
                feature_store=_RetainingFeatureStore(),
                channel=channel,
                draft_model=object(),
                optimizer_factory=object(),
                run_id="consumer-lifecycle",
                output_dir=work,
                metadata_store=InMemoryMetadataStore(),
                inbox_dir=os.path.join(work, "inboxes"),
            )
        return trainer, channel, captured

    def test_direct_builder_success_signals_done_and_stops_distributor(self):
        with tempfile.TemporaryDirectory(prefix="consumer_lifecycle_") as work:
            trainer, channel, captured = self._build(work)
            self.assertTrue(trainer.ref_distributor.started)

            captured["on_fit_success"](3)
            captured["on_fit_finally"]()

            self.assertTrue(channel.consumer_stopped())
            self.assertIsNone(channel.consumer_failure())
            self.assertTrue(trainer.ref_distributor.stopped)

    def test_direct_builder_failure_signals_and_stops_distributor(self):
        with tempfile.TemporaryDirectory(prefix="consumer_lifecycle_") as work:
            trainer, channel, captured = self._build(work)

            captured["on_fit_failure"](RuntimeError("optimizer failed"))
            captured["on_fit_finally"]()

            self.assertEqual(
                channel.consumer_failure(), "RuntimeError: optimizer failed"
            )
            self.assertTrue(trainer.ref_distributor.stopped)

    def test_assembly_failure_notifies_producer_before_fit_exists(self):
        with tempfile.TemporaryDirectory(prefix="consumer_lifecycle_") as work:
            error = RuntimeError("assembly failed")
            with self.assertRaises(RuntimeError) as raised:
                self._build(work, assembly_error=error)
            self.assertIs(raised.exception, error)
            channel_path = os.path.join(work, "refs.jsonl")
            with open(channel_path + ".consumer_failed", encoding="utf-8") as stream:
                self.assertEqual(stream.read(), "RuntimeError: assembly failed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
