"""``training.checkpoint_async``: background checkpoint writes with deferred
completion (collective outcome check, latest pointer, rotation). CPU-only."""

import json
import os
import socket
import tempfile
import unittest
from datetime import timedelta

import torch

STATE_FILE = "training_state.pt"


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _mgr(out, **kw):
    from specforge.training.checkpoint import CheckpointManager

    return CheckpointManager(out, "run", async_write=True, **kw)


class TestAsyncSave(unittest.TestCase):
    def test_wait_completes_files_pointer_and_rotation(self):
        out = tempfile.mkdtemp(prefix="ckpt_async_")
        mgr = _mgr(out, max_checkpoints=1)
        d1 = mgr.save({"global_step": 1, "w": torch.ones(2)}, 1, rank_state={"rng": [1]})
        self.assertTrue(mgr.has_pending_save)
        mgr.wait()
        self.assertFalse(mgr.has_pending_save)
        self.assertTrue(os.path.isfile(os.path.join(d1, STATE_FILE)))
        self.assertTrue(os.path.isfile(os.path.join(d1, "training_state_rank0.pt")))
        self.assertEqual(os.path.realpath(mgr.latest_dir()), os.path.realpath(d1))
        # the next save finalizes the pending one before writing (rotation keeps 1)
        d2 = mgr.save({"global_step": 2, "w": torch.ones(2)}, 2, rank_state={"rng": [2]})
        self.assertTrue(mgr.has_pending_save)
        mgr.wait()
        self.assertEqual(os.path.realpath(mgr.latest_dir()), os.path.realpath(d2))
        self.assertFalse(os.path.isdir(d1), "rotation must drop the older checkpoint")
        self.assertEqual(mgr.wait(), None)  # idempotent

    def test_pointer_moves_only_after_completion(self):
        out = tempfile.mkdtemp(prefix="ckpt_async_ptr_")
        mgr = _mgr(out)
        mgr.save({"global_step": 1}, 1, rank_state={"rng": [1]})
        mgr.wait()
        first = os.path.realpath(mgr.latest_dir())
        mgr.save({"global_step": 2}, 2, rank_state={"rng": [2]})
        # before wait() the latest pointer still names the complete checkpoint
        self.assertEqual(os.path.realpath(mgr.latest_dir()), first)
        mgr.wait()
        self.assertEqual(os.path.realpath(mgr.latest_dir()), os.path.realpath(mgr.checkpoint_dir(2)))

    def test_write_failure_surfaces_at_wait(self):
        out = tempfile.mkdtemp(prefix="ckpt_async_fail_")
        mgr = _mgr(out)

        class Unpicklable:
            def __reduce__(self):
                raise RuntimeError("cannot pickle me")

        mgr.save({"global_step": 1, "bad": Unpicklable()}, 1, rank_state={"rng": [1]})
        with self.assertRaisesRegex(RuntimeError, "checkpoint save failed"):
            mgr.wait()
        self.assertIsNone(mgr.latest_dir())

    def test_staged_copy_moves_only_accelerator_tensors(self):
        from specforge.training.checkpoint import _staged_copy

        host = torch.zeros(3)
        staged = _staged_copy({"a": host, "b": [1, (host,)], "c": "x"})
        self.assertIs(staged["a"], host)  # host tensors are referenced
        self.assertIs(staged["b"][1][0], host)
        self.assertEqual(staged["c"], "x")

    def test_sync_default_unchanged(self):
        from specforge.training.checkpoint import CheckpointManager

        out = tempfile.mkdtemp(prefix="ckpt_sync_")
        mgr = CheckpointManager(out, "run")
        d = mgr.save({"global_step": 1}, 1, rank_state={"rng": [1]})
        self.assertFalse(mgr.has_pending_save)
        self.assertTrue(os.path.isfile(os.path.join(d, STATE_FILE)))


def _dist_worker(rank, world, port, out_dir, results_dir):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=60),
    )
    from specforge.training.checkpoint import CheckpointManager

    mgr = CheckpointManager(out_dir, "dist", async_write=True)
    mgr.save({"global_step": 3, "world_size": world}, 3, rank_state={"optimizer": {"rank": rank}, "rng": {"torch": [rank]}})
    mgr.save({"global_step": 4, "world_size": world}, 4, rank_state={"optimizer": {"rank": rank}, "rng": {"torch": [rank]}})
    mgr.wait()
    ckpt = mgr.latest_dir()
    st = CheckpointManager.read_resume_state(ckpt)
    with open(os.path.join(results_dir, f"rank{rank}.json"), "w") as fh:
        json.dump(
            {
                "ckpt": os.path.basename(ckpt),
                "files": sorted(os.listdir(ckpt)),
                "backend_rank": st["backend"]["optimizer"]["rank"],
                "global_step": st["global_step"],
                "steps_on_disk": sorted(s for s, _ in mgr._all_checkpoints()),
            },
            fh,
        )
    dist.destroy_process_group()


class TestDistributedAsyncSave(unittest.TestCase):
    def test_two_rank_async_saves_complete_in_order(self):
        import torch.multiprocessing as mp

        out = tempfile.mkdtemp(prefix="ckpt_async_dist_")
        results = tempfile.mkdtemp(prefix="ckpt_async_dist_res_")
        mp.spawn(_dist_worker, args=(2, _free_port(), out, results), nprocs=2, join=True)
        for r in range(2):
            with open(os.path.join(results, f"rank{r}.json")) as fh:
                res = json.load(fh)
            self.assertEqual(res["ckpt"], "dist-step4")
            self.assertEqual(
                res["files"],
                sorted([STATE_FILE, "training_state_rank0.pt", "training_state_rank1.pt"]),
            )
            self.assertEqual(res["backend_rank"], r)
            self.assertEqual(res["global_step"], 4)
            self.assertEqual(res["steps_on_disk"], [3, 4])


if __name__ == "__main__":
    unittest.main()
