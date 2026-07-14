# coding=utf-8
"""CheckpointManager gates: atomic/complete-dir scanning, timeline rewind,
run-scoped latest pointers, rotation, read_resume_state, plus a 2-process gloo
pass over save + per-rank resume. CPU-only.
"""

import json
import os
import shutil
import socket
import tempfile
import unittest
from datetime import timedelta

import torch

LOGGER = "specforge.training.checkpoint"


def _mgr(out, run_id="run", **kw):
    from specforge.training.checkpoint import CheckpointManager

    return CheckpointManager(out, run_id, **kw)


def _state(step, **extra):
    return {"draft_state_dict": {"w": torch.zeros(1)}, "global_step": step, **extra}


def _steps(mgr):
    return sorted(step for step, _ in mgr._all_checkpoints())


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestLayoutAndAtomicity(unittest.TestCase):
    def test_incomplete_dir_is_invisible(self):
        from specforge.training.checkpoint import STATE_FILE

        out = tempfile.mkdtemp(prefix="ckpt_atomic_")
        mgr = _mgr(out)
        mgr.save(_state(1), 1)
        mgr.save(_state(2), 2)
        # a truncated save: step dir exists but STATE_FILE never landed
        os.makedirs(os.path.join(out, "run-step5"))
        self.assertEqual(_steps(mgr), [1, 2])
        os.remove(os.path.join(out, "run-latest"))
        self.assertEqual(
            os.path.realpath(mgr.latest_dir()), os.path.realpath(mgr.checkpoint_dir(2))
        )
        leftovers = [
            f for _, _, files in os.walk(out) for f in files if f.endswith(".tmp")
        ]
        self.assertEqual(leftovers, [], "atomic writes must not leave .tmp files")
        self.assertTrue(os.path.isfile(os.path.join(mgr.checkpoint_dir(2), STATE_FILE)))

    def test_latest_requires_symlink_and_shadow_is_repaired(self):
        from specforge.training.checkpoint import STATE_FILE

        out = tempfile.mkdtemp(prefix="ckpt_shadow_")
        mgr = _mgr(out)
        mgr.save(_state(1), 1)
        link = os.path.join(out, "run-latest")
        # materialized shadow: a real dir (with a plausible STATE_FILE) at the
        # link path, e.g. from a copier that dereferenced the symlink
        os.remove(link)
        os.makedirs(link)
        shutil.copy(os.path.join(mgr.checkpoint_dir(1), STATE_FILE), link)
        got = mgr.latest_dir()
        self.assertEqual(os.path.realpath(got), os.path.realpath(mgr.checkpoint_dir(1)))
        self.assertNotEqual(os.path.realpath(got), os.path.realpath(link))

        mgr.save(_state(2), 2)
        self.assertTrue(os.path.islink(link))
        self.assertEqual(
            os.path.realpath(link), os.path.realpath(mgr.checkpoint_dir(2))
        )

        os.remove(link)
        with open(link, "w") as fh:
            fh.write("x")
        self.assertEqual(
            os.path.realpath(mgr.latest_dir()),
            os.path.realpath(mgr.checkpoint_dir(2)),
        )
        mgr.save(_state(3), 3)
        self.assertTrue(os.path.islink(link))
        self.assertEqual(
            os.path.realpath(link), os.path.realpath(mgr.checkpoint_dir(3))
        )


class TestRewind(unittest.TestCase):
    def test_save_rewinds_future_steps(self):
        out = tempfile.mkdtemp(prefix="ckpt_rewind_")
        mgr = _mgr(out)
        for s in (1, 2, 3):
            mgr.save(_state(s), s)

        with self.assertLogs(LOGGER, level="WARNING"):
            mgr.save(_state(2), 2)

        self.assertFalse(os.path.exists(mgr.checkpoint_dir(3)))
        self.assertTrue(os.path.exists(mgr.checkpoint_dir(1)))
        self.assertEqual(_steps(mgr), [1, 2])
        self.assertEqual(
            os.path.realpath(mgr.latest_dir()), os.path.realpath(mgr.checkpoint_dir(2))
        )


class TestRunScoping(unittest.TestCase):
    def test_two_run_ids_do_not_interact(self):
        out = tempfile.mkdtemp(prefix="ckpt_runs_")
        a = _mgr(out, "alpha", max_checkpoints=2)
        b = _mgr(out, "beta")
        a.save(_state(1), 1)
        a.save(_state(2), 2)
        a.save(_state(3), 3)  # rotation drops alpha-step1 only
        b.save(_state(5), 5)

        self.assertEqual(_steps(a), [2, 3])
        self.assertEqual(_steps(b), [5])
        self.assertEqual(
            os.path.realpath(a.latest_dir()), os.path.realpath(a.checkpoint_dir(3))
        )
        self.assertEqual(
            os.path.realpath(b.latest_dir()), os.path.realpath(b.checkpoint_dir(5))
        )

        a.save(_state(2), 2)  # alpha rewind must not touch beta's step 5
        self.assertTrue(os.path.exists(b.checkpoint_dir(5)))
        self.assertEqual(_steps(b), [5])

    def test_glob_metachar_run_id(self):
        from specforge.training.checkpoint import STATE_FILE

        out = tempfile.mkdtemp(prefix="ckpt_glob_")
        # decoys that an UNescaped "run[ab]-step*" glob would match
        for decoy in ("runa-step9", "runb-step7"):
            os.makedirs(os.path.join(out, decoy))
            with open(os.path.join(out, decoy, STATE_FILE), "wb") as fh:
                fh.write(b"x")
        mgr = _mgr(out, "run[ab]")
        mgr.save(_state(1), 1)
        self.assertEqual(_steps(mgr), [1])
        os.remove(os.path.join(out, "run[ab]-latest"))
        self.assertEqual(
            os.path.realpath(mgr.latest_dir()), os.path.realpath(mgr.checkpoint_dir(1))
        )
        mgr.save(_state(0), 0)  # rewind deletes run[ab]-step1, not the decoys
        self.assertFalse(os.path.exists(mgr.checkpoint_dir(1)))
        self.assertTrue(os.path.exists(os.path.join(out, "runa-step9")))
        self.assertTrue(os.path.exists(os.path.join(out, "runb-step7")))


class TestReadResumeState(unittest.TestCase):
    def test_backend_passthrough_and_path_forms(self):
        from specforge.training.checkpoint import STATE_FILE, CheckpointManager

        out = tempfile.mkdtemp(prefix="ckpt_read_")
        rng = torch.get_rng_state()
        rank_state = {"optimizer": {"lr": 0.5}, "rng": {"torch": rng}, "custom": 7}
        ckpt = _mgr(out).save(_state(3, world_size=1), 3, rank_state=rank_state)

        for target in (ckpt, os.path.join(ckpt, STATE_FILE), f"file://{ckpt}"):
            st = CheckpointManager.read_resume_state(target)
            self.assertEqual(st["global_step"], 3)
            self.assertEqual(st["backend"]["optimizer"], {"lr": 0.5})
            self.assertEqual(st["backend"]["custom"], 7)
            self.assertTrue(torch.equal(st["backend"]["rng"]["torch"], rng))
            self.assertNotIn("optimizer_state_dict", st)
            self.assertNotIn("rng_state", st)

    def test_require_full_state_raises_without_rank_file(self):
        from specforge.training.checkpoint import CheckpointManager

        out = tempfile.mkdtemp(prefix="ckpt_norank_")
        ckpt = _mgr(out).save(_state(5, world_size=1), 5)
        with self.assertRaisesRegex(ValueError, "per-rank state"):
            CheckpointManager.read_resume_state(ckpt)
        st = CheckpointManager.read_resume_state(ckpt, require_full_state=False)
        self.assertEqual(st["backend"], {})
        self.assertEqual(st["global_step"], 5)

    def test_step0_weights_only_is_fine(self):
        from specforge.training.checkpoint import CheckpointManager

        out = tempfile.mkdtemp(prefix="ckpt_step0_")
        ckpt = _mgr(out).save(_state(0, world_size=1), 0)
        st = CheckpointManager.read_resume_state(ckpt)  # require_full_state default
        self.assertEqual(st["backend"], {})

    def test_world_size_mismatch(self):
        from specforge.training.checkpoint import CheckpointManager

        out = tempfile.mkdtemp(prefix="ckpt_ws_")
        ckpt = _mgr(out).save(
            _state(4, world_size=4), 4, rank_state={"optimizer": {}, "rng": {}}
        )
        with self.assertRaisesRegex(ValueError, r"written at world_size=4"):
            CheckpointManager.read_resume_state(ckpt)
        st = CheckpointManager.read_resume_state(ckpt, require_full_state=False)
        self.assertEqual(st["backend"], {})


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

    mgr = CheckpointManager(out_dir, "dist")
    ckpt = mgr.save(
        {"global_step": 3, "world_size": world},
        3,
        rank_state={"optimizer": {"rank": rank}, "rng": {"torch": [rank]}},
    )
    st = CheckpointManager.read_resume_state(ckpt)
    with open(os.path.join(results_dir, f"rank{rank}.json"), "w") as fh:
        json.dump(
            {
                "ckpt": ckpt,
                "files": sorted(os.listdir(ckpt)),
                "backend_rank": st["backend"]["optimizer"]["rank"],
                "backend_rng": st["backend"]["rng"]["torch"],
                "global_step": st["global_step"],
            },
            fh,
        )
    dist.destroy_process_group()


class TestDistributedSaveAndResume(unittest.TestCase):
    def test_two_rank_save_and_per_rank_resume(self):
        import torch.multiprocessing as mp

        from specforge.training.checkpoint import STATE_FILE

        out = tempfile.mkdtemp(prefix="ckpt_dist_")
        results = tempfile.mkdtemp(prefix="ckpt_dist_res_")
        mp.spawn(
            _dist_worker, args=(2, _free_port(), out, results), nprocs=2, join=True
        )

        loaded = []
        for r in range(2):
            with open(os.path.join(results, f"rank{r}.json")) as fh:
                loaded.append(json.load(fh))
        self.assertEqual(loaded[0]["ckpt"], loaded[1]["ckpt"])
        expected = sorted(
            [STATE_FILE, "training_state_rank0.pt", "training_state_rank1.pt"]
        )
        for r, res in enumerate(loaded):
            self.assertEqual(res["files"], expected)
            self.assertEqual(res["backend_rank"], r)  # each rank got its own shard
            self.assertEqual(res["backend_rng"], [r])
            self.assertEqual(res["global_step"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
