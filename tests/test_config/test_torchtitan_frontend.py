"""Native runtime launch/resume policy without importing the optional engine."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from specforge.training.torchtitan.frontend import (
    _online_training_horizon,
    _parallelism_layout,
    _resume_location,
    _training_horizon,
)
from tests.test_config.test_torchtitan_runtime import recipe


class TorchTitanFrontendTest(unittest.TestCase):
    def test_online_assembly_failure_notifies_waiting_producer(self):
        from specforge.training.assembly import build_training_run

        cfg = SimpleNamespace(
            mode="online",
            training=SimpleNamespace(
                backend="torchtitan", role="consumer", strategy="dflash"
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            channel = str(Path(directory) / "refs")
            with (
                mock.patch.dict("os.environ", {"DISAGG_REF_CHANNEL": channel}),
                mock.patch(
                    "specforge.training.torchtitan.frontend.build_torchtitan_training_run",
                    side_effect=RuntimeError("engine unavailable"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "engine unavailable"):
                    build_training_run(cfg, algorithm=SimpleNamespace(name="dflash"))
            self.assertIn(
                "engine unavailable", Path(channel + ".consumer_failed").read_text()
            )

    def test_online_schedule_does_not_stop_the_finite_stream(self):
        for schedule in (10, 1000):
            cfg = recipe(backend="torchtitan", total_steps=schedule)
            self.assertEqual(_online_training_horizon(cfg, 100), (100, schedule))
        cfg = recipe(backend="torchtitan", total_steps=1000, max_steps=20)
        self.assertEqual(_online_training_horizon(cfg, 100), (20, 1000))
        cfg = recipe(backend="torchtitan", max_steps=200)
        self.assertEqual(_online_training_horizon(cfg, 100), (100, 200))

    def test_tp_cp_pp_axes_leave_only_data_ranks_for_sampling(self):
        cfg = recipe(
            backend="torchtitan", tp_size=2, torchtitan={"cp_size": 2, "pp_size": 2}
        )
        layout, degree = _parallelism_layout(cfg, 16)
        cfg.validate_world_size(16)
        self.assertEqual(degree, 2)
        self.assertEqual(layout["data_parallel_shard_degree"], 2)

    def test_no_shard_uses_replicated_data_parallelism(self):
        cfg = recipe(
            backend="torchtitan",
            tp_size=2,
            fsdp_sharding="NO_SHARD",
            torchtitan={"dp_shard": 1},
        )
        cfg.validate_world_size(8)
        layout, degree = _parallelism_layout(cfg, 8)
        self.assertEqual(degree, 4)
        self.assertEqual(layout["data_parallel_replicate_degree"], 4)
        self.assertEqual(layout["data_parallel_shard_degree"], 1)

    def test_schedule_horizon_can_outlive_this_process(self):
        cfg = recipe(
            backend="torchtitan",
            batch_size=2,
            accumulation_steps=2,
            max_steps=2,
            total_steps=100,
        )
        self.assertEqual(_training_horizon(cfg, 32, 2), (2, 100))

    def test_explicit_checkpoint_and_file_uri_do_not_load_newer_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint with spaces and % sign"
            for step in (1, 9):
                path = root / f"step-{step}"
                path.mkdir(parents=True)
                (path / ".metadata").touch()
            source, step = _resume_location((root / "step-1").as_uri(), root)
            self.assertEqual(source, str((root / "step-1").resolve()))
            self.assertEqual(step, 1)
            self.assertEqual(_resume_location(str(root), root)[1], 9)

    def test_existing_outputs_require_explicit_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "step-2").mkdir()
            (root / "step-2/.metadata").touch()
            with self.assertRaisesRegex(ValueError, "explicit"):
                _resume_location(None, root)

    def test_external_checkpoint_cannot_be_shadowed_by_output_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ("source/step-1", "output/step-9"):
                path = root / directory
                path.mkdir(parents=True)
                (path / ".metadata").touch()
            with self.assertRaisesRegex(ValueError, "fresh output"):
                _resume_location(str(root / "source/step-1"), root / "output")


if __name__ == "__main__":
    unittest.main()
