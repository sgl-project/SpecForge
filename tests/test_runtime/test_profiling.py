# coding=utf-8
"""Typed profiler-window lifecycle for the unified trainer."""

import tempfile
import unittest
from unittest import mock

from specforge.training.profiling import ProfilingOptions, StepProfiler


class _FakeProfiler:
    def __init__(self):
        self.started = 0
        self.stopped = 0
        self.exported = []

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def export_chrome_trace(self, path):
        self.exported.append(path)


class ProfilingTest(unittest.TestCase):
    def test_window_starts_before_requested_step_and_exports_after_length(self):
        fake = _FakeProfiler()
        with tempfile.TemporaryDirectory() as output_dir:
            profiler = StepProfiler(
                ProfilingOptions(
                    enabled=True,
                    start_step=2,
                    num_steps=3,
                    record_shapes=True,
                ),
                output_dir=output_dir,
            )
            with (
                mock.patch("torch.cuda.is_available", return_value=False),
                mock.patch("torch.profiler.profile", return_value=fake) as build,
            ):
                profiler.before_micro_step(1)
                profiler.before_micro_step(2)
                profiler.before_micro_step(2)
                profiler.after_optimizer_step(4)
                self.assertEqual(fake.stopped, 0)
                profiler.after_optimizer_step(5)

        self.assertEqual(fake.started, 1)
        self.assertEqual(fake.stopped, 1)
        self.assertEqual(len(fake.exported), 1)
        self.assertEqual(profiler.output_path, fake.exported[0])
        self.assertIn("profile_rank", profiler.output_path)
        self.assertTrue(build.call_args.kwargs["record_shapes"])

    def test_close_exports_a_partial_window_exactly_once(self):
        fake = _FakeProfiler()
        with tempfile.TemporaryDirectory() as output_dir:
            profiler = StepProfiler(
                ProfilingOptions(enabled=True, start_step=0, num_steps=4),
                output_dir=output_dir,
            )
            with (
                mock.patch("torch.cuda.is_available", return_value=False),
                mock.patch("torch.profiler.profile", return_value=fake),
            ):
                profiler.before_micro_step(0)
                profiler.close(1)
                profiler.close(1)

        self.assertEqual(fake.started, 1)
        self.assertEqual(fake.stopped, 1)
        self.assertEqual(len(fake.exported), 1)

    def test_resume_inside_window_profiles_the_remaining_steps(self):
        fake = _FakeProfiler()
        with tempfile.TemporaryDirectory() as output_dir:
            profiler = StepProfiler(
                ProfilingOptions(enabled=True, start_step=2, num_steps=4),
                output_dir=output_dir,
            )
            with (
                mock.patch("torch.cuda.is_available", return_value=False),
                mock.patch("torch.profiler.profile", return_value=fake),
            ):
                profiler.before_micro_step(4)
                profiler.after_optimizer_step(6)

        self.assertEqual(fake.started, 1)
        self.assertEqual(fake.stopped, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
