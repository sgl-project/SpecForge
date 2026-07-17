# coding=utf-8
"""Input-pipeline timing aggregates and TrainBatch metadata propagation."""

import threading
import unittest
from types import SimpleNamespace

from specforge.runtime.input_pipeline import (
    InputPipelineRecorder,
    attach_input_pipeline_recorder,
    batch_input_pipeline_stage,
    input_pipeline_recorder,
)


class TestInputPipelineRecorder(unittest.TestCase):
    def test_stage_and_counter_snapshot_is_bounded_aggregate(self):
        ticks = iter((1.0, 1.25, 2.0, 2.5))
        recorder = InputPipelineRecorder(clock=lambda: next(ticks), emit_nvtx=False)

        with recorder.stage("fetch"):
            pass
        with recorder.stage("fetch"):
            pass
        recorder.increment("ready", 2)

        snapshot = recorder.snapshot()
        self.assertEqual(snapshot["counters"], {"ready": 2})
        self.assertEqual(
            snapshot["stages"]["fetch"],
            {
                "count": 2,
                "total_s": 0.75,
                "mean_s": 0.375,
                "max_s": 0.5,
            },
        )

    def test_concurrent_recording_is_lossless(self):
        recorder = InputPipelineRecorder(emit_nvtx=False)

        def record_many() -> None:
            for _ in range(100):
                recorder.record("fetch", 0.01)
                recorder.increment("samples")

        threads = [threading.Thread(target=record_many) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        snapshot = recorder.snapshot()
        self.assertEqual(snapshot["stages"]["fetch"]["count"], 400)
        self.assertEqual(snapshot["counters"]["samples"], 400)

    def test_batch_stage_uses_attached_recorder(self):
        recorder = InputPipelineRecorder(emit_nvtx=False)
        metadata = {}
        attach_input_pipeline_recorder(metadata, recorder)
        batch = SimpleNamespace(metadata=metadata)

        with batch_input_pipeline_stage(batch, "h2d"):
            pass

        self.assertIs(input_pipeline_recorder(metadata), recorder)
        self.assertEqual(recorder.snapshot()["stages"]["h2d"]["count"], 1)

    def test_invalid_measurements_are_rejected(self):
        recorder = InputPipelineRecorder(emit_nvtx=False)
        for duration in (-1.0, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                recorder.record("stage", duration)
        with self.assertRaises(ValueError):
            recorder.increment("counter", True)


if __name__ == "__main__":
    unittest.main()
