# coding=utf-8
"""CPU-only tests for role-aware NVML telemetry."""

from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from specforge.runtime.gpu_monitor import (
    GpuDevice,
    GpuMonitor,
    GpuMonitorError,
    GpuOwnershipError,
    GpuRoleAssignment,
    NvmlBackend,
    _GpuAggregate,
    iter_gpu_samples,
    summarize_gpu_window,
)


def _sample(*, pids=(123,), gpu_utilization=75):
    return {
        "utilization_gpu_pct": gpu_utilization,
        "utilization_memory_pct": 40,
        "memory_used_bytes": 8 << 30,
        "memory_free_bytes": 72 << 30,
        "memory_total_bytes": 80 << 30,
        "power_draw_mw": 350_000,
        "power_limit_mw": 700_000,
        "temperature_c": 61,
        "performance_state": 0,
        "clock_sm_mhz": 1_650,
        "clock_memory_mhz": 2_600,
        "compute_processes": [
            {"pid": pid, "used_memory_bytes": 4 << 30} for pid in pids
        ],
    }


class _FakeBackend:
    def __init__(self, values=None, *, open_error=None, sample_error=None):
        self.values = values or {3: _sample()}
        self.open_error = open_error
        self.sample_error = sample_error
        self.opened = False
        self.closed = False
        self.sample_calls = 0
        self.sampled = threading.Event()

    def open(self, gpu_indices):
        if self.open_error is not None:
            raise self.open_error
        self.opened = True
        return {
            gpu: GpuDevice(gpu=gpu, uuid=f"GPU-{gpu}", name="Fake H200")
            for gpu in gpu_indices
        }

    def sample(self, gpu):
        self.sample_calls += 1
        self.sampled.set()
        if self.sample_error is not None:
            raise self.sample_error
        return dict(self.values[gpu])

    def close(self):
        self.closed = True


class _BlockingBackend(_FakeBackend):
    def __init__(self):
        super().__init__()
        self.sample_entered = threading.Event()
        self.release_sample = threading.Event()

    def sample(self, gpu):
        self.sample_calls += 1
        self.sample_entered.set()
        self.release_sample.wait(timeout=5.0)
        return dict(self.values[gpu])


class GpuMonitorFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="gpu-monitor-")
        self.root = Path(self.temporary.name)
        self.assignment = GpuRoleAssignment(3, "producer", "target-server")

    def tearDown(self):
        self.temporary.cleanup()

    def monitor(self, backend, **kwargs):
        return GpuMonitor(
            [self.assignment],
            self.root / "samples.jsonl",
            self.root / "summary.json",
            poll_s=0.01,
            backend=backend,
            output=io.StringIO(),
            **kwargs,
        )


class TestGpuMonitor(GpuMonitorFixture):
    def test_streams_samples_and_finalizes_bounded_summary(self):
        backend = _FakeBackend()
        monitor = self.monitor(backend, max_compute_processes=1)

        self.assertTrue(monitor.start())
        self.assertTrue(backend.sampled.wait(timeout=1.0))
        deadline = time.monotonic() + 1.0
        while backend.sample_calls < 3 and time.monotonic() < deadline:
            time.sleep(0.005)
        summary = monitor.stop()

        samples = list(iter_gpu_samples(monitor.sample_path))
        self.assertGreaterEqual(len(samples), 1)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["gpus"]["3"]["sample_count"], len(samples))
        self.assertEqual(summary["gpus"]["3"]["logical_role"], "producer")
        self.assertEqual(summary["gpus"]["3"]["observed_compute_pids"], [123])
        self.assertEqual(
            summary["gpus"]["3"]["device_memory_used_bytes"]["max"], 8 << 30
        )
        self.assertEqual(
            summary["gpus"]["3"]["compute_process_memory_used_bytes"]["max"],
            4 << 30,
        )
        self.assertEqual(samples[0]["memory_used_mib"], 8192)
        self.assertEqual(samples[0]["compute_pids"], [123])
        self.assertEqual(
            samples[0]["timestamp_ns"], samples[0]["timestamp_monotonic_ns"]
        )
        self.assertGreaterEqual(summary["gpus"]["3"]["energy_joules"], 0.0)
        self.assertGreaterEqual(summary["collection_duration_ms"]["count"], 1)
        self.assertIsNotNone(summary["collection_duration_ms"]["max"])
        self.assertFalse(monitor.thread.is_alive())
        self.assertTrue(backend.closed)
        self.assertIsNone(monitor._handle)
        self.assertFalse(hasattr(monitor, "samples"))
        persisted = json.loads((self.root / "summary.json").read_text())
        self.assertEqual(persisted["gpus"]["3"]["sample_count"], len(samples))

    def test_session_start_write_failure_cleans_partial_start(self):
        backend = _FakeBackend()
        monitor = self.monitor(backend)
        handle = mock.Mock()
        handle.write.side_effect = OSError("disk full")

        with mock.patch("builtins.open", return_value=handle):
            self.assertFalse(monitor.start())

        self.assertTrue(backend.closed)
        self.assertFalse(monitor.thread.is_alive())
        self.assertIsNone(monitor._handle)
        handle.close.assert_called_once_with()
        summary = monitor.stop()
        self.assertEqual(summary["status"], "degraded")
        self.assertIn("disk full", summary["write_failure"])

    def test_thread_start_failure_cleans_partial_start(self):
        backend = _FakeBackend()
        monitor = self.monitor(backend)

        with mock.patch.object(
            monitor.thread, "start", side_effect=RuntimeError("thread unavailable")
        ):
            self.assertFalse(monitor.start())

        self.assertTrue(backend.closed)
        self.assertFalse(monitor.thread.is_alive())
        self.assertIsNone(monitor._handle)
        summary = monitor.stop()
        self.assertEqual(summary["status"], "degraded")
        self.assertTrue(
            any(value["stage"] == "thread-start" for value in summary["errors"])
        )

    def test_initialization_failure_is_visible_but_does_not_raise(self):
        backend = _FakeBackend(open_error=RuntimeError("NVML unavailable"))
        output = io.StringIO()
        monitor = GpuMonitor(
            [self.assignment],
            self.root / "samples.jsonl",
            self.root / "summary.json",
            poll_s=0.01,
            backend=backend,
            output=output,
        )

        self.assertFalse(monitor.start())
        summary = monitor.stop()

        self.assertEqual(summary["status"], "degraded")
        self.assertEqual(summary["gpus"]["3"]["sample_count"], 0)
        self.assertIn("NVML unavailable", output.getvalue())
        self.assertFalse(monitor.thread.is_alive())
        self.assertTrue(backend.closed)
        self.assertIsNone(monitor._handle)

    def test_sample_failure_is_counted_and_preserves_error_records(self):
        backend = _FakeBackend(sample_error=RuntimeError("driver timeout"))
        monitor = self.monitor(backend)
        monitor.start()
        self.assertTrue(backend.sampled.wait(timeout=1.0))
        summary = monitor.stop()

        self.assertEqual(summary["status"], "degraded")
        self.assertEqual(summary["gpus"]["3"]["sample_count"], 0)
        self.assertTrue(
            any("driver timeout" in value["error"] for value in summary["errors"])
        )
        records = [
            json.loads(line)
            for line in (self.root / "samples.jsonl").read_text().splitlines()
        ]
        self.assertTrue(any(value["record_type"] == "error" for value in records))

    def test_process_count_and_process_group_violations_are_deduplicated(self):
        backend = _FakeBackend(values={3: _sample(pids=(101, 202))})
        process_groups = {101: 7001, 202: 9009}
        monitor = GpuMonitor(
            [self.assignment],
            self.root / "samples.jsonl",
            self.root / "summary.json",
            poll_s=0.005,
            backend=backend,
            max_compute_processes=1,
            strict_process_ownership=True,
            output=io.StringIO(),
            process_group_lookup=process_groups.__getitem__,
        )
        monitor.register_process_group("target-server", 7001)
        monitor.start()
        deadline = time.monotonic() + 1.0
        while len(monitor.violations) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)

        with self.assertRaisesRegex(GpuOwnershipError, "ownership violation"):
            monitor.raise_if_ownership_violated()
        summary = monitor.stop()

        self.assertEqual(summary["status"], "violated")
        self.assertEqual(
            {value["kind"] for value in monitor.violations},
            {"compute_process_count", "unexpected_compute_process"},
        )
        self.assertEqual(len(monitor.violations), 2)

    def test_cross_process_group_descendant_is_owned(self):
        backend = _FakeBackend(values={3: _sample(pids=(202,))})
        process_groups = {202: 9009}
        process_parents = {202: 101, 101: 7001, 7001: 1}
        monitor = GpuMonitor(
            [self.assignment],
            self.root / "samples.jsonl",
            self.root / "summary.json",
            poll_s=0.005,
            backend=backend,
            strict_process_ownership=True,
            output=io.StringIO(),
            process_group_lookup=process_groups.__getitem__,
            process_parent_lookup=process_parents.__getitem__,
        )
        monitor.register_process_group("target-server", 7001)
        monitor.start()
        self.assertTrue(backend.sampled.wait(timeout=1.0))
        summary = monitor.stop()

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(monitor.violations, [])

    def test_stop_is_idempotent(self):
        backend = _FakeBackend()
        monitor = self.monitor(backend)
        monitor.start()
        backend.sampled.wait(timeout=1.0)
        first = monitor.stop()
        second = monitor.stop()
        self.assertIs(first, second)

    def test_optional_field_errors_mark_summary_degraded(self):
        value = _sample()
        value["field_errors"] = {"power_draw_mw": "not supported"}
        backend = _FakeBackend(values={3: value})
        monitor = self.monitor(backend)
        monitor.start()
        self.assertTrue(backend.sampled.wait(timeout=1.0))
        summary = monitor.stop()

        self.assertEqual(summary["status"], "degraded")
        self.assertGreaterEqual(
            summary["gpus"]["3"]["field_error_counts"]["power_draw_mw"], 1
        )

    def test_summary_write_failure_is_returned_to_the_caller(self):
        backend = _FakeBackend()
        monitor = self.monitor(backend)
        monitor.start()
        self.assertTrue(backend.sampled.wait(timeout=1.0))
        with mock.patch(
            "specforge.runtime.gpu_monitor._atomic_write_json",
            side_effect=OSError("disk full"),
        ):
            summary = monitor.stop()

        self.assertEqual(summary["status"], "degraded")
        self.assertIn("disk full", summary["summary_write_failure"])

    def test_sample_write_failure_does_not_disable_collection(self):
        monitor = self.monitor(_FakeBackend())
        handle = mock.Mock()
        handle.write.side_effect = OSError("disk full")
        monitor._handle = handle

        monitor._write_record({"record_type": "sample"})

        self.assertIn("disk full", monitor._write_failure)
        self.assertFalse(monitor._stop_event.is_set())

    def test_stop_timeout_finalizes_without_racing_blocked_sample(self):
        backend = _BlockingBackend()
        monitor = self.monitor(backend)
        self.assertTrue(monitor.start())
        self.assertTrue(backend.sample_entered.wait(timeout=1.0))

        with mock.patch.object(monitor.thread, "join", return_value=None):
            summary = monitor.stop()

        self.assertEqual(summary["status"], "degraded")
        self.assertEqual(summary["gpus"]["3"]["sample_count"], 0)
        self.assertTrue(
            any(value["stage"] == "shutdown" for value in summary["errors"])
        )
        backend.release_sample.set()
        monitor.thread.join(timeout=1.0)
        self.assertFalse(monitor.thread.is_alive())
        self.assertTrue(backend.closed)


class TestBoundedAggregation(unittest.TestCase):
    def test_one_hundred_thousand_samples_do_not_accumulate_rows(self):
        aggregate = _GpuAggregate()
        row = {
            "timestamp_monotonic_ns": 0,
            **_sample(),
        }
        for index in range(100_000):
            row["timestamp_monotonic_ns"] = index * 1_000_000_000
            aggregate.add(row, low_utilization_pct=10)

        self.assertEqual(aggregate.sample_count, 100_000)
        self.assertEqual(len(aggregate.gpu_utilization_histogram), 101)
        self.assertEqual(sum(aggregate.gpu_utilization_histogram), 100_000)
        self.assertEqual(aggregate.observed_pids, {123})
        self.assertEqual(aggregate.summary()["gpu_utilization_pct"]["p95"], 75)
        self.assertEqual(
            aggregate.summary()["compute_process_memory_used_bytes"]["count"],
            100_000,
        )

    def test_compute_process_memory_aggregation_semantics(self):
        aggregate = _GpuAggregate()
        rows = [
            {
                "timestamp_monotonic_ns": 1,
                **_sample(pids=(101, 202)),
            },
            {
                "timestamp_monotonic_ns": 2,
                **_sample(pids=()),
            },
            {
                "timestamp_monotonic_ns": 3,
                **_sample(pids=(303,)),
            },
            {
                "timestamp_monotonic_ns": 4,
                **_sample(pids=()),
                "field_errors": {"compute_processes": "NVML unavailable"},
            },
        ]
        rows[2]["compute_processes"][0]["used_memory_bytes"] = None
        for row in rows:
            aggregate.add(row, low_utilization_pct=10)

        summary = aggregate.summary()
        self.assertEqual(
            summary["memory_used_bytes"], summary["device_memory_used_bytes"]
        )
        self.assertEqual(
            summary["compute_process_memory_used_bytes"],
            {"count": 2, "mean": 4 << 30, "min": 0.0, "max": 8 << 30},
        )
        self.assertEqual(summary["compute_process_memory_unavailable_samples"], 2)

    def test_window_summary_excludes_startup_samples(self):
        samples = [
            {
                "timestamp_monotonic_ns": timestamp_ns,
                "gpu": 3,
                **_sample(gpu_utilization=utilization),
            }
            for timestamp_ns, utilization in (
                (1_000_000_000, 0),
                (2_000_000_000, 80),
                (3_000_000_000, 100),
            )
        ]

        summary = summarize_gpu_window(
            samples,
            [GpuRoleAssignment(3, "consumer:dflash-b4", "consumer:dflash-b4")],
            start_monotonic_ns=2_000_000_000,
            end_monotonic_ns=3_000_000_000,
            interval_name="steady",
        )

        gpu = summary["gpus"]["3"]
        self.assertEqual(summary["interval"], "steady")
        self.assertEqual(summary["duration_s"], 1.0)
        self.assertEqual(gpu["sample_count"], 2)
        self.assertEqual(gpu["first_monotonic_ns"], 2_000_000_000)
        self.assertEqual(gpu["last_monotonic_ns"], 3_000_000_000)
        self.assertEqual(gpu["gpu_utilization_pct"]["mean"], 90.0)
        self.assertEqual(gpu["gpu_utilization_pct"]["p50"], 80)
        self.assertEqual(gpu["gpu_utilization_pct"]["p95"], 100)
        self.assertEqual(gpu["max_compute_processes"], 1)
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["invalid_reasons"], [])

    def test_window_summary_rejects_one_sample_and_zero_coverage(self):
        assignment = GpuRoleAssignment(3, "producer", "target-server")
        sample = {
            "timestamp_monotonic_ns": 2_000_000_000,
            "gpu": 3,
            **_sample(),
        }

        one_sample = summarize_gpu_window(
            [sample],
            [assignment],
            start_monotonic_ns=1_000_000_000,
            end_monotonic_ns=3_000_000_000,
        )
        self.assertFalse(one_sample["valid"])
        self.assertIn("at least 2", one_sample["invalid_reasons"][0])

        zero_coverage = summarize_gpu_window(
            [sample, dict(sample)],
            [assignment],
            start_monotonic_ns=1_000_000_000,
            end_monotonic_ns=3_000_000_000,
        )
        self.assertFalse(zero_coverage["valid"])
        self.assertIn("zero sample coverage", zero_coverage["invalid_reasons"][0])

    def test_window_summary_rejects_empty_interval(self):
        with self.assertRaisesRegex(ValueError, "positive duration"):
            summarize_gpu_window(
                [],
                [GpuRoleAssignment(3, "producer", "target-server")],
                start_monotonic_ns=4,
                end_monotonic_ns=4,
            )

    def test_reader_ignores_non_samples_and_torn_final_line(self):
        path = Path(tempfile.mktemp(prefix="gpu-samples-", suffix=".jsonl"))
        self.addCleanup(path.unlink, missing_ok=True)
        path.write_text(
            '{"record_type":"session_start"}\n'
            '{"record_type":"sample","gpu":2}\n'
            '{"record_type":"sample"',
            encoding="utf-8",
        )
        self.assertEqual(
            list(iter_gpu_samples(path)), [{"record_type": "sample", "gpu": 2}]
        )

    def test_reader_rejects_a_corrupt_complete_line(self):
        path = Path(tempfile.mktemp(prefix="gpu-samples-", suffix=".jsonl"))
        self.addCleanup(path.unlink, missing_ok=True)
        path.write_text('{"record_type":]\n', encoding="utf-8")
        with self.assertRaisesRegex(GpuMonitorError, "line 1"):
            list(iter_gpu_samples(path))


class _FakeNvmlModule:
    NVML_TEMPERATURE_GPU = 0
    NVML_CLOCK_SM = 1
    NVML_CLOCK_MEM = 2
    NVML_VALUE_NOT_AVAILABLE = -1

    def __init__(self):
        self.initialized = False
        self.shutdown = False

    def nvmlInit(self):
        self.initialized = True

    def nvmlShutdown(self):
        self.shutdown = True

    def nvmlDeviceGetHandleByIndex(self, gpu):
        return f"handle-{gpu}"

    def nvmlDeviceGetUUID(self, handle):
        return b"GPU-real"

    def nvmlDeviceGetName(self, handle):
        return b"NVIDIA H200"

    def nvmlDeviceGetMemoryInfo(self, handle):
        return SimpleNamespace(used=10, free=20, total=30)

    def nvmlDeviceGetUtilizationRates(self, handle):
        return SimpleNamespace(gpu=90, memory=50)

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        return [SimpleNamespace(pid=321, usedGpuMemory=10)]

    def nvmlDeviceGetPowerUsage(self, handle):
        return 400_000

    def nvmlDeviceGetEnforcedPowerLimit(self, handle):
        return 700_000

    def nvmlDeviceGetTemperature(self, handle, kind):
        return 60

    def nvmlDeviceGetPerformanceState(self, handle):
        return 0

    def nvmlDeviceGetClockInfo(self, handle, kind):
        return 1_500 + kind


class TestNvmlBackend(unittest.TestCase):
    def test_normalizes_official_binding_values(self):
        module = _FakeNvmlModule()
        backend = NvmlBackend(module)
        devices = backend.open([2])
        sample = backend.sample(2)
        backend.close()

        self.assertEqual(devices[2], GpuDevice(2, "GPU-real", "NVIDIA H200"))
        self.assertEqual(sample["utilization_gpu_pct"], 90)
        self.assertEqual(sample["compute_processes"][0]["pid"], 321)
        self.assertTrue(module.initialized)
        self.assertTrue(module.shutdown)


@unittest.skipUnless(
    os.environ.get("SPECFORGE_RUN_NVML_TESTS") == "1",
    "set SPECFORGE_RUN_NVML_TESTS=1 for the real NVML smoke test",
)
class TestRealNvmlMonitor(unittest.TestCase):
    def test_samples_one_physical_gpu_without_creating_cuda_context(self):
        gpu = int(os.environ.get("SPECFORGE_NVML_GPU", "0"))
        with tempfile.TemporaryDirectory(prefix="gpu-monitor-real-") as directory:
            monitor = GpuMonitor(
                [GpuRoleAssignment(gpu, "smoke", "smoke")],
                Path(directory) / "samples.jsonl",
                Path(directory) / "summary.json",
                poll_s=0.05,
                output=io.StringIO(),
            )
            self.assertTrue(monitor.start())
            deadline = time.monotonic() + 2.0
            while len(list(iter_gpu_samples(monitor.sample_path))) < 2:
                if time.monotonic() >= deadline:
                    self.fail("NVML monitor did not produce two samples")
                time.sleep(0.01)
            summary = monitor.stop()

            self.assertIn(summary["status"], {"ok", "degraded"})
            self.assertNotEqual(summary["gpus"][str(gpu)]["gpu_uuid"], "unknown")
            samples = list(iter_gpu_samples(monitor.sample_path))
            gpu_summary = summary["gpus"][str(gpu)]
            self.assertEqual(
                gpu_summary["compute_process_memory_used_bytes"]["count"],
                len(samples),
            )
            self.assertEqual(
                gpu_summary["compute_process_memory_unavailable_samples"], 0
            )
            own_pid_rows = [
                value for value in samples if os.getpid() in value["compute_pids"]
            ]
            self.assertEqual(own_pid_rows, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
