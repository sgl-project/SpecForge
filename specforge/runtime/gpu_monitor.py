# coding=utf-8
"""Role-aware, CUDA-free GPU telemetry for long-running launchers."""

from __future__ import annotations

import importlib
import json
import math
import os
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TextIO,
)

SAMPLE_SCHEMA_VERSION = 1
_MAX_RECORDED_ERRORS = 128
_MAX_RECORDED_PIDS = 128
_MAX_RECORDED_VIOLATIONS = 128


class GpuMonitorError(RuntimeError):
    """GPU telemetry could not satisfy its runtime contract."""


class GpuOwnershipError(GpuMonitorError):
    """A reserved GPU was used outside its configured process role."""


@dataclass(frozen=True)
class GpuRoleAssignment:
    gpu: int
    logical_role: str
    process_role: str

    def __post_init__(self) -> None:
        if self.gpu < 0:
            raise ValueError("gpu index must be non-negative")
        if not self.logical_role or not self.process_role:
            raise ValueError("GPU role names must be non-empty")


@dataclass(frozen=True)
class GpuDevice:
    gpu: int
    uuid: str
    name: str


class GpuTelemetryBackend(Protocol):
    """Dependency-injection seam used by the real NVML and fake test backends."""

    def open(self, gpu_indices: Sequence[int]) -> Mapping[int, GpuDevice]: ...

    def sample(self, gpu: int) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


def _error_text(exc: BaseException) -> str:
    message = str(exc).replace("\n", " ").strip()
    return f"{type(exc).__name__}: {message}"[:512]


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _process_parent_pid(pid: int) -> int:
    """Read a process parent without adding a runtime dependency."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as stream:
            stat = stream.read()
    except FileNotFoundError as exc:
        raise ProcessLookupError(pid) from exc
    _, separator, suffix = stat.rpartition(")")
    fields = suffix.split()
    if not separator or len(fields) < 2:
        raise OSError(f"could not parse /proc/{pid}/stat")
    return int(fields[1])


class NvmlBackend:
    """Thin lazy wrapper around the official ``nvidia-ml-py`` bindings."""

    def __init__(self, module: Any = None) -> None:
        self._module = module
        self._handles: dict[int, Any] = {}
        self._opened = False

    def open(self, gpu_indices: Sequence[int]) -> Mapping[int, GpuDevice]:
        if self._opened:
            raise GpuMonitorError("NVML backend is already open")
        if self._module is None:
            try:
                self._module = importlib.import_module("pynvml")
            except ImportError as exc:
                raise GpuMonitorError(
                    "NVML monitoring requires nvidia-ml-py; install "
                    "SpecForge with the 'nvml' extra"
                ) from exc
        nvml = self._module
        try:
            nvml.nvmlInit()
            self._opened = True
            devices: dict[int, GpuDevice] = {}
            for gpu in gpu_indices:
                handle = nvml.nvmlDeviceGetHandleByIndex(gpu)
                self._handles[gpu] = handle
                devices[gpu] = GpuDevice(
                    gpu=gpu,
                    uuid=_as_text(nvml.nvmlDeviceGetUUID(handle)),
                    name=_as_text(nvml.nvmlDeviceGetName(handle)),
                )
            return devices
        except Exception:
            self.close()
            raise

    def _optional(
        self,
        errors: dict[str, str],
        field_name: str,
        callback: Callable[[], Any],
    ) -> Any:
        try:
            return callback()
        except Exception as exc:
            errors[field_name] = _error_text(exc)
            return None

    def sample(self, gpu: int) -> Mapping[str, Any]:
        if not self._opened or gpu not in self._handles:
            raise GpuMonitorError(f"NVML GPU {gpu} is not open")
        nvml = self._module
        handle = self._handles[gpu]
        errors: dict[str, str] = {}
        memory = self._optional(
            errors, "memory", lambda: nvml.nvmlDeviceGetMemoryInfo(handle)
        )
        utilization = self._optional(
            errors,
            "utilization",
            lambda: nvml.nvmlDeviceGetUtilizationRates(handle),
        )
        processes = self._optional(
            errors,
            "compute_processes",
            lambda: nvml.nvmlDeviceGetComputeRunningProcesses(handle),
        )

        normalized_processes = []
        unavailable = getattr(nvml, "NVML_VALUE_NOT_AVAILABLE", None)
        for process in processes or ():
            used_memory = getattr(process, "usedGpuMemory", None)
            if used_memory == unavailable or not isinstance(used_memory, int):
                used_memory = None
            normalized_processes.append(
                {"pid": int(process.pid), "used_memory_bytes": used_memory}
            )
        normalized_processes.sort(key=lambda value: value["pid"])

        temperature_kind = getattr(nvml, "NVML_TEMPERATURE_GPU", 0)
        clock_sm = getattr(nvml, "NVML_CLOCK_SM", 1)
        clock_memory = getattr(nvml, "NVML_CLOCK_MEM", 2)
        values = {
            "utilization_gpu_pct": (
                int(utilization.gpu) if utilization is not None else None
            ),
            "utilization_memory_pct": (
                int(utilization.memory) if utilization is not None else None
            ),
            "memory_used_bytes": int(memory.used) if memory is not None else None,
            "memory_free_bytes": int(memory.free) if memory is not None else None,
            "memory_total_bytes": int(memory.total) if memory is not None else None,
            "power_draw_mw": self._optional(
                errors, "power_draw_mw", lambda: nvml.nvmlDeviceGetPowerUsage(handle)
            ),
            "power_limit_mw": self._optional(
                errors,
                "power_limit_mw",
                lambda: nvml.nvmlDeviceGetEnforcedPowerLimit(handle),
            ),
            "temperature_c": self._optional(
                errors,
                "temperature_c",
                lambda: nvml.nvmlDeviceGetTemperature(handle, temperature_kind),
            ),
            "performance_state": self._optional(
                errors,
                "performance_state",
                lambda: nvml.nvmlDeviceGetPerformanceState(handle),
            ),
            "clock_sm_mhz": self._optional(
                errors,
                "clock_sm_mhz",
                lambda: nvml.nvmlDeviceGetClockInfo(handle, clock_sm),
            ),
            "clock_memory_mhz": self._optional(
                errors,
                "clock_memory_mhz",
                lambda: nvml.nvmlDeviceGetClockInfo(handle, clock_memory),
            ),
            "compute_processes": normalized_processes,
        }
        if errors:
            values["field_errors"] = errors
        return values

    def close(self) -> None:
        self._handles.clear()
        if not self._opened:
            return
        self._opened = False
        try:
            self._module.nvmlShutdown()
        except Exception:
            # Shutdown runs during launcher cleanup; collection errors are already
            # represented in the monitor artifact and must not mask child failure.
            pass


@dataclass
class _ScalarStats:
    count: int = 0
    total: float = 0.0
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    def add(self, value: Any) -> None:
        if value is None:
            return
        value = float(value)
        if not math.isfinite(value):
            return
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def summary(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.total / self.count if self.count else None,
            "min": self.minimum,
            "max": self.maximum,
        }


def _compute_process_memory_used_bytes(row: Mapping[str, Any]) -> Optional[int]:
    field_errors = row.get("field_errors") or {}
    if "compute_processes" in field_errors:
        return None
    processes = row.get("compute_processes")
    if processes is None:
        return None
    total = 0
    for process in processes:
        value = process.get("used_memory_bytes")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        total += value
    return total


def _add_histogram(histogram: list[int], value: Any) -> None:
    if value is None:
        return
    value = int(round(float(value)))
    if 0 <= value <= 100:
        histogram[value] += 1


def _histogram_percentile(histogram: Sequence[int], fraction: float) -> Optional[int]:
    total = sum(histogram)
    if total == 0:
        return None
    target = max(1, math.ceil(total * fraction))
    seen = 0
    for value, count in enumerate(histogram):
        seen += count
        if seen >= target:
            return value
    return len(histogram) - 1


@dataclass
class _GpuAggregate:
    sample_count: int = 0
    first_monotonic_ns: Optional[int] = None
    last_monotonic_ns: Optional[int] = None
    gpu_utilization_histogram: list[int] = field(default_factory=lambda: [0] * 101)
    memory_utilization_histogram: list[int] = field(default_factory=lambda: [0] * 101)
    memory_used: _ScalarStats = field(default_factory=_ScalarStats)
    compute_process_memory_used: _ScalarStats = field(default_factory=_ScalarStats)
    compute_process_memory_unavailable_samples: int = 0
    power_draw: _ScalarStats = field(default_factory=_ScalarStats)
    temperature: _ScalarStats = field(default_factory=_ScalarStats)
    clock_sm: _ScalarStats = field(default_factory=_ScalarStats)
    clock_memory: _ScalarStats = field(default_factory=_ScalarStats)
    performance_states: Counter[str] = field(default_factory=Counter)
    field_errors: Counter[str] = field(default_factory=Counter)
    observed_pids: set[int] = field(default_factory=set)
    observed_pids_truncated: bool = False
    max_compute_processes: int = 0
    low_utilization_samples: int = 0
    energy_joules: float = 0.0
    _previous_power_mw: Optional[float] = None
    _previous_power_timestamp_ns: Optional[int] = None

    def add(self, row: Mapping[str, Any], *, low_utilization_pct: int) -> None:
        timestamp_value = row.get("timestamp_monotonic_ns", row.get("timestamp_ns"))
        if isinstance(timestamp_value, bool) or not isinstance(timestamp_value, int):
            raise ValueError("GPU sample is missing an integer monotonic timestamp")
        timestamp_ns = timestamp_value
        self.sample_count += 1
        if self.first_monotonic_ns is None:
            self.first_monotonic_ns = timestamp_ns
        self.last_monotonic_ns = timestamp_ns

        gpu_utilization = row.get("utilization_gpu_pct")
        _add_histogram(self.gpu_utilization_histogram, gpu_utilization)
        _add_histogram(
            self.memory_utilization_histogram, row.get("utilization_memory_pct")
        )
        if gpu_utilization is not None and float(gpu_utilization) < low_utilization_pct:
            self.low_utilization_samples += 1
        self.memory_used.add(row.get("memory_used_bytes"))
        process_memory = _compute_process_memory_used_bytes(row)
        if process_memory is None:
            self.compute_process_memory_unavailable_samples += 1
        else:
            self.compute_process_memory_used.add(process_memory)
        self.power_draw.add(row.get("power_draw_mw"))
        self.temperature.add(row.get("temperature_c"))
        self.clock_sm.add(row.get("clock_sm_mhz"))
        self.clock_memory.add(row.get("clock_memory_mhz"))

        performance_state = row.get("performance_state")
        if performance_state is not None:
            self.performance_states[str(performance_state)] += 1
        for field_name in row.get("field_errors") or {}:
            self.field_errors[str(field_name)] += 1
        processes = row.get("compute_processes") or ()
        compute_pids = row.get("compute_pids")
        if compute_pids is None:
            compute_pids = [process["pid"] for process in processes]
        self.max_compute_processes = max(self.max_compute_processes, len(compute_pids))
        for value in compute_pids:
            pid = int(value)
            if len(self.observed_pids) < _MAX_RECORDED_PIDS:
                self.observed_pids.add(pid)
            elif pid not in self.observed_pids:
                self.observed_pids_truncated = True

        power_mw = row.get("power_draw_mw")
        if power_mw is None:
            self._previous_power_mw = None
            self._previous_power_timestamp_ns = None
            return
        power_mw = float(power_mw)
        if (
            self._previous_power_mw is not None
            and self._previous_power_timestamp_ns is not None
            and timestamp_ns > self._previous_power_timestamp_ns
        ):
            elapsed_s = (timestamp_ns - self._previous_power_timestamp_ns) / 1e9
            mean_watts = (self._previous_power_mw + power_mw) / 2000.0
            self.energy_joules += mean_watts * elapsed_s
        self._previous_power_mw = power_mw
        self._previous_power_timestamp_ns = timestamp_ns

    def summary(self) -> dict[str, Any]:
        gpu_count = sum(self.gpu_utilization_histogram)
        memory_count = sum(self.memory_utilization_histogram)
        coverage_s = None
        if self.first_monotonic_ns is not None and self.last_monotonic_ns is not None:
            coverage_s = (self.last_monotonic_ns - self.first_monotonic_ns) / 1e9
        return {
            "sample_count": self.sample_count,
            "first_monotonic_ns": self.first_monotonic_ns,
            "last_monotonic_ns": self.last_monotonic_ns,
            "coverage_s": coverage_s,
            "gpu_utilization_pct": {
                "count": gpu_count,
                "mean": (
                    sum(
                        value * count
                        for value, count in enumerate(self.gpu_utilization_histogram)
                    )
                    / gpu_count
                    if gpu_count
                    else None
                ),
                "p50": _histogram_percentile(self.gpu_utilization_histogram, 0.50),
                "p95": _histogram_percentile(self.gpu_utilization_histogram, 0.95),
                "low_fraction": (
                    self.low_utilization_samples / gpu_count if gpu_count else None
                ),
            },
            "memory_utilization_pct": {
                "count": memory_count,
                "mean": (
                    sum(
                        value * count
                        for value, count in enumerate(self.memory_utilization_histogram)
                    )
                    / memory_count
                    if memory_count
                    else None
                ),
                "p50": _histogram_percentile(self.memory_utilization_histogram, 0.50),
                "p95": _histogram_percentile(self.memory_utilization_histogram, 0.95),
            },
            # Keep memory_used_bytes as the legacy device-wide field.
            "memory_used_bytes": self.memory_used.summary(),
            "device_memory_used_bytes": self.memory_used.summary(),
            "compute_process_memory_used_bytes": (
                self.compute_process_memory_used.summary()
            ),
            "compute_process_memory_unavailable_samples": (
                self.compute_process_memory_unavailable_samples
            ),
            "power_draw_mw": self.power_draw.summary(),
            "energy_joules": self.energy_joules,
            "temperature_c": self.temperature.summary(),
            "clock_sm_mhz": self.clock_sm.summary(),
            "clock_memory_mhz": self.clock_memory.summary(),
            "performance_states": dict(sorted(self.performance_states.items())),
            "field_error_counts": dict(sorted(self.field_errors.items())),
            "observed_compute_pids": sorted(self.observed_pids),
            "observed_compute_pids_truncated": self.observed_pids_truncated,
            "max_compute_processes": self.max_compute_processes,
        }


def summarize_gpu_window(
    samples: Iterable[Mapping[str, Any]],
    assignments: Sequence[GpuRoleAssignment],
    *,
    start_monotonic_ns: int,
    end_monotonic_ns: int,
    interval_name: str = "measurement",
    low_utilization_pct: int = 10,
) -> dict[str, Any]:
    """Aggregate samples inside one explicit monotonic-clock interval."""

    if isinstance(start_monotonic_ns, bool) or not isinstance(start_monotonic_ns, int):
        raise TypeError("start_monotonic_ns must be an integer")
    if isinstance(end_monotonic_ns, bool) or not isinstance(end_monotonic_ns, int):
        raise TypeError("end_monotonic_ns must be an integer")
    if end_monotonic_ns <= start_monotonic_ns:
        raise ValueError("GPU measurement window must have positive duration")
    if not interval_name:
        raise ValueError("GPU measurement interval name must be non-empty")
    if not 0 <= low_utilization_pct <= 100:
        raise ValueError("low_utilization_pct must be in [0, 100]")

    assignments = tuple(assignments)
    if not assignments:
        raise ValueError("at least one GPU role assignment is required")
    assignment_by_gpu = {assignment.gpu: assignment for assignment in assignments}
    if len(assignment_by_gpu) != len(assignments):
        raise ValueError("GPU role assignments must use distinct GPU indices")
    aggregates = {gpu: _GpuAggregate() for gpu in assignment_by_gpu}

    for row in samples:
        timestamp_ns = row.get("timestamp_monotonic_ns", row.get("timestamp_ns"))
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int):
            raise ValueError("GPU sample is missing an integer monotonic timestamp")
        if not start_monotonic_ns <= timestamp_ns <= end_monotonic_ns:
            continue
        gpu = row.get("gpu")
        if isinstance(gpu, bool) or not isinstance(gpu, int):
            raise ValueError("GPU sample is missing an integer GPU index")
        aggregate = aggregates.get(gpu)
        if aggregate is None:
            continue
        aggregate.add(row, low_utilization_pct=low_utilization_pct)

    per_gpu = {
        str(gpu): {
            **asdict(assignment_by_gpu[gpu]),
            **aggregates[gpu].summary(),
        }
        for gpu in sorted(assignment_by_gpu)
    }
    invalid_reasons = []
    for gpu, aggregate in per_gpu.items():
        if aggregate["sample_count"] < 2:
            invalid_reasons.append(
                f"GPU {gpu} has {aggregate['sample_count']} sample(s); at least 2 are "
                "required"
            )
        elif aggregate["coverage_s"] is None or aggregate["coverage_s"] <= 0:
            invalid_reasons.append(f"GPU {gpu} has zero sample coverage")

    return {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "interval": interval_name,
        "start_monotonic_ns": start_monotonic_ns,
        "end_monotonic_ns": end_monotonic_ns,
        "duration_s": (end_monotonic_ns - start_monotonic_ns) / 1e9,
        "low_utilization_threshold_pct": low_utilization_pct,
        "valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "gpus": per_gpu,
    }


class GpuMonitor:
    """Continuously stream NVML samples and retain only bounded aggregates."""

    def __init__(
        self,
        assignments: Sequence[GpuRoleAssignment],
        sample_path: os.PathLike[str] | str,
        summary_path: os.PathLike[str] | str,
        *,
        poll_s: float = 1.0,
        backend: Optional[GpuTelemetryBackend] = None,
        max_compute_processes: Optional[int] = None,
        strict_process_ownership: bool = True,
        low_utilization_pct: int = 10,
        output: TextIO = sys.stderr,
        process_group_lookup: Callable[[int], int] = os.getpgid,
        process_parent_lookup: Callable[[int], int] = _process_parent_pid,
    ) -> None:
        assignments = tuple(assignments)
        if not assignments:
            raise ValueError("at least one GPU role assignment is required")
        if poll_s <= 0:
            raise ValueError("poll_s must be positive")
        if max_compute_processes is not None and max_compute_processes < 1:
            raise ValueError("max_compute_processes must be positive when set")
        if not 0 <= low_utilization_pct <= 100:
            raise ValueError("low_utilization_pct must be in [0, 100]")
        gpu_ids = [assignment.gpu for assignment in assignments]
        if len(gpu_ids) != len(set(gpu_ids)):
            raise ValueError("GPU role assignments must use distinct GPU indices")
        process_roles = [assignment.process_role for assignment in assignments]
        if len(process_roles) != len(set(process_roles)):
            raise ValueError("GPU process roles must be unique")

        self.assignments = assignments
        self.sample_path = os.fspath(sample_path)
        self.summary_path = os.fspath(summary_path)
        if os.path.realpath(self.sample_path) == os.path.realpath(self.summary_path):
            raise ValueError("sample and summary paths must differ")
        self.poll_s = poll_s
        self.backend = backend or NvmlBackend()
        self.max_compute_processes = max_compute_processes
        self.strict_process_ownership = strict_process_ownership
        self.low_utilization_pct = low_utilization_pct
        self.output = output
        self._process_group_lookup = process_group_lookup
        self._process_parent_lookup = process_parent_lookup
        self._assignment_by_role = {
            assignment.process_role: assignment for assignment in assignments
        }
        self._devices: dict[int, GpuDevice] = {}
        self._aggregates = {
            assignment.gpu: _GpuAggregate() for assignment in assignments
        }
        self._collection_duration_ms = _ScalarStats()
        self._poll_overrun_count = 0
        self._expected_process_groups: dict[str, int] = {}
        self._error_counts: dict[str, dict[str, Any]] = {}
        self._violations_by_key: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._collection_lock = threading.Lock()
        self._stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="specforge-gpu-monitor", daemon=True
        )
        self._handle: Optional[TextIO] = None
        self._backend_open = False
        self._started = False
        self._stopped = False
        self._started_at_unix_ns: Optional[int] = None
        self._started_at_monotonic_ns: Optional[int] = None
        self._ended_at_unix_ns: Optional[int] = None
        self._ended_at_monotonic_ns: Optional[int] = None
        self._write_failure: Optional[str] = None
        self._summary_write_failure: Optional[str] = None
        self._summary: Optional[dict[str, Any]] = None

    @property
    def violations(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(value) for value in self._violations_by_key.values()]

    @property
    def errors(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(value) for value in self._error_counts.values()]

    @property
    def summary(self) -> Optional[dict[str, Any]]:
        return self._summary

    def register_process_group(self, process_role: str, pgid: int) -> None:
        """Register a supervised process-group leader and its descendants."""
        if process_role not in self._assignment_by_role:
            raise ValueError(f"unknown monitored process role {process_role!r}")
        if pgid <= 0:
            raise ValueError("process group id must be positive")
        with self._lock:
            self._expected_process_groups[process_role] = int(pgid)

    def _is_process_tree_member(self, pid: int, root_pid: int) -> bool:
        current = pid
        visited: set[int] = set()
        while current > 1 and current not in visited:
            if current == root_pid:
                return True
            visited.add(current)
            try:
                current = self._process_parent_lookup(current)
            except ProcessLookupError:
                return False
            except OSError as exc:
                self._record_error(f"process-parent:pid={current}", exc)
                return False
        return current == root_pid

    def start(self) -> bool:
        if self._started:
            raise GpuMonitorError("GPU monitor can only be started once")
        self._started = True
        self._started_at_unix_ns = time.time_ns()
        self._started_at_monotonic_ns = time.monotonic_ns()
        os.makedirs(os.path.dirname(os.path.abspath(self.sample_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(self.summary_path)), exist_ok=True)
        self._handle = open(self.sample_path, "x", encoding="utf-8", buffering=1)
        try:
            devices = self.backend.open([value.gpu for value in self.assignments])
            self._devices = {int(gpu): device for gpu, device in devices.items()}
            self._backend_open = True
            missing = sorted(
                {value.gpu for value in self.assignments} - set(self._devices)
            )
            if missing:
                raise GpuMonitorError(f"NVML backend omitted GPUs {missing}")
        except Exception as exc:
            if not self._backend_open:
                try:
                    self.backend.close()
                except Exception as close_exc:
                    self._record_error("initialization-close", close_exc)
            return self._abort_start("initialization", exc)

        self._write_record(
            {
                "schema_version": SAMPLE_SCHEMA_VERSION,
                "record_type": "session_start",
                "timestamp_ns": self._started_at_monotonic_ns,
                "timestamp_unix_ns": self._started_at_unix_ns,
                "timestamp_monotonic_ns": self._started_at_monotonic_ns,
                "poll_s": self.poll_s,
                "assignments": [asdict(value) for value in self.assignments],
                "devices": [
                    asdict(self._devices[value.gpu]) for value in self.assignments
                ],
            }
        )
        if self._write_failure is not None:
            return self._abort_start(
                "session-start-write",
                GpuMonitorError(
                    f"failed to write session_start: {self._write_failure}"
                ),
            )
        try:
            self.thread.start()
        except Exception as exc:
            return self._abort_start("thread-start", exc)
        print(
            f"[gpu-monitor] started gpus={[value.gpu for value in self.assignments]} "
            f"samples={self.sample_path}",
            file=self.output,
            flush=True,
        )
        return True

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                cycle_started = time.monotonic()
                try:
                    self._collect_once()
                except Exception as exc:
                    with self._collection_lock:
                        if self._stop_event.is_set():
                            break
                        self._record_error("collection-loop", exc)
                        timestamp_ns = time.time_ns()
                        timestamp_monotonic_ns = time.monotonic_ns()
                        self._write_record(
                            {
                                "schema_version": SAMPLE_SCHEMA_VERSION,
                                "record_type": "error",
                                "timestamp_ns": timestamp_monotonic_ns,
                                "timestamp_unix_ns": timestamp_ns,
                                "timestamp_monotonic_ns": timestamp_monotonic_ns,
                                "stage": "collection-loop",
                                "error": _error_text(exc),
                            }
                        )
                elapsed = time.monotonic() - cycle_started
                with self._collection_lock:
                    if self._stop_event.is_set():
                        break
                    self._collection_duration_ms.add(elapsed * 1000.0)
                    if elapsed > self.poll_s:
                        self._poll_overrun_count += 1
                self._stop_event.wait(max(0.0, self.poll_s - elapsed))
        finally:
            self._close_backend()

    def _collect_once(self) -> None:
        timestamp_ns = time.time_ns()
        timestamp_monotonic_ns = time.monotonic_ns()
        for assignment in self.assignments:
            if self._stop_event.is_set():
                return
            try:
                metrics = dict(self.backend.sample(assignment.gpu))
            except Exception as exc:
                with self._collection_lock:
                    if self._stop_event.is_set():
                        return
                    self._record_error(f"sample:gpu={assignment.gpu}", exc)
                    self._write_record(
                        {
                            "schema_version": SAMPLE_SCHEMA_VERSION,
                            "record_type": "error",
                            "timestamp_ns": timestamp_monotonic_ns,
                            "timestamp_unix_ns": timestamp_ns,
                            "timestamp_monotonic_ns": timestamp_monotonic_ns,
                            "gpu": assignment.gpu,
                            "logical_role": assignment.logical_role,
                            "process_role": assignment.process_role,
                            "stage": "sample",
                            "error": _error_text(exc),
                        }
                    )
                continue
            processes = list(metrics.get("compute_processes") or ())
            compute_pids = sorted(int(value["pid"]) for value in processes)
            memory_used_bytes = metrics.get("memory_used_bytes")
            row = {
                "schema_version": SAMPLE_SCHEMA_VERSION,
                "record_type": "sample",
                "timestamp_ns": timestamp_monotonic_ns,
                "timestamp_unix_ns": timestamp_ns,
                "timestamp_monotonic_ns": timestamp_monotonic_ns,
                "gpu": assignment.gpu,
                "gpu_uuid": self._devices[assignment.gpu].uuid,
                "gpu_name": self._devices[assignment.gpu].name,
                "logical_role": assignment.logical_role,
                "process_role": assignment.process_role,
                **metrics,
                "memory_used_mib": (
                    memory_used_bytes / (1 << 20)
                    if memory_used_bytes is not None
                    else None
                ),
                "compute_pids": compute_pids,
            }
            with self._collection_lock:
                if self._stop_event.is_set():
                    return
                self._aggregates[assignment.gpu].add(
                    row, low_utilization_pct=self.low_utilization_pct
                )
                self._audit_processes(assignment, compute_pids, timestamp_monotonic_ns)
                self._write_record(row)

    def _audit_processes(
        self,
        assignment: GpuRoleAssignment,
        compute_pids: Sequence[int],
        timestamp_ns: int,
    ) -> None:
        if (
            self.max_compute_processes is not None
            and len(compute_pids) > self.max_compute_processes
        ):
            self._record_violation(
                {
                    "timestamp_ns": timestamp_ns,
                    "kind": "compute_process_count",
                    "gpu": assignment.gpu,
                    "process_role": assignment.process_role,
                    "compute_pids": list(compute_pids),
                    "maximum": self.max_compute_processes,
                }
            )
        if not self.strict_process_ownership:
            return
        with self._lock:
            expected_pgid = self._expected_process_groups.get(assignment.process_role)
        if expected_pgid is None:
            return
        unexpected = []
        for pid in compute_pids:
            try:
                observed_pgid = self._process_group_lookup(pid)
            except ProcessLookupError:
                continue
            except OSError as exc:
                self._record_error(f"process-group:pid={pid}", exc)
                continue
            if observed_pgid != expected_pgid and not self._is_process_tree_member(
                pid, expected_pgid
            ):
                unexpected.append({"pid": pid, "pgid": observed_pgid})
        if unexpected:
            self._record_violation(
                {
                    "timestamp_ns": timestamp_ns,
                    "kind": "unexpected_compute_process",
                    "gpu": assignment.gpu,
                    "process_role": assignment.process_role,
                    "expected_pgid": expected_pgid,
                    "unexpected": unexpected,
                }
            )

    def _record_violation(self, violation: Mapping[str, Any]) -> None:
        key_payload = {
            key: value for key, value in violation.items() if key != "timestamp_ns"
        }
        key = json.dumps(key_payload, sort_keys=True, separators=(",", ":"))
        with self._lock:
            if key in self._violations_by_key:
                self._violations_by_key[key]["count"] += 1
                return
            if len(self._violations_by_key) >= _MAX_RECORDED_VIOLATIONS - 1:
                key = "__truncated__"
                if key in self._violations_by_key:
                    self._violations_by_key[key]["count"] += 1
                    return
                self._violations_by_key[key] = {
                    "kind": "additional-violations-truncated",
                    "count": 1,
                }
                return
            self._violations_by_key[key] = {**violation, "count": 1}

    def _record_error(self, stage: str, exc: BaseException) -> None:
        error = _error_text(exc)
        key = f"{stage}:{error}"
        with self._lock:
            if key in self._error_counts:
                self._error_counts[key]["count"] += 1
                self._error_counts[key]["last_timestamp_ns"] = time.time_ns()
                return
            if len(self._error_counts) >= _MAX_RECORDED_ERRORS - 1:
                key = "__truncated__"
                if key in self._error_counts:
                    self._error_counts[key]["count"] += 1
                    self._error_counts[key]["last_timestamp_ns"] = time.time_ns()
                    return
                stage = "additional-errors-truncated"
                error = "additional distinct errors were omitted"
            timestamp_ns = time.time_ns()
            self._error_counts[key] = {
                "stage": stage,
                "error": error,
                "count": 1,
                "first_timestamp_ns": timestamp_ns,
                "last_timestamp_ns": timestamp_ns,
            }

    def _write_record(self, value: Mapping[str, Any]) -> None:
        if self._handle is None or self._write_failure is not None:
            return
        try:
            self._handle.write(
                json.dumps(
                    value, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
        except (OSError, TypeError, ValueError) as exc:
            self._write_failure = _error_text(exc)

    def _close_output(self) -> None:
        if self._handle is None:
            return
        try:
            self._handle.close()
        except OSError as exc:
            self._record_error("output-close", exc)
        finally:
            self._handle = None

    def _abort_start(self, stage: str, exc: BaseException) -> bool:
        self._record_error(stage, exc)
        timestamp_ns = time.time_ns()
        timestamp_monotonic_ns = time.monotonic_ns()
        self._write_record(
            {
                "schema_version": SAMPLE_SCHEMA_VERSION,
                "record_type": "error",
                "timestamp_ns": timestamp_monotonic_ns,
                "timestamp_unix_ns": timestamp_ns,
                "timestamp_monotonic_ns": timestamp_monotonic_ns,
                "stage": stage,
                "error": _error_text(exc),
            }
        )
        self._stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=max(10.0, self.poll_s * 2.0))
        if self.thread.is_alive():
            self._record_error(
                "startup-cleanup", GpuMonitorError("GPU monitor thread did not stop")
            )
        else:
            self._close_backend()
        self._close_output()
        print(
            f"[gpu-monitor] degraded: {_error_text(exc)}",
            file=self.output,
            flush=True,
        )
        return False

    def _close_backend(self) -> None:
        if not self._backend_open:
            return
        try:
            self.backend.close()
        except Exception as exc:
            self._record_error("shutdown", exc)
        finally:
            self._backend_open = False

    def raise_if_ownership_violated(self) -> None:
        violations = self.violations
        if violations:
            raise GpuOwnershipError(
                "GPU process ownership violation: "
                + json.dumps(violations[0], sort_keys=True, separators=(",", ":"))
            )

    def stop(self) -> dict[str, Any]:
        if not self._started:
            raise GpuMonitorError("GPU monitor was not started")
        if self._stopped:
            assert self._summary is not None
            return self._summary
        self._stopped = True
        self._stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=max(10.0, self.poll_s * 2.0))
        if self.thread.is_alive():
            self._record_error(
                "shutdown", GpuMonitorError("GPU monitor thread did not stop")
            )
        else:
            self._close_backend()
        with self._collection_lock:
            self._ended_at_unix_ns = time.time_ns()
            self._ended_at_monotonic_ns = time.monotonic_ns()
            self._write_record(
                {
                    "schema_version": SAMPLE_SCHEMA_VERSION,
                    "record_type": "session_end",
                    "timestamp_ns": self._ended_at_monotonic_ns,
                    "timestamp_unix_ns": self._ended_at_unix_ns,
                    "timestamp_monotonic_ns": self._ended_at_monotonic_ns,
                }
            )
            if self._handle is not None:
                try:
                    self._handle.flush()
                    os.fsync(self._handle.fileno())
                except OSError as exc:
                    self._write_failure = self._write_failure or _error_text(exc)
                finally:
                    self._close_output()
            self._summary = self._build_summary()
        try:
            _atomic_write_json(self.summary_path, self._summary)
        except (OSError, TypeError, ValueError) as exc:
            self._summary_write_failure = _error_text(exc)
            self._summary["summary_write_failure"] = self._summary_write_failure
            self._summary["status"] = "degraded"
            print(
                f"[gpu-monitor] summary write failed: {_error_text(exc)}",
                file=self.output,
                flush=True,
            )
        else:
            print(
                f"[gpu-monitor] stopped summary={self.summary_path} "
                f"status={self._summary['status']}",
                file=self.output,
                flush=True,
            )
        return self._summary

    def _build_summary(self) -> dict[str, Any]:
        violations = self.violations
        errors = self.errors
        field_errors = any(
            aggregate.field_errors for aggregate in self._aggregates.values()
        )
        if violations:
            status = "violated"
        elif errors or field_errors or self._write_failure:
            status = "degraded"
        elif not any(value.sample_count for value in self._aggregates.values()):
            status = "empty"
        else:
            status = "ok"
        per_gpu = {}
        for assignment in self.assignments:
            device = self._devices.get(
                assignment.gpu,
                GpuDevice(assignment.gpu, uuid="unknown", name="unknown"),
            )
            per_gpu[str(assignment.gpu)] = {
                **asdict(assignment),
                "gpu_uuid": device.uuid,
                "gpu_name": device.name,
                **self._aggregates[assignment.gpu].summary(),
            }
        return {
            "schema_version": SAMPLE_SCHEMA_VERSION,
            "status": status,
            "sample_path": os.path.abspath(self.sample_path),
            "started_at_unix_ns": self._started_at_unix_ns,
            "ended_at_unix_ns": self._ended_at_unix_ns,
            "duration_s": (
                (self._ended_at_monotonic_ns - self._started_at_monotonic_ns) / 1e9
                if self._ended_at_monotonic_ns is not None
                and self._started_at_monotonic_ns is not None
                else None
            ),
            "poll_s": self.poll_s,
            "collection_duration_ms": self._collection_duration_ms.summary(),
            "poll_overrun_count": self._poll_overrun_count,
            "low_utilization_threshold_pct": self.low_utilization_pct,
            "write_failure": self._write_failure,
            "summary_write_failure": self._summary_write_failure,
            "errors": errors,
            "ownership_violations": violations,
            "gpus": per_gpu,
        }


def _atomic_write_json(path: str, value: Mapping[str, Any]) -> None:
    target = os.path.abspath(path)
    temporary = f"{target}.tmp.{os.getpid()}.{threading.get_ident()}"
    try:
        with open(temporary, "x", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def iter_gpu_samples(path: os.PathLike[str] | str) -> Iterator[dict[str, Any]]:
    """Yield complete sample records, tolerating a torn final JSONL line."""

    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                if not line.endswith("\n"):
                    return
                raise GpuMonitorError(
                    f"invalid GPU sample JSONL at line {line_number}: {exc}"
                ) from exc
            if value.get("record_type") == "sample":
                yield value


__all__ = [
    "GpuDevice",
    "GpuMonitor",
    "GpuMonitorError",
    "GpuOwnershipError",
    "GpuRoleAssignment",
    "GpuTelemetryBackend",
    "NvmlBackend",
    "SAMPLE_SCHEMA_VERSION",
    "iter_gpu_samples",
]
