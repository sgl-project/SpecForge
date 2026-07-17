# coding=utf-8
"""Process-group and exclusive-GPU lifecycle helpers for local launchers."""

from __future__ import annotations

import csv
import fcntl
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, TextIO


class LauncherError(RuntimeError):
    """A managed role or launch resource failed."""

    def __init__(
        self,
        message: str,
        *,
        role: Optional[str] = None,
        log_path: Optional[str] = None,
        returncode: int = 1,
    ) -> None:
        super().__init__(message)
        self.role = role
        self.log_path = log_path
        self.returncode = returncode


@dataclass(frozen=True)
class RoleCommand:
    role: str
    argv: tuple[str, ...]
    env: Mapping[str, str]
    log_path: str
    persistent: bool = False


_LOCAL_LOCK_FDS: set[int] = set()


class GPUReservationSet:
    """Hold cooperative per-GPU locks for a complete launcher lifetime."""

    def __init__(self, gpu_ids: Sequence[int], *, lock_path_pattern: str) -> None:
        self.gpu_ids = tuple(sorted(set(int(gpu_id) for gpu_id in gpu_ids)))
        if not self.gpu_ids:
            raise ValueError("at least one GPU must be reserved")
        if "{gpu_id}" not in lock_path_pattern:
            raise ValueError("lock_path_pattern must contain {gpu_id}")
        self.lock_path_pattern = lock_path_pattern
        self._locks: dict[int, tuple[int, bool]] = {}

    @staticmethod
    def _inherited_fd(path: str) -> Optional[int]:
        expected = os.path.realpath(path)
        try:
            entries = os.listdir("/proc/self/fd")
        except OSError:
            return None
        for entry in entries:
            if not entry.isdigit() or int(entry) <= 2:
                continue
            fd = int(entry)
            if fd in _LOCAL_LOCK_FDS:
                continue
            try:
                if os.path.realpath(f"/proc/self/fd/{fd}") == expected:
                    return fd
            except OSError:
                continue
        return None

    def acquire(self) -> None:
        if self._locks:
            raise RuntimeError("GPU reservations are already acquired")
        completed = False
        try:
            for gpu_id in self.gpu_ids:
                path = self.lock_path_pattern.format(gpu_id=gpu_id)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                fd = self._inherited_fd(path)
                owned = fd is None
                if fd is None:
                    # Shared lock files commonly end up 0644 after umask. flock()
                    # does not need a writable descriptor, so opening read-only
                    # lets another user reserve the GPU after the first user exits.
                    fd = os.open(
                        path,
                        os.O_RDONLY | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
                        0o666,
                    )
                    _LOCAL_LOCK_FDS.add(fd)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    if owned:
                        _LOCAL_LOCK_FDS.discard(fd)
                        os.close(fd)
                    raise LauncherError(
                        f"GPU {gpu_id} is reserved by another launcher via {path}"
                    ) from exc
                self._locks[gpu_id] = (fd, owned)
            completed = True
        finally:
            if not completed:
                self.close()

    def close(self) -> None:
        for gpu_id in reversed(tuple(self._locks)):
            fd, owned = self._locks.pop(gpu_id)
            if not owned:
                continue
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                _LOCAL_LOCK_FDS.discard(fd)
                os.close(fd)


def _query_nvidia_smi(
    executable: str,
    query: str,
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> str:
    try:
        result = run(
            [
                executable,
                f"--query-{query}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise LauncherError(f"nvidia-smi query failed for {query}: {exc}") from exc
    return result.stdout


def gpu_busy_reasons(
    executable: str,
    gpu_ids: Sequence[int],
    *,
    max_used_memory_mib: int,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> tuple[str, ...]:
    requested = set(gpu_ids)
    inventory: dict[int, tuple[str, int]] = {}
    raw_inventory = _query_nvidia_smi(executable, "gpu=index,uuid,memory.used", run=run)
    for row in csv.reader(raw_inventory.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 3:
            raise LauncherError(f"unexpected nvidia-smi GPU row: {row!r}")
        try:
            inventory[int(row[0].strip())] = (
                row[1].strip(),
                int(row[2].strip()),
            )
        except ValueError as exc:
            raise LauncherError(f"invalid nvidia-smi GPU row: {row!r}") from exc
    missing = sorted(requested - set(inventory))
    if missing:
        raise LauncherError(f"requested GPUs are absent from nvidia-smi: {missing}")

    processes: dict[str, list[str]] = {}
    raw_processes = _query_nvidia_smi(
        executable,
        "compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        run=run,
    )
    for row in csv.reader(raw_processes.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 4:
            raise LauncherError(f"unexpected nvidia-smi process row: {row!r}")
        processes.setdefault(row[0].strip(), []).append(
            f"pid={row[1].strip()} process={row[2].strip()} memory_mib={row[3].strip()}"
        )

    reasons: list[str] = []
    for gpu_id in sorted(requested):
        gpu_uuid, used_memory = inventory[gpu_id]
        if gpu_uuid in processes:
            reasons.append(f"GPU {gpu_id} has compute processes: {processes[gpu_uuid]}")
        if used_memory > max_used_memory_mib:
            reasons.append(
                f"GPU {gpu_id} uses {used_memory} MiB, above {max_used_memory_mib} MiB"
            )
    return tuple(reasons)


def wait_for_free_gpus(
    executable: str,
    gpu_ids: Sequence[int],
    *,
    max_used_memory_mib: int,
    wait: bool,
    poll_s: float,
    output: TextIO,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    while True:
        reasons = gpu_busy_reasons(
            executable,
            gpu_ids,
            max_used_memory_mib=max_used_memory_mib,
            run=run,
        )
        if not reasons:
            print(
                f"[gpu-free] physical_gpus={sorted(set(gpu_ids))}",
                file=output,
                flush=True,
            )
            return
        if not wait:
            raise LauncherError("; ".join(reasons))
        print(
            f"[gpu-wait] {'; '.join(reasons)}; retrying in {poll_s:.1f}s",
            file=output,
            flush=True,
        )
        sleep(poll_s)


@dataclass
class ManagedChild:
    command: RoleCommand
    process: subprocess.Popen
    log_handle: Any
    pgid: int


def _returncode(value: int) -> int:
    if value > 0:
        return value
    if value < 0:
        return 128 + abs(value)
    return 1


def _parse_host_port(value: str) -> tuple[str, int]:
    host, separator, raw_port = value.rpartition(":")
    if not separator or not host or not raw_port.isdigit():
        raise ValueError(f"endpoint must be host:port, got {value!r}")
    return host, int(raw_port)


class ProcessSupervisor:
    """Fail-fast process-group manager with bounded TERM/KILL cleanup."""

    def __init__(
        self,
        *,
        termination_grace_s: float,
        kill_grace_s: float,
        poll_s: float,
        cwd: str,
        output: TextIO = sys.stdout,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        pre_start_check: Optional[Callable[[RoleCommand], None]] = None,
    ) -> None:
        self.termination_grace_s = termination_grace_s
        self.kill_grace_s = kill_grace_s
        self.poll_s = poll_s
        self.cwd = cwd
        self.output = output
        self._popen = popen
        self._pre_start_check = pre_start_check
        self.children: dict[str, ManagedChild] = {}
        self._shutdown_signal: Optional[int] = None
        self._closed = False

    def request_shutdown(self, signum: int, _frame: Any = None) -> None:
        if self._shutdown_signal is None:
            self._shutdown_signal = int(signum)

    def consume_shutdown_request(self) -> Optional[int]:
        """Acknowledge one graceful-stop request before failure cleanup starts."""
        signum = self._shutdown_signal
        self._shutdown_signal = None
        return signum

    def _check_shutdown(self) -> None:
        if self._shutdown_signal is not None:
            raise LauncherError(
                f"launcher received signal {self._shutdown_signal}",
                role="launcher",
                returncode=128 + self._shutdown_signal,
            )

    def start(self, command: RoleCommand) -> ManagedChild:
        self._check_shutdown()
        if command.role in self.children:
            raise LauncherError(f"duplicate managed role {command.role!r}")
        os.makedirs(os.path.dirname(command.log_path), exist_ok=True)
        log_handle = open(command.log_path, "xb", buffering=0)
        process = None
        try:
            if self._pre_start_check is not None:
                self._pre_start_check(command)
            self._check_shutdown()
            process = self._popen(
                list(command.argv),
                cwd=self.cwd,
                env=dict(command.env),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                shell=False,
                start_new_session=True,
            )
        finally:
            if process is None:
                log_handle.close()
        child = ManagedChild(command, process, log_handle, process.pid)
        self.children[command.role] = child
        print(
            f"[start] {command.role} pid={process.pid} log={command.log_path}",
            file=self.output,
            flush=True,
        )
        return child

    @staticmethod
    def _assert_alive(child: ManagedChild, purpose: str) -> None:
        returncode = child.process.poll()
        if returncode is not None:
            raise LauncherError(
                f"{child.command.role} exited with {returncode} while {purpose}",
                role=child.command.role,
                log_path=child.command.log_path,
                returncode=_returncode(returncode),
            )

    def wait_for_tcp(
        self,
        address: str,
        *,
        timeout_s: float,
        poll_s: float,
        child: Optional[ManagedChild] = None,
    ) -> None:
        host, port = _parse_host_port(address)
        deadline = time.monotonic() + timeout_s
        while True:
            self._check_shutdown()
            if child is not None:
                self._assert_alive(child, "waiting for its TCP endpoint")
            try:
                with socket.create_connection((host, port), timeout=min(1.0, poll_s)):
                    return
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise LauncherError(
                    f"TCP endpoint {address} was not ready within {timeout_s:.1f}s",
                    role=(
                        child.command.role if child is not None else "external-service"
                    ),
                    log_path=child.command.log_path if child is not None else None,
                )
            time.sleep(min(poll_s, max(0.0, deadline - time.monotonic())))

    def wait_for_http(
        self,
        url: str,
        *,
        timeout_s: float,
        poll_s: float,
        child: ManagedChild,
    ) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            self._check_shutdown()
            self._assert_alive(child, "waiting for its health endpoint")
            try:
                with urllib.request.urlopen(url, timeout=min(2.0, poll_s)) as response:
                    if 200 <= response.status < 300:
                        return
            except (OSError, urllib.error.URLError):
                pass
            if time.monotonic() >= deadline:
                raise LauncherError(
                    f"{child.command.role} was not healthy within {timeout_s:.1f}s",
                    role=child.command.role,
                    log_path=child.command.log_path,
                )
            time.sleep(min(poll_s, max(0.0, deadline - time.monotonic())))

    def monitor(
        self,
        worker_roles: Sequence[str],
        *,
        health_check: Optional[Callable[[], None]] = None,
    ) -> None:
        if not worker_roles:
            raise ValueError("monitor requires at least one finite worker")
        while True:
            self._check_shutdown()
            if health_check is not None:
                health_check()
            for role, child in self.children.items():
                returncode = child.process.poll()
                if child.command.persistent and returncode is not None:
                    raise LauncherError(
                        f"persistent role {role} exited with {returncode}",
                        role=role,
                        log_path=child.command.log_path,
                        returncode=_returncode(returncode),
                    )
            all_success = True
            for role in worker_roles:
                child = self.children[role]
                returncode = child.process.poll()
                if returncode is None:
                    all_success = False
                elif returncode != 0:
                    raise LauncherError(
                        f"role {role} exited with {returncode}",
                        role=role,
                        log_path=child.command.log_path,
                        returncode=_returncode(returncode),
                    )
            if all_success:
                if health_check is not None:
                    health_check()
                return
            time.sleep(self.poll_s)

    @staticmethod
    def _group_exists(pgid: int) -> bool:
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    @staticmethod
    def _signal(children: Sequence[ManagedChild], signum: int) -> None:
        for child in children:
            try:
                os.killpg(child.pgid, signum)
            except ProcessLookupError:
                pass

    def _wait_groups(
        self, children: Sequence[ManagedChild], timeout_s: float
    ) -> list[ManagedChild]:
        deadline = time.monotonic() + timeout_s
        while True:
            for child in children:
                child.process.poll()
            alive = [child for child in children if self._group_exists(child.pgid)]
            if not alive or time.monotonic() >= deadline:
                return alive
            time.sleep(min(self.poll_s, max(0.0, deadline - time.monotonic())))

    def _terminate(self, children: Sequence[ManagedChild]) -> None:
        active = [child for child in children if not child.log_handle.closed]
        if not active:
            return
        self._signal(active, signal.SIGTERM)
        alive = self._wait_groups(active, self.termination_grace_s)
        if alive:
            self._signal(alive, signal.SIGKILL)
            alive = self._wait_groups(alive, self.kill_grace_s)
        for child in active:
            try:
                child.process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                child.process.kill()
                child.process.wait(timeout=self.kill_grace_s)
            finally:
                child.log_handle.close()
        if alive:
            raise LauncherError(
                "process groups remained after SIGKILL: "
                f"{[child.command.role for child in alive]}"
            )

    def stop_roles(self, roles: Sequence[str]) -> None:
        unknown = [role for role in roles if role not in self.children]
        if unknown:
            raise LauncherError(f"cannot stop unmanaged roles: {unknown}")
        ordered = tuple(dict.fromkeys(roles))
        self._terminate([self.children[role] for role in ordered])
        print(
            f"[stop] controlled roles: {', '.join(ordered)}",
            file=self.output,
            flush=True,
        )

    def wait_for_role(
        self,
        role: str,
        *,
        timeout_s: float,
        required_alive_roles: Sequence[str] = (),
    ) -> None:
        child = self.children[role]
        deadline = time.monotonic() + timeout_s
        while True:
            self._check_shutdown()
            for required_role in required_alive_roles:
                self._assert_alive(self.children[required_role], f"waiting for {role}")
            returncode = child.process.poll()
            if returncode is not None:
                if returncode != 0:
                    raise LauncherError(
                        f"role {role} exited with {returncode}",
                        role=role,
                        log_path=child.command.log_path,
                        returncode=_returncode(returncode),
                    )
                return
            if time.monotonic() >= deadline:
                raise LauncherError(
                    f"role {role} did not finish within {timeout_s:.1f}s",
                    role=role,
                    log_path=child.command.log_path,
                )
            time.sleep(min(self.poll_s, max(0.0, deadline - time.monotonic())))

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._terminate(tuple(self.children.values()))


__all__ = [
    "GPUReservationSet",
    "LauncherError",
    "ProcessSupervisor",
    "RoleCommand",
    "gpu_busy_reasons",
    "wait_for_free_gpus",
]
