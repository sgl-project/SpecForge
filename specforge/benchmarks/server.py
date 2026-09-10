# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Launch an SGLang server for one matrix entry and tear it down afterwards."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import List, Optional
from urllib.parse import urlparse

from specforge.benchmarks.client import SGLangClient
from specforge.benchmarks.config import BenchmarkConfig, SpeculativeConfig


def build_server_args(config: BenchmarkConfig, entry: SpeculativeConfig) -> List[str]:
    """Compose ``sglang.launch_server`` arguments for one matrix entry.

    Anything not derived from the matrix entry comes from ``server.args``
    verbatim, so every SGLang flag stays available without being mirrored
    here.
    """
    url = urlparse(config.server.base_url)
    batch_size = entry.batch_size or config.concurrency
    args = [
        "--model-path", config.model,
        "--host", url.hostname or "127.0.0.1",
        "--port", str(url.port or 30000),
        "--cuda-graph-max-bs-decode", str(batch_size),
        "--max-running-requests", str(batch_size),
    ]  # fmt: skip
    if entry.enabled:
        args += [
            "--speculative-algorithm", entry.algorithm,
            "--speculative-draft-model-path", config.draft_model,
            "--speculative-num-steps", str(entry.steps),
            "--speculative-eagle-topk", str(entry.topk),
            "--speculative-num-draft-tokens", str(entry.draft_tokens),
        ]  # fmt: skip
    if config.trust_remote_code:
        args.append("--trust-remote-code")
    return args + list(config.server.args)


class ManagedServer:
    """Context manager owning one ``python -m sglang.launch_server`` process."""

    def __init__(
        self,
        client: SGLangClient,
        args: List[str],
        env: Optional[dict] = None,
        launch_timeout_seconds: float = 600.0,
        poll_seconds: float = 2.0,
    ):
        self._client = client
        self._args = args
        self._env = {**os.environ, **(env or {})}
        self._launch_timeout = launch_timeout_seconds
        self._poll_seconds = poll_seconds
        self._process: Optional[subprocess.Popen] = None

    @property
    def command(self) -> List[str]:
        return [sys.executable, "-m", "sglang.launch_server", *self._args]

    def __enter__(self) -> "ManagedServer":
        print("Launching server:", " ".join(self.command))
        self._process = subprocess.Popen(self.command, env=self._env)
        try:
            self._wait_until_ready()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self._launch_timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"server exited with status {self._process.returncode} "
                    "before becoming ready"
                )
            if self._client.is_ready():
                return
            time.sleep(self._poll_seconds)
        raise TimeoutError(
            f"server did not become ready within {self._launch_timeout:.0f}s"
        )

    def stop(self) -> None:
        if self._process is None or self._process.poll() is not None:
            self._process = None
            return
        _terminate_tree(self._process.pid)
        try:
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._process = None


def _terminate_tree(pid: int) -> None:
    """SIGTERM a process and its children, then SIGKILL stragglers."""
    import psutil

    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    processes = parent.children(recursive=True) + [parent]
    for process in processes:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=15)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass


__all__ = ["ManagedServer", "build_server_args"]
