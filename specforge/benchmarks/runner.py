# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Drive tasks against an SGLang server and collect metrics.

Request scheduling is adapted from z-lab/dflash's MIT-licensed benchmark:
https://github.com/z-lab/dflash/blob/main/dflash/benchmark.py

The runner is speculative-algorithm agnostic; it reads whatever telemetry the
server returns.
"""

from __future__ import annotations

import importlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from specforge.benchmarks.client import PromptRenderer, SGLangClient
from specforge.benchmarks.config import BenchmarkConfig, SpeculativeConfig, TaskConfig
from specforge.benchmarks.report import (
    BenchmarkReport,
    RunRecord,
    TaskMetrics,
    format_summary,
)
from specforge.benchmarks.server import ManagedServer, build_server_args
from specforge.benchmarks.tasks import TASKS, BenchmarkTask, Sample


@dataclass
class SampleOutcome:
    """Everything one conversation produced, summed over its turns."""

    final_output: str
    output_tokens: int = 0
    spec_verify_count: int = 0
    num_requests: int = 0


@dataclass
class TaskRunner:
    """Runs one task's samples through the server and aggregates the result."""

    task: BenchmarkTask
    client: SGLangClient
    renderer: PromptRenderer
    sampling_params: Dict[str, Any]
    concurrency: int
    warmup: bool = True
    progress: bool = True

    def run(self) -> TaskMetrics:
        samples = self.task.samples()
        if not samples:
            raise ValueError(f"task {self.task.name!r} produced no samples")
        if self.warmup:
            self._run_batch(samples[: self.concurrency], timed=False)
            self.client.flush_cache()
        start = time.perf_counter()
        outcomes = self._run_batch(samples, timed=True)
        elapsed = time.perf_counter() - start
        return self._aggregate(samples, outcomes, elapsed)

    def _run_batch(self, samples: List[Sample], timed: bool) -> List[SampleOutcome]:
        outcomes: List[Optional[SampleOutcome]] = [None] * len(samples)
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {
                executor.submit(self._run_sample, sample): index
                for index, sample in enumerate(samples)
            }
            iterator = as_completed(futures)
            if self.progress:
                from tqdm import tqdm

                label = self.task.name + ("" if timed else " (warmup)")
                iterator = tqdm(iterator, total=len(futures), desc=label, leave=False)
            for future in iterator:
                outcomes[futures[future]] = future.result()
        return outcomes  # type: ignore[return-value]

    def _run_sample(self, sample: Sample) -> SampleOutcome:
        messages: List[Dict[str, Any]] = []
        if self.task.system_prompt:
            messages.append({"role": "system", "content": self.task.system_prompt})
        outcome = SampleOutcome(final_output="")
        for turn in sample.turns:
            messages.append({"role": "user", "content": turn})
            generation = self.client.generate(
                self.renderer.render(messages),
                self.sampling_params,
                image_path=sample.image,
            )
            messages.append({"role": "assistant", "content": generation.text})
            outcome.final_output = generation.text
            outcome.output_tokens += generation.completion_tokens
            outcome.spec_verify_count += generation.spec_verify_count or 0
            outcome.num_requests += 1
        return outcome

    def _aggregate(
        self, samples: List[Sample], outcomes: List[SampleOutcome], elapsed: float
    ) -> TaskMetrics:
        output_tokens = sum(outcome.output_tokens for outcome in outcomes)
        verify_count = sum(outcome.spec_verify_count for outcome in outcomes)
        accuracy = None
        num_scored = 0
        if self.task.scored:
            verdicts = [
                self.task.score(outcome.final_output, sample.label)
                for sample, outcome in zip(samples, outcomes)
            ]
            scored = [verdict for verdict in verdicts if verdict is not None]
            num_scored = len(scored)
            accuracy = sum(scored) / num_scored if num_scored else None
        return TaskMetrics(
            num_samples=len(samples),
            num_requests=sum(outcome.num_requests for outcome in outcomes),
            output_tokens=output_tokens,
            latency_seconds=elapsed,
            throughput_tokens_per_second=output_tokens / max(elapsed, 1e-12),
            accept_length=output_tokens / verify_count if verify_count else None,
            spec_verify_count=verify_count or None,
            accuracy=accuracy,
            num_scored=num_scored,
        )


def load_plugins(modules: List[str]) -> None:
    """Import modules whose side effect is registering extra tasks."""
    for module in modules:
        importlib.import_module(module)


def build_task(spec: TaskConfig, seed: int) -> BenchmarkTask:
    return TASKS.get(spec.name)(
        num_samples=spec.num_samples, subset=spec.subset, seed=seed
    )


def sampling_params_for(config: BenchmarkConfig, spec: TaskConfig, task: BenchmarkTask):
    """Per-task cap wins over the global cap, which wins over the task default."""
    max_new_tokens = (
        spec.max_new_tokens or config.sampling.max_new_tokens or task.max_new_tokens
    )
    params: Dict[str, Any] = {
        "temperature": config.sampling.temperature,
        "top_p": config.sampling.top_p,
        "top_k": config.sampling.top_k,
        "max_new_tokens": max_new_tokens,
    }
    if task.stop:
        params["stop"] = list(task.stop)
    return params


def run_tasks(
    config: BenchmarkConfig,
    client: SGLangClient,
    renderer: PromptRenderer,
    entry: Optional[SpeculativeConfig],
    report: BenchmarkReport,
    progress: bool = True,
) -> None:
    """Run every configured task against the current server, appending to report."""
    concurrency = (entry.batch_size if entry else None) or config.concurrency
    for spec in config.tasks:
        task = build_task(spec, config.seed)
        label = f"[{entry.describe() if entry else 'server'}] {spec.name}"
        print(f"Running {label} (num_samples={spec.num_samples or 'all'})")
        try:
            metrics = TaskRunner(
                task=task,
                client=client,
                renderer=renderer,
                sampling_params=sampling_params_for(config, spec, task),
                concurrency=concurrency,
                warmup=config.warmup,
                progress=progress,
            ).run()
        finally:
            task.cleanup()
        client.flush_cache()
        report.runs.append(
            RunRecord(
                config=(
                    {"label": entry.describe(), **entry.model_dump(exclude={"label"})}
                    if entry
                    else None
                ),
                task=spec.model_dump(),
                metrics=metrics,
            )
        )


def run_benchmark(config: BenchmarkConfig, progress: bool = True) -> BenchmarkReport:
    """Execute the whole config: every matrix entry, every task."""
    load_plugins(config.plugins)
    for spec in config.tasks:
        TASKS.get(spec.name)  # fail fast on typos before any server work
    client = SGLangClient(config.server.base_url, config.server.request_timeout_seconds)
    renderer = PromptRenderer(
        config.model,
        trust_remote_code=config.trust_remote_code,
        enable_thinking=config.enable_thinking,
    )
    report = BenchmarkReport(
        model=config.model,
        draft_model=config.draft_model,
        sampling=config.sampling.model_dump(),
        concurrency=config.concurrency,
    )
    if config.server.launch:
        for entry in config.matrix:
            server = ManagedServer(
                client,
                build_server_args(config, entry),
                env=config.server.env,
                launch_timeout_seconds=config.server.launch_timeout_seconds,
            )
            with server:
                run_tasks(config, client, renderer, entry, report, progress)
    else:
        if not client.is_ready():
            raise RuntimeError(
                f"no healthy SGLang server at {config.server.base_url}; start one "
                "or set server.launch=true"
            )
        run_tasks(config, client, renderer, None, report, progress)
    return report


def main(config: BenchmarkConfig) -> int:
    report = run_benchmark(config)
    print()
    print(format_summary(report))
    path = report.write(config.output.dir, config.output.name)
    print(f"\nReport written to {path}")
    return 0


__all__ = [
    "SampleOutcome",
    "TaskRunner",
    "build_task",
    "load_plugins",
    "main",
    "run_benchmark",
    "run_tasks",
    "sampling_params_for",
]
