# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Typed configuration for ``specforge benchmark``.

A benchmark run is described by one validated :class:`BenchmarkConfig`.  It can
come from a YAML/JSON file, from CLI flags, or from both (flags win), and dotted
``section.field=value`` overrides re-validate through the same schema, exactly
like ``specforge train``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from specforge.config.schema import apply_dotted_overrides


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskConfig(StrictModel):
    """One task to run: a registered dataset plus how much of it to use."""

    #: Registered task name (see ``specforge benchmark --list-tasks``).
    name: str
    #: Number of samples to run; ``None`` runs the whole dataset.
    num_samples: Optional[int] = Field(default=None, gt=0)
    #: Dataset subsets/configs for tasks that support them (C-Eval, MMLU).
    subset: Optional[List[str]] = None
    #: Per-task generation cap; ``None`` uses the task's own default.
    max_new_tokens: Optional[int] = Field(default=None, gt=0)

    @classmethod
    def parse(cls, spec: str) -> "TaskConfig":
        """Parse the compact CLI form ``name[:num_samples[:subset,subset...]]``."""
        parts = spec.split(":")
        if len(parts) > 3 or not parts[0]:
            raise ValueError(
                f"invalid task spec {spec!r}; expected name[:num_samples[:subset,...]]"
            )
        name = parts[0]
        num_samples = None
        subset = None
        if len(parts) >= 2 and parts[1]:
            try:
                num_samples = int(parts[1])
            except ValueError as exc:
                raise ValueError(
                    f"invalid task spec {spec!r}; num_samples must be an integer"
                ) from exc
        if len(parts) == 3 and parts[2]:
            subset = [item for item in parts[2].split(",") if item]
        return cls(name=name, num_samples=num_samples, subset=subset)


class SamplingConfig(StrictModel):
    """Sampling parameters sent with every request."""

    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=1, ge=-1)
    #: Global cap applied on top of each task's default when set.
    max_new_tokens: Optional[int] = Field(default=None, gt=0)


class ServerConfig(StrictModel):
    """Where the SGLang server is, or how to launch it."""

    base_url: str = "http://127.0.0.1:30000"
    #: Launch a server per matrix entry instead of using a running one.
    launch: bool = False
    #: Seconds to wait for a launched server to report healthy.
    launch_timeout_seconds: int = Field(default=600, gt=0)
    #: Per-request HTTP timeout.
    request_timeout_seconds: int = Field(default=3600, gt=0)
    #: Extra ``sglang.launch_server`` arguments passed through verbatim, e.g.
    #: ``["--tp-size", "2", "--mem-fraction-static", "0.8"]``.
    args: List[str] = Field(default_factory=list)
    #: Extra environment variables for the launched server process.
    env: Dict[str, str] = Field(default_factory=dict)


class SpeculativeConfig(StrictModel):
    """One server configuration in the benchmark matrix.

    ``steps == 0`` is the target-only baseline; anything else enables
    speculative decoding with the given tree shape.
    """

    #: Human-readable label; derived from the fields when omitted.
    label: Optional[str] = None
    #: Request concurrency and server ``--max-running-requests`` for this
    #: entry; ``None`` uses the top-level ``concurrency``.
    batch_size: Optional[int] = Field(default=None, gt=0)
    algorithm: str = "EAGLE3"
    steps: int = Field(default=0, ge=0)
    topk: int = Field(default=0, ge=0)
    draft_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check_tree_shape(self) -> "SpeculativeConfig":
        if self.enabled and (self.topk <= 0 or self.draft_tokens <= 0):
            raise ValueError(
                "speculative entries need positive topk and draft_tokens "
                f"(got steps={self.steps}, topk={self.topk}, "
                f"draft_tokens={self.draft_tokens})"
            )
        return self

    @property
    def enabled(self) -> bool:
        return self.steps > 0

    def describe(self) -> str:
        if self.label:
            return self.label
        if not self.enabled:
            return "baseline"
        return (
            f"{self.algorithm.lower()}-s{self.steps}-k{self.topk}-d{self.draft_tokens}"
        )


class OutputConfig(StrictModel):
    """Where the JSON report goes: ``<dir>/<name>_<timestamp>.json``."""

    dir: str = "./benchmark_results"
    name: str = "benchmark"


class BenchmarkConfig(StrictModel):
    """Everything ``specforge benchmark`` needs for one invocation."""

    #: Target model path or Hugging Face id; supplies the tokenizer and chat
    #: template, and the served model when launching.
    model: str
    #: Draft model path; required to launch a speculative server.
    draft_model: Optional[str] = None
    tasks: List[TaskConfig] = Field(min_length=1)
    #: Server configurations to sweep.  Only consulted with ``server.launch``;
    #: a running server is measured as-is.
    matrix: Optional[List[SpeculativeConfig]] = None
    server: ServerConfig = Field(default_factory=ServerConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    #: Concurrent requests during the timed phase.
    concurrency: int = Field(default=1, gt=0)
    #: Send one concurrency-sized untimed batch before measuring.
    warmup: bool = True
    #: Pass ``enable_thinking`` to chat templates that support it.
    enable_thinking: bool = False
    trust_remote_code: bool = False
    #: Seed for tasks that randomize (e.g. GPQA choice order).
    seed: int = 42
    #: Importable modules that register additional tasks on import.
    plugins: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_matrix(self) -> "BenchmarkConfig":
        if not self.server.launch:
            if self.matrix is not None:
                raise ValueError(
                    "matrix is only used when server.launch is true; a running "
                    "server is measured with its current configuration"
                )
            return self
        if self.matrix is None:
            self.matrix = [SpeculativeConfig()]
        if not self.matrix:
            raise ValueError("matrix must list at least one entry when launching")
        if any(entry.enabled for entry in self.matrix) and not self.draft_model:
            raise ValueError("draft_model is required to launch a speculative server")
        labels = [entry.describe() for entry in self.matrix]
        if len(set(labels)) != len(labels):
            raise ValueError(f"matrix labels must be unique, got {labels}")
        return self

    @classmethod
    def from_file(cls, path: str) -> "BenchmarkConfig":
        return cls.model_validate(_read_mapping(path))

    def with_overrides(self, overrides: List[str]) -> "BenchmarkConfig":
        if not overrides:
            return self
        return type(self).model_validate(
            apply_dotted_overrides(self.model_dump(), list(overrides))
        )


def _read_mapping(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        if path.endswith((".yaml", ".yml")):
            import yaml

            raw = yaml.safe_load(handle)
        else:
            raw = json.load(handle)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"benchmark config {path!r} must be a mapping")
    return raw


#: CLI flag name -> dotted config path.  Flags override the config file.
CLI_FIELD_PATHS = {
    "model": "model",
    "draft_model": "draft_model",
    "base_url": "server.base_url",
    "launch_server": "server.launch",
    "concurrency": "concurrency",
    "max_new_tokens": "sampling.max_new_tokens",
    "output_dir": "output.dir",
    "name": "output.name",
    "trust_remote_code": "trust_remote_code",
    "enable_thinking": "enable_thinking",
}


def build_config(
    config_path: Optional[str],
    cli_values: Dict[str, Any],
    tasks: List[str],
    overrides: List[str],
) -> BenchmarkConfig:
    """Compose a config from a file, CLI flags, and dotted overrides.

    Precedence, lowest to highest: file, flags in ``cli_values`` whose value is
    not ``None``, ``--task`` specs (which replace any file tasks), overrides.
    """
    raw: Dict[str, Any] = _read_mapping(config_path) if config_path else {}
    for flag, path in CLI_FIELD_PATHS.items():
        value = cli_values.get(flag)
        if value is None:
            continue
        node = raw
        *parents, leaf = path.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    if tasks:
        raw["tasks"] = [TaskConfig.parse(spec).model_dump() for spec in tasks]
    return BenchmarkConfig.model_validate(raw).with_overrides(overrides)


__all__ = [
    "BenchmarkConfig",
    "OutputConfig",
    "SamplingConfig",
    "ServerConfig",
    "SpeculativeConfig",
    "TaskConfig",
    "build_config",
    "CLI_FIELD_PATHS",
]
