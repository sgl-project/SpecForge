# coding=utf-8
"""Strict manifest schema for production DFlash online fan-out runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from specforge.config.flex_attention import validate_flex_kernel_options

SCHEMA_VERSION = 1
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_LOSS_TYPES = {
    "dflash",
    "vp_drafter",
    "dpace",
    "dpace-cumulative-confidence-only",
    "dpace-continuation-value-only",
}


class ManifestError(ValueError):
    """A fan-out manifest violates the production launch contract."""


def _absolute(value: str, field: str) -> str:
    if not os.path.isabs(value):
        raise ValueError(f"{field} must be absolute")
    return os.path.abspath(value)


def parse_host_port(value: str, field: str) -> tuple[str, int]:
    host, separator, raw_port = value.rpartition(":")
    if not separator or not host or not raw_port.isdigit():
        raise ValueError(f"{field} must be host:port")
    port = int(raw_port)
    if not 1 <= port <= 65535:
        raise ValueError(f"{field} port must be in [1, 65535]")
    return host, port


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CaptureConfig(_StrictModel):
    target_model_path: str = Field(min_length=1)
    tokenizer_path: str = Field(min_length=1)
    draft_config_path: str = Field(min_length=1)
    train_data_path: str = Field(min_length=1)
    chat_template: str = Field(min_length=1)
    is_preformatted: bool
    is_pretokenized: bool = False
    max_length: int = Field(ge=1)
    max_prompts: int = Field(ge=1)
    dataset_num_proc: int = Field(ge=1)
    capture_layer_ids: tuple[int, ...] = Field(min_length=1)

    @field_validator(
        "target_model_path",
        "tokenizer_path",
        "draft_config_path",
        "train_data_path",
    )
    @classmethod
    def absolute_paths(cls, value: str, info) -> str:
        return _absolute(value, f"capture.{info.field_name}")

    @field_validator("capture_layer_ids")
    @classmethod
    def increasing_layers(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(layer < 0 for layer in value):
            raise ValueError("capture layer ids must be non-negative")
        if tuple(sorted(set(value))) != value:
            raise ValueError("capture layer ids must be unique and increasing")
        return value

    @model_validator(mode="after")
    def input_format(self) -> "CaptureConfig":
        if self.is_preformatted and self.is_pretokenized:
            raise ValueError("capture input cannot be preformatted and pretokenized")
        return self


class ServerConfig(_StrictModel):
    python_executable: str = Field(min_length=1)
    url: str = Field(min_length=1)
    gpu: int = Field(ge=0)
    mem_fraction_static: float = Field(gt=0.0, lt=1.0)
    readiness_timeout_s: float = Field(gt=0.0)
    readiness_poll_s: float = Field(gt=0.0)
    trust_remote_code: bool

    @field_validator("python_executable")
    @classmethod
    def absolute_python(cls, value: str) -> str:
        return _absolute(value, "server.python_executable")

    @field_validator("url")
    @classmethod
    def explicit_http_port(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "http" or not parsed.hostname or parsed.port is None:
            raise ValueError("server.url must be HTTP with an explicit port")
        return value.rstrip("/")


class MooncakeConfig(_StrictModel):
    mode: Literal["managed", "external"]
    master_executable: str = Field(min_length=1)
    local_hostname: str = Field(min_length=1)
    metadata_server: str = Field(min_length=1)
    master_server_addr: str = Field(min_length=1)
    protocol: Literal["tcp", "rdma"]
    rdma_devices: str
    global_segment_size: int = Field(ge=1)
    client_buffer_size: int = Field(ge=1)
    metrics_port: int = Field(ge=1, le=65535)

    @field_validator("metadata_server")
    @classmethod
    def metadata_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.port is None
        ):
            raise ValueError("metadata_server must be HTTP(S) with an explicit port")
        return value

    @field_validator("master_server_addr")
    @classmethod
    def master_endpoint(cls, value: str) -> str:
        parse_host_port(value, "mooncake.master_server_addr")
        return value


class PeriodicDraftCheckpointConfig(_StrictModel):
    step_interval: int = Field(ge=1)
    sliding_window_size: int = Field(ge=1)


class DraftCheckpointConfig(_StrictModel):
    # ``null`` disables periodic saves without implying an interval or retention.
    periodic: Optional[PeriodicDraftCheckpointConfig]
    save_epoch_end: bool


class TrainingConfig(_StrictModel):
    python_executable: str = Field(min_length=1)
    batch_size: int = Field(ge=1)
    accumulation_steps: int = Field(ge=1)
    attention_backend: Literal["eager", "sdpa", "flex_attention"]
    flex_kernel_options: Optional[dict[str, bool | int | float | str]] = None
    compact_zero_weight_ce_rows: bool = False
    draft_kernel_backend: Literal["torch", "liger"] = "torch"
    gradient_clip_backend: Literal["torch", "fused_adamw"] = "torch"
    linear_cross_entropy_backend: Literal["torch", "liger"] = "torch"
    max_grad_norm: float = Field(gt=0.0)
    checkpoint: DraftCheckpointConfig
    log_interval: int = Field(ge=1)
    mask_token_id: Optional[int] = Field(default=None, ge=0)
    embedding_key: Optional[str]
    lm_head_key: Optional[str]
    trust_remote_code: bool

    @field_validator("python_executable")
    @classmethod
    def absolute_python(cls, value: str) -> str:
        return _absolute(value, "training.python_executable")

    @field_validator("flex_kernel_options")
    @classmethod
    def valid_flex_kernel_options(cls, value):
        return validate_flex_kernel_options(value, field_name="flex_kernel_options")

    @model_validator(mode="after")
    def flex_options_contract(self) -> "TrainingConfig":
        if (
            self.flex_kernel_options is not None
            and self.attention_backend != "flex_attention"
        ):
            raise ValueError(
                "flex_kernel_options require attention_backend='flex_attention'"
            )
        if (
            self.compact_zero_weight_ce_rows
            and self.linear_cross_entropy_backend != "liger"
        ):
            raise ValueError(
                "compact_zero_weight_ce_rows=true requires "
                "linear_cross_entropy_backend='liger'"
            )
        return self


class GpuMonitorConfig(_StrictModel):
    enabled: bool = False
    poll_s: float = Field(default=1.0, gt=0.0)
    strict_process_ownership: bool = True


class RuntimeConfig(_StrictModel):
    delivery_mode: Literal["async_window"] = "async_window"
    consumer_prefetch_batches: Literal[0, 1] = 0
    run_dir: str = Field(min_length=1)
    gpu_lock_path_pattern: str = Field(min_length=1)
    nvidia_smi_executable: str = Field(min_length=1)
    gpu_max_used_memory_mib: int = Field(ge=0)
    gpu_poll_s: float = Field(gt=0.0)
    process_poll_s: float = Field(gt=0.0)
    termination_grace_s: float = Field(gt=0.0)
    kill_grace_s: float = Field(gt=0.0)
    master_readiness_timeout_s: float = Field(gt=0.0)
    master_readiness_poll_s: float = Field(gt=0.0)
    idle_timeout_s: float = Field(gt=0.0)
    consumer_registration_timeout_s: float = Field(gt=0.0)
    consumer_heartbeat_timeout_s: float = Field(gt=0.0)
    consumer_heartbeat_interval_s: float = Field(gt=0.0)
    finalize_timeout_s: float = Field(gt=0.0)
    gc_poll_s: float = Field(gt=0.0)
    window_lookbehind: int = Field(default=2, ge=0)
    window_lookahead: int = Field(default=40, ge=0)
    max_prefetch_per_consumer: int = Field(default=8, ge=0)
    max_outstanding_per_consumer: int = Field(default=8, ge=1)
    max_live_refs: int = Field(default=48, ge=1)
    max_live_bytes: Optional[int] = Field(default=None, ge=1)
    capture_batch_size: int = Field(default=8, ge=1)
    capture_batch_wait_s: float = Field(default=0.002, ge=0.0)
    registry_poll_s: float = Field(default=0.01, gt=0.0)
    max_capture_retries: int = Field(default=2, ge=0)
    capture_retry_backoff_s: float = Field(default=0.05, ge=0.0)
    capture_reserve_bytes_per_sample: int = Field(default=128 << 20, ge=1)
    gpu_monitor: GpuMonitorConfig = Field(default_factory=GpuMonitorConfig)

    @field_validator("run_dir", "gpu_lock_path_pattern")
    @classmethod
    def absolute_paths(cls, value: str, info) -> str:
        return _absolute(value, f"runtime.{info.field_name}")

    @model_validator(mode="after")
    def runtime_contract(self) -> "RuntimeConfig":
        if "{gpu_id}" not in self.gpu_lock_path_pattern:
            raise ValueError("gpu_lock_path_pattern must contain {gpu_id}")
        if self.consumer_heartbeat_timeout_s <= self.consumer_heartbeat_interval_s:
            raise ValueError("consumer heartbeat timeout must exceed its interval")
        if self.max_prefetch_per_consumer > self.window_lookahead + 1:
            raise ValueError(
                "max_prefetch_per_consumer must not exceed window_lookahead + 1"
            )
        return self


class VariantConfig(_StrictModel):
    subscription_id: str = Field(min_length=1, max_length=128)
    seed: int = Field(ge=0)
    loss_type: str
    loss_decay_gamma: Optional[float] = Field(default=None, ge=0.0)
    dpace_alpha: float = Field(ge=0.0, le=1.0)
    block_size: Optional[int] = Field(default=None, ge=2)
    num_anchors: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0)
    warmup_ratio: float = Field(ge=0.0, le=1.0)
    gpu: int = Field(ge=0)
    window_lookbehind: Optional[int] = Field(default=None, ge=0)
    window_lookahead: Optional[int] = Field(default=None, ge=0)
    max_prefetch: Optional[int] = Field(default=None, ge=0)

    @field_validator("subscription_id")
    @classmethod
    def safe_id(cls, value: str) -> str:
        if _SAFE_ID.fullmatch(value) is None:
            raise ValueError(f"subscription_id must match {_SAFE_ID.pattern!r}")
        return value

    @field_validator("loss_type")
    @classmethod
    def known_loss(cls, value: str) -> str:
        if value not in _LOSS_TYPES:
            raise ValueError(f"loss_type must be one of {sorted(_LOSS_TYPES)}")
        return value


class FanoutConfig(_StrictModel):
    schema_version: Literal[SCHEMA_VERSION]
    run_id: str = Field(min_length=1, max_length=128)
    capture: CaptureConfig
    server: ServerConfig
    mooncake: MooncakeConfig
    training: TrainingConfig
    runtime: RuntimeConfig
    variants: tuple[VariantConfig, ...] = Field(min_length=1, max_length=3)

    @field_validator("run_id")
    @classmethod
    def safe_run_id(cls, value: str) -> str:
        if _SAFE_ID.fullmatch(value) is None:
            raise ValueError(f"run_id must match {_SAFE_ID.pattern!r}")
        return value

    @model_validator(mode="after")
    def topology_contract(self) -> "FanoutConfig":
        subscriptions = [variant.subscription_id for variant in self.variants]
        if len(subscriptions) != len(set(subscriptions)):
            raise ValueError("variant subscription ids must be unique")
        gpus = [self.server.gpu, *(variant.gpu for variant in self.variants)]
        if len(gpus) != len(set(gpus)):
            raise ValueError("server and consumer GPU ids must be distinct")
        trainer_ids = [
            f"{self.run_id}-{variant.subscription_id}" for variant in self.variants
        ]
        if any(len(value) > 128 for value in trainer_ids):
            raise ValueError("derived trainer run ids must be at most 128 characters")
        effective_batch = self.training.batch_size * self.training.accumulation_steps
        if self.runtime.max_outstanding_per_consumer < effective_batch:
            raise ValueError(
                "max_outstanding_per_consumer must cover the effective batch"
            )
        if self.runtime.max_live_refs < self.runtime.max_outstanding_per_consumer:
            raise ValueError("max_live_refs must cover max_outstanding_per_consumer")
        if self.capture.max_prompts % effective_batch:
            raise ValueError("max_prompts must be divisible by the effective batch")
        oversized_blocks = {
            variant.subscription_id: variant.block_size
            for variant in self.variants
            if variant.block_size is not None
            and variant.block_size > self.capture.max_length
        }
        if oversized_blocks:
            raise ValueError(
                "variant block_size must not exceed capture.max_length: "
                f"{oversized_blocks}"
            )
        invalid_prefetch = {
            variant.subscription_id: (
                (
                    self.runtime.max_prefetch_per_consumer
                    if variant.max_prefetch is None
                    else variant.max_prefetch
                ),
                (
                    self.runtime.window_lookahead
                    if variant.window_lookahead is None
                    else variant.window_lookahead
                ),
            )
            for variant in self.variants
            if (
                self.runtime.max_prefetch_per_consumer
                if variant.max_prefetch is None
                else variant.max_prefetch
            )
            > (
                self.runtime.window_lookahead
                if variant.window_lookahead is None
                else variant.window_lookahead
            )
            + 1
        }
        if invalid_prefetch:
            raise ValueError(
                "variant max_prefetch must not exceed effective lookahead + 1: "
                f"{invalid_prefetch}"
            )
        server_port = urlparse(self.server.url).port
        metadata_port = urlparse(self.mooncake.metadata_server).port
        _, master_port = parse_host_port(
            self.mooncake.master_server_addr, "mooncake.master_server_addr"
        )
        ports = [server_port, metadata_port, master_port]
        if self.mooncake.mode == "managed":
            ports.append(self.mooncake.metrics_port)
        if len(ports) != len(set(ports)):
            raise ValueError("server and Mooncake ports must be distinct")
        return self


@dataclass(frozen=True)
class FanoutManifest:
    path: str
    digest: str
    config: FanoutConfig

    def __getattr__(self, name: str):
        return getattr(self.config, name)

    @property
    def log_dir(self) -> str:
        return os.path.join(self.runtime.run_dir, "logs")

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.runtime.run_dir, "dataset-cache")

    @property
    def metrics_dir(self) -> str:
        return os.path.join(self.runtime.run_dir, "metrics")

    @property
    def gpu_samples_path(self) -> str:
        return os.path.join(self.metrics_dir, "gpu_samples.jsonl")

    @property
    def gpu_summary_path(self) -> str:
        return os.path.join(self.metrics_dir, "gpu_summary.json")

    @property
    def producer_metadata_db_path(self) -> str:
        return os.path.join(self.runtime.run_dir, "state", "producer.db")

    @property
    def lifecycle_db_path(self) -> str:
        return self.producer_metadata_db_path + ".mooncake-lifecycle"

    @property
    def capture_inventory_db_path(self) -> str:
        return self.producer_metadata_db_path + ".server-capture"

    @property
    def window_registry_db_path(self) -> str:
        return os.path.join(self.runtime.run_dir, "state", "windowed-capture.db")

    def window_config(self, variant: VariantConfig) -> tuple[int, int, int]:
        lookbehind = (
            self.runtime.window_lookbehind
            if variant.window_lookbehind is None
            else variant.window_lookbehind
        )
        lookahead = (
            self.runtime.window_lookahead
            if variant.window_lookahead is None
            else variant.window_lookahead
        )
        max_prefetch = (
            self.runtime.max_prefetch_per_consumer
            if variant.max_prefetch is None
            else variant.max_prefetch
        )
        if max_prefetch > lookahead + 1:
            raise ManifestError(
                f"variant {variant.subscription_id!r} max_prefetch={max_prefetch} "
                f"exceeds lookahead + 1 ({lookahead + 1})"
            )
        return lookbehind, lookahead, max_prefetch

    def variant(self, subscription_id: str) -> VariantConfig:
        for variant in self.variants:
            if variant.subscription_id == subscription_id:
                return variant
        raise ManifestError(
            f"unknown subscription_id {subscription_id!r}; expected one of "
            f"{[variant.subscription_id for variant in self.variants]}"
        )

    def variant_metadata_db_path(self, variant: VariantConfig) -> str:
        return os.path.join(
            self.runtime.run_dir, "state", variant.subscription_id, "metadata.db"
        )

    def variant_output_dir(self, variant: VariantConfig) -> str:
        return os.path.join(
            self.runtime.run_dir, "checkpoints", variant.subscription_id
        )


def load_manifest(path: str | os.PathLike[str]) -> FanoutManifest:
    manifest_path = os.path.abspath(os.fspath(path))
    try:
        payload = Path(manifest_path).read_bytes()
        config = FanoutConfig.model_validate_json(payload)
    except OSError as exc:
        raise ManifestError(f"cannot read manifest {manifest_path!r}: {exc}") from exc
    except ValidationError as exc:
        raise ManifestError(f"invalid fan-out manifest: {exc}") from exc
    return FanoutManifest(
        path=manifest_path,
        digest=hashlib.sha256(payload).hexdigest(),
        config=config,
    )


def resolve_executable(value: str, *, sibling_of: Optional[str] = None) -> str:
    candidates = [value]
    if sibling_of is not None and not os.path.isabs(value):
        candidates.insert(0, os.path.join(os.path.dirname(sibling_of), value))
    for candidate in candidates:
        resolved = candidate if os.path.isabs(candidate) else shutil.which(candidate)
        if resolved and os.path.isfile(resolved) and os.access(resolved, os.X_OK):
            return os.path.realpath(resolved)
    raise ManifestError(f"executable is unavailable: {value!r}")


def validate_launch_inputs(manifest: FanoutManifest) -> None:
    required = {
        "target model": (manifest.capture.target_model_path, os.path.isdir),
        "tokenizer": (manifest.capture.tokenizer_path, os.path.isdir),
        "draft config": (manifest.capture.draft_config_path, os.path.isfile),
        "training data": (manifest.capture.train_data_path, os.path.isfile),
    }
    invalid = {
        name: path
        for name, (path, predicate) in required.items()
        if not predicate(path)
    }
    if invalid:
        raise ManifestError(f"required launch inputs are missing: {invalid}")
    resolve_executable(manifest.training.python_executable)
    resolve_executable(manifest.server.python_executable)
    resolve_executable(manifest.runtime.nvidia_smi_executable)
    if manifest.mooncake.mode == "managed":
        resolve_executable(
            manifest.mooncake.master_executable,
            sibling_of=manifest.training.python_executable,
        )
    if os.path.exists(manifest.runtime.run_dir):
        raise ManifestError(
            f"fresh-only runtime.run_dir already exists: {manifest.runtime.run_dir}"
        )
    try:
        with open(manifest.capture.draft_config_path, encoding="utf-8") as handle:
            draft_config = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read draft config: {exc}") from exc
    dflash = draft_config.get("dflash_config")
    if not isinstance(dflash, dict):
        raise ManifestError("draft config must contain dflash_config")
    actual_layers = tuple(dflash.get("target_layer_ids", ()))
    if actual_layers != manifest.capture.capture_layer_ids:
        raise ManifestError(
            f"capture layer ids differ from draft config: {actual_layers}"
        )
    if not isinstance(draft_config.get("hidden_size"), int):
        raise ManifestError("draft config hidden_size must be an integer")


__all__ = [
    "FanoutManifest",
    "GpuMonitorConfig",
    "ManifestError",
    "VariantConfig",
    "load_manifest",
    "parse_host_port",
    "resolve_executable",
    "validate_launch_inputs",
]
