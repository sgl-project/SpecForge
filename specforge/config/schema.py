# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The typed run config behind the single ``specforge train`` entry point.

The schema intentionally describes a *run*, not a legacy Python script.  Model
assembly, prompt preparation, topology and the strategy-specific objective live
behind this one validated contract.  YAML or JSON on disk and dotted CLI
overrides both re-validate through the same schema.
"""

from __future__ import annotations

import copy
import json
import os
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from specforge.config.flex_attention import validate_flex_kernel_options

# SGLang reserves one generated-token slot plus five internal slots, and its
# request validator rejects ``input_len >= context_len - 6``.  Accepting a
# prompt whose length is exactly ``data.max_length`` therefore needs 7 slots.
SGLANG_CAPTURE_CONTEXT_HEADROOM = 7


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictConfigModel):
    target_model_path: str
    #: Draft config as a local JSON file, a directory containing config.json,
    #: or a Hugging Face repository. EAGLE3, P-EAGLE, and DFlash can derive it
    #: from the target when omitted.
    draft_model_config: Optional[str] = None
    #: Load draft weights only. Unlike training.resume_from, this never restores
    #: optimizer/scheduler state, counters, data position, or RNG state.
    draft_checkpoint_path: Optional[str] = None
    #: Optional architecture override. Auto-generated defaults preserve the
    #: former trainers: EAGLE3=1, P-EAGLE=4, DFlash=1.
    draft_num_hidden_layers: Optional[int] = Field(default=None, gt=0)
    #: Optional DFlash block-size override (auto-generated default: 16).
    draft_block_size: Optional[int] = Field(default=None, gt=0)
    #: Online capture always runs on an external SGLang server; the in-process
    #: HF/custom target backends were removed with the server-only cutover, so
    #: configs naming them fail at load instead of being silently ignored.
    target_backend: Literal["sglang"] = "sglang"
    #: Retained for offline/config migration only. The server-only online path
    #: transports complete feature records and does not shard target outputs in
    #: the trainer.
    shard_target_output: bool = False
    #: Input family consumed by the capture provider. Built-in algorithms support
    #: text only; unsupported modalities fail during application resolution.
    input_modality: str = "text"
    trust_remote_code: bool = False
    embedding_key: str = "model.embed_tokens.weight"
    lm_head_key: str = "lm_head.weight"
    #: t2d/d2t vocab-mapping tensor file for the draft ("" = model has none).
    vocab_mapping_path: str = ""
    #: copy the frozen target embedding into the draft before training.
    load_target_embedding: bool = True
    #: aux hidden-state layer ids to capture (None = backend defaults).
    aux_hidden_state_layer_ids: Optional[List[int]] = None
    torch_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    cache_dir: Optional[str] = None
    #: DFlash-family and P-EAGLE mask token. ``None`` resolves it from the
    #: draft config, then the target tokenizer.
    mask_token_id: Optional[int] = None
    #: SGLang target-engine tuning. Ignored by hf/custom backends.
    sglang_attention_backend: str = "flashinfer"
    sglang_mem_fraction_static: float = Field(default=0.4, gt=0.0, le=1.0)
    sglang_context_length: Optional[int] = Field(default=None, gt=0)
    sglang_enable_nccl_nvls: bool = False
    sglang_enable_symm_mem: bool = False
    sglang_enable_torch_compile: bool = False
    sglang_enable_dp_attention: bool = False
    sglang_enable_dp_lm_head: bool = False
    sglang_ep_size: int = Field(default=1, gt=0)
    sglang_max_running_requests: Optional[int] = Field(default=None, gt=0)
    sglang_max_total_tokens: Optional[int] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_input_modality(self):
        if (
            not self.input_modality
            or self.input_modality.strip() != self.input_modality
            or any(character.isspace() for character in self.input_modality)
        ):
            raise ValueError(
                "model.input_modality must be a non-empty plugin identifier "
                "without whitespace"
            )
        return self


class DataConfig(StrictConfigModel):
    #: online mode — raw conversation JSON/JSONL.
    train_data_path: str = ""
    #: online mode — already-tokenized JSONL records.
    prompts_path: str = ""
    #: offline mode — directory of precomputed hidden-state .ckpt files.
    hidden_states_path: str = ""
    #: Reserved migration field. Online evaluation is unsupported; keep empty.
    eval_data_path: str = ""
    #: offline evaluation — directory of precomputed hidden-state .ckpt files.
    eval_hidden_states_path: str = ""
    max_length: int = Field(default=2048, gt=0)
    chat_template: str = "llama3"
    is_preformatted: bool = False
    train_only_last_turn: bool = False
    build_dataset_num_proc: int = Field(default=8, gt=0)
    #: Ordered background feature-loader workers. ``None`` preserves the
    #: former strategy defaults (EAGLE/P-EAGLE=4, DFlash-family=8).
    dataloader_num_workers: Optional[int] = Field(default=None, ge=0)
    cache_dir: str = "./cache"
    cache_key: Optional[str] = None
    max_prompts: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _exactly_one_source(self):
        sources = [
            bool(self.train_data_path),
            bool(self.prompts_path),
            bool(self.hidden_states_path),
        ]
        if sum(sources) != 1:
            raise ValueError(
                "set exactly one of data.train_data_path (raw online data), "
                "data.prompts_path (pre-tokenized online data), or "
                "data.hidden_states_path (offline features)"
            )
        return self


class TrackingConfig(StrictConfigModel):
    """Optional experiment tracking behind the trainer's logger seam."""

    report_to: Literal["none", "wandb", "tensorboard", "swanlab", "mlflow"] = "none"
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_key: Optional[str] = None
    wandb_offline: bool = False
    wandb_dir: Optional[str] = None
    swanlab_project: Optional[str] = None
    swanlab_name: Optional[str] = None
    swanlab_key: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: Optional[str] = None
    mlflow_run_name: Optional[str] = None


class ProfilingConfig(StrictConfigModel):
    """Per-rank PyTorch trace window in completed optimizer steps."""

    enabled: bool = False
    start_step: int = Field(default=30, ge=0)
    num_steps: int = Field(default=4, gt=0)
    record_shapes: bool = False


class RuntimeConfig(StrictConfigModel):
    """Streaming bounds shared by unified disaggregated producer roles."""

    producer_lease: int = Field(default=8, gt=0)
    in_flight_high_watermark: int = Field(default=256, gt=0)
    in_flight_low_watermark: int = Field(default=192, ge=0)
    resident_high_watermark_bytes: Optional[int] = Field(default=None, gt=0)
    resident_low_watermark_bytes: Optional[int] = Field(default=None, ge=0)
    feature_store_max_resident_bytes: Optional[int] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_watermarks(self):
        if self.in_flight_low_watermark > self.in_flight_high_watermark:
            raise ValueError(
                "runtime.in_flight_low_watermark must be <= "
                "runtime.in_flight_high_watermark"
            )
        if (
            self.resident_high_watermark_bytes is None
            and self.resident_low_watermark_bytes is not None
        ):
            raise ValueError(
                "runtime.resident_low_watermark_bytes requires "
                "runtime.resident_high_watermark_bytes"
            )
        if (
            self.resident_high_watermark_bytes is not None
            and self.resident_low_watermark_bytes is not None
            and self.resident_low_watermark_bytes > self.resident_high_watermark_bytes
        ):
            raise ValueError(
                "runtime.resident_low_watermark_bytes must be <= "
                "runtime.resident_high_watermark_bytes"
            )
        if (
            self.feature_store_max_resident_bytes is not None
            and self.resident_high_watermark_bytes is not None
            and self.feature_store_max_resident_bytes
            < self.resident_high_watermark_bytes
        ):
            raise ValueError(
                "runtime.feature_store_max_resident_bytes must be >= "
                "runtime.resident_high_watermark_bytes"
            )
        return self


class TrainerDeploymentConfig(StrictConfigModel):
    """Process topology used by the unified CLI launcher."""

    nnodes: int = Field(default=1, gt=0)
    nproc_per_node: int = Field(default=1, gt=0)
    #: Node-local identity. Shared multi-node configs normally provide this
    #: through ``specforge train --node-rank`` instead.
    node_rank: Optional[int] = Field(default=None, ge=0)
    master_addr: Optional[str] = None
    master_port: int = Field(default=29500, gt=0, le=65535)

    @model_validator(mode="after")
    def _validate_topology(self):
        if self.node_rank is not None and self.node_rank >= self.nnodes:
            raise ValueError(
                "deployment.trainer.node_rank must be smaller than "
                "deployment.trainer.nnodes"
            )
        if self.nnodes > 1 and not self.master_addr:
            raise ValueError(
                "deployment.trainer.master_addr is required when nnodes > 1"
            )
        return self


def _validate_cuda_devices(devices: List[str], *, field_name: str) -> None:
    for device in devices:
        if not device or device.strip() != device or "," in device:
            raise ValueError(
                f"{field_name} entries must be non-empty CUDA device tokens "
                f"without whitespace or commas, got {device!r}"
            )
    if len(set(devices)) != len(devices):
        raise ValueError(f"{field_name} must not contain duplicate CUDA devices")


class ManagedLocalMooncakeConfig(StrictConfigModel):
    """One loopback Mooncake master owned by the local CLI supervisor."""

    rpc_port: int = Field(default=35551, gt=0, le=65535)
    metadata_port: int = Field(default=35880, gt=0, le=65535)
    metrics_port: int = Field(default=35903, gt=0, le=65535)
    local_hostname: str = "127.0.0.1"
    protocol: Literal["tcp", "rdma"] = "tcp"
    rdma_devices: Optional[str] = None
    global_segment_size_bytes: int = Field(default=32 << 30, gt=0)
    local_buffer_size_bytes: int = Field(default=1 << 30, gt=0)
    startup_timeout_s: float = Field(default=60.0, gt=0)

    @model_validator(mode="after")
    def _validate_endpoint(self):
        ports = (self.rpc_port, self.metadata_port, self.metrics_port)
        if len(set(ports)) != len(ports):
            raise ValueError("managed_local Mooncake ports must be unique")
        if (
            not self.local_hostname
            or self.local_hostname.strip() != self.local_hostname
        ):
            raise ValueError(
                "managed_local.mooncake.local_hostname must be non-empty and "
                "must not contain surrounding whitespace"
            )
        return self


class ManagedLocalCaptureServerConfig(StrictConfigModel):
    """One patched SGLang capture server owned by the local supervisor."""

    port: int = Field(gt=0, le=65535)
    cuda_visible_devices: List[str] = Field(min_length=1)
    tp_size: int = Field(default=1, gt=0)
    #: Optional server-specific override. When omitted, inherit the canonical
    #: model.sglang_mem_fraction_static setting instead of silently shadowing it.
    mem_fraction_static: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    attention_backend: Optional[str] = None
    startup_timeout_s: float = Field(default=1800.0, gt=0)

    @model_validator(mode="after")
    def _validate_devices(self):
        _validate_cuda_devices(
            self.cuda_visible_devices,
            field_name="managed_local.capture_servers[].cuda_visible_devices",
        )
        if len(self.cuda_visible_devices) != self.tp_size:
            raise ValueError(
                "managed_local capture server tp_size must equal the number of "
                "cuda_visible_devices"
            )
        return self


class ManagedLocalStackConfig(StrictConfigModel):
    """Opt-in ownership of a complete single-node online capture stack."""

    trainer_cuda_visible_devices: List[str] = Field(min_length=1)
    mooncake: ManagedLocalMooncakeConfig = Field(
        default_factory=ManagedLocalMooncakeConfig
    )
    capture_servers: List[ManagedLocalCaptureServerConfig] = Field(min_length=1)
    shutdown_grace_s: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def _validate_local_resources(self):
        _validate_cuda_devices(
            self.trainer_cuda_visible_devices,
            field_name="managed_local.trainer_cuda_visible_devices",
        )
        mooncake_ports = {
            self.mooncake.rpc_port,
            self.mooncake.metadata_port,
            self.mooncake.metrics_port,
        }
        capture_ports = [server.port for server in self.capture_servers]
        if len(set(capture_ports)) != len(capture_ports):
            raise ValueError("managed_local capture server ports must be unique")
        overlap = mooncake_ports.intersection(capture_ports)
        if overlap:
            raise ValueError(
                "managed_local capture and Mooncake ports must not overlap: "
                f"{sorted(overlap)}"
            )

        server_devices = [
            device
            for server in self.capture_servers
            for device in server.cuda_visible_devices
        ]
        if len(set(server_devices)) != len(server_devices):
            raise ValueError(
                "managed_local capture servers must not share CUDA devices"
            )
        overlap_devices = set(server_devices).intersection(
            self.trainer_cuda_visible_devices
        )
        if overlap_devices:
            raise ValueError(
                "managed_local capture and trainer CUDA devices must be disjoint: "
                f"{sorted(overlap_devices)}"
            )
        return self


class DisaggregatedDeploymentConfig(StrictConfigModel):
    """Shared, non-secret topology for producer/consumer launch planning."""

    #: Attempt-scoped shared directory. The launcher derives refs, manifest,
    #: and lifecycle markers beneath it.
    control_dir: str
    #: Optional node-local root for online consumer SQLite/WAL and rank inboxes.
    #: When omitted, the historical control_dir-derived paths remain in use.
    consumer_state_dir: Optional[str] = None
    backend: Literal["shared_dir", "mooncake"]
    store_root: Optional[str] = None
    store_id: Optional[str] = None
    server_urls: List[str] = Field(default_factory=list)
    mooncake_metadata_server: Optional[str] = None
    mooncake_master_server_addr: Optional[str] = None
    mooncake_local_hostname: Optional[str] = None
    mooncake_protocol: Optional[str] = None
    mooncake_rdma_devices: Optional[str] = None
    #: Offline Mooncake producers own ingested feature objects until the
    #: consumer acknowledges them, so they require a positive segment. Online
    #: capture remains server-owned and the launcher forces client segments to
    #: zero for both SpecForge roles.
    producer_segment_size: Optional[int] = Field(default=None, gt=0)
    client_buffer_size: int = Field(default=256 << 20, gt=0)
    idle_timeout_s: Optional[float] = Field(default=None, gt=0)
    peer_wait_timeout_s: Optional[float] = Field(default=None, gt=0)
    producer_hold_s: Optional[float] = Field(default=None, gt=0)
    #: SIGTERM-to-SIGKILL grace for a plain (non-managed) supervisor teardown.
    #: Workers translate SIGTERM into cleanup (Mooncake drains, checkpoint
    #: flush, failure sentinels), so this window must cover that work.
    #: managed_local supervisors use managed_local.shutdown_grace_s instead.
    shutdown_grace_s: float = Field(default=30.0, gt=0)
    managed_local: Optional[ManagedLocalStackConfig] = None

    @model_validator(mode="after")
    def _validate_store(self):
        if not self.control_dir:
            raise ValueError("deployment.disaggregated.control_dir must not be empty")
        if self.consumer_state_dir is not None and (
            not self.consumer_state_dir
            or self.consumer_state_dir.strip() != self.consumer_state_dir
        ):
            raise ValueError(
                "deployment.disaggregated.consumer_state_dir must be non-empty "
                "and must not contain surrounding whitespace"
            )
        if self.backend == "shared_dir" and not self.store_root:
            raise ValueError(
                "deployment.disaggregated.store_root is required for shared_dir"
            )
        if self.managed_local is not None:
            if self.backend != "mooncake":
                raise ValueError("managed_local requires backend=mooncake")
            if self.store_root:
                raise ValueError("managed_local does not use store_root")
            if self.server_urls:
                raise ValueError(
                    "managed_local derives capture server URLs; do not set "
                    "deployment.disaggregated.server_urls"
                )
            configured_endpoints = {
                "mooncake_metadata_server": self.mooncake_metadata_server,
                "mooncake_master_server_addr": self.mooncake_master_server_addr,
                "mooncake_local_hostname": self.mooncake_local_hostname,
                "mooncake_protocol": self.mooncake_protocol,
                "mooncake_rdma_devices": self.mooncake_rdma_devices,
            }
            explicit = [name for name, value in configured_endpoints.items() if value]
            if explicit:
                raise ValueError(
                    "managed_local derives Mooncake endpoints; do not set "
                    f"{explicit}"
                )
            if self.producer_segment_size is not None:
                raise ValueError(
                    "managed_local online capture is server-owned; do not set "
                    "producer_segment_size"
                )
        return self


class DeploymentConfig(StrictConfigModel):
    """User-facing launch topology for ``specforge train``."""

    mode: Literal["local_colocated", "disaggregated"] = "local_colocated"
    trainer: TrainerDeploymentConfig = Field(default_factory=TrainerDeploymentConfig)
    disaggregated: Optional[DisaggregatedDeploymentConfig] = None

    @model_validator(mode="after")
    def _validate_mode(self):
        if self.mode == "disaggregated" and self.disaggregated is None:
            raise ValueError(
                "deployment.disaggregated is required when "
                "deployment.mode=disaggregated"
            )
        if self.mode != "disaggregated" and self.disaggregated is not None:
            raise ValueError(
                "deployment.disaggregated requires deployment.mode=disaggregated"
            )
        return self


class TrainingConfig(StrictConfigModel):
    strategy: str = "eagle3"
    num_epochs: int = Field(default=1, gt=0)
    max_steps: Optional[int] = Field(default=None, gt=0)
    total_steps: Optional[int] = Field(default=None, gt=0)
    batch_size: int = Field(default=1, gt=0)
    accumulation_steps: int = Field(default=1, gt=0)
    fsdp_sharding: Literal["SHARD_GRAD_OP", "FULL_SHARD", "NO_SHARD"] = "SHARD_GRAD_OP"
    learning_rate: float = Field(default=1e-4, gt=0.0)
    warmup_ratio: float = Field(default=0.015, ge=0.0, le=1.0)
    max_grad_norm: float = Field(default=0.5, gt=0.0)
    ttt_length: int = Field(default=7, gt=0)
    attention_backend: Literal["eager", "sdpa", "flex_attention", "fa", "usp"] = (
        "flex_attention"
    )
    flex_kernel_options: Optional[dict[str, bool | int | float | str]] = None
    draft_kernel_backend: Literal["torch", "liger"] = "torch"
    linear_cross_entropy_backend: Literal["torch", "liger"] = "torch"
    compact_zero_weight_ce_rows: bool = False
    adamw_backend: Literal["torch", "fused"] = "torch"
    #: Trainer tensor parallelism. The unified runtime currently requires one;
    #: target-model TP belongs to external or managed capture servers.
    tp_size: int = Field(default=1, gt=0)
    sp_ulysses_size: int = Field(default=1, gt=0)
    sp_ring_size: int = Field(default=1, gt=0)
    dist_timeout: int = Field(default=10, gt=0)
    #: EAGLE3 objective.
    lk_loss_type: Optional[Literal["lambda", "alpha"]] = None
    kl_scale: float = 1.0
    kl_decay: float = 1.0
    #: DFlash-family objective/model knobs.
    num_anchors: int = Field(default=512, gt=0)
    loss_decay_gamma: Optional[float] = None
    loss_type: Literal[
        "dflash",
        "dpace",
        "dpace-cumulative-confidence-only",
        "dpace-continuation-value-only",
    ] = "dflash"
    dpace_alpha: float = 0.5
    lambda_base_start: float = 1.0
    lambda_base_decay_ratio: float = 0.5
    dspark_ce_loss_alpha: float = 0.1
    dspark_l1_loss_alpha: float = 0.9
    dspark_confidence_head_alpha: float = 1.0
    #: P-EAGLE COD sampling/model knobs.
    num_depths: int = Field(default=8, gt=0)
    down_sample_ratio: float = 0.8
    down_sample_ratio_min: float = 0.2
    norm_before_residual: Optional[bool] = None
    save_interval: int = Field(default=0, ge=0)
    #: Run a full evaluation pass every N optimizer steps (0 = disabled).
    eval_interval: int = Field(default=0, ge=0)
    log_interval: int = Field(default=50, gt=0)
    #: CheckpointManager rotation: keep the newest N checkpoints (0 = keep all).
    max_checkpoints: int = Field(default=0, ge=0)
    #: Offline EAGLE3 teacher projection without materializing full-vocab fp32
    #: logits. Exact, but trades additional head passes for lower peak memory.
    compact_teacher: bool = False
    compact_teacher_chunk_size: Optional[int] = Field(default=None, gt=0)
    #: resume target: a checkpoint dir / file:// URI / run root.
    resume_from: Optional[str] = None
    #: ``all`` is an offline colocated run. Online runs launch producer and
    #: consumer as separate ``specforge train`` processes with the same config
    #: and different roles.
    role: Literal["auto", "all", "producer", "consumer"] = "all"
    seed: int = 42

    @field_validator("flex_kernel_options")
    @classmethod
    def _validate_flex_kernel_options(cls, value):
        return validate_flex_kernel_options(
            value, field_name="training.flex_kernel_options"
        )

    @model_validator(mode="after")
    def _validate_training_shape(self):
        if not 0.0 <= self.dpace_alpha <= 1.0:
            raise ValueError("training.dpace_alpha must be in [0, 1]")
        if not 0.0 < self.down_sample_ratio <= 1.0:
            raise ValueError("training.down_sample_ratio must be in (0, 1]")
        if not 0.0 < self.down_sample_ratio_min <= self.down_sample_ratio:
            raise ValueError(
                "training.down_sample_ratio_min must be in "
                "(0, training.down_sample_ratio]"
            )
        sp_size = self.sp_ulysses_size * self.sp_ring_size
        if self.attention_backend == "usp":
            if self.batch_size != 1:
                raise ValueError(
                    "USP attention currently requires training.batch_size=1"
                )
            if sp_size <= 1:
                raise ValueError(
                    "USP attention requires sp_ulysses_size * sp_ring_size > 1"
                )
        elif sp_size != 1:
            raise ValueError(
                "training.sp_ulysses_size/sp_ring_size require "
                "training.attention_backend=usp"
            )
        if (
            self.flex_kernel_options is not None
            and self.attention_backend != "flex_attention"
        ):
            raise ValueError(
                "training.flex_kernel_options require attention_backend='flex_attention'"
            )
        if (
            self.compact_zero_weight_ce_rows
            and self.linear_cross_entropy_backend != "liger"
        ):
            raise ValueError(
                "training.compact_zero_weight_ce_rows=true requires "
                "linear_cross_entropy_backend='liger'"
            )
        dflash_only = (
            self.draft_kernel_backend != "torch"
            or self.linear_cross_entropy_backend != "torch"
            or self.compact_zero_weight_ce_rows
        )
        if dflash_only and self.strategy != "dflash":
            raise ValueError(
                "training draft/linear kernel backends are supported only for dflash"
            )
        return self


def migrate_legacy_config(values: dict) -> dict:
    """Translate legacy deployment keys before constructing the domain config.

    Compatibility exists only at this raw loader boundary. The returned domain
    model never projects canonical deployment state back into ``training``.
    """

    if not isinstance(values, dict):
        return values
    raw = copy.deepcopy(values)
    training = raw.get("training")
    if not isinstance(training, dict):
        return raw

    has_legacy_mode = "deployment_mode" in training
    has_legacy_urls = "server_urls" in training
    has_legacy_database = "metadata_db_path" in training
    if not (has_legacy_mode or has_legacy_urls or has_legacy_database):
        return raw

    legacy_mode = training.pop("deployment_mode", None)
    legacy_urls = training.pop("server_urls", None)
    legacy_database = training.pop("metadata_db_path", None)
    deployment = raw.setdefault("deployment", {})
    if not isinstance(deployment, dict):
        return raw

    canonical_mode = deployment.get("mode")
    if legacy_mode is not None:
        if canonical_mode is not None and canonical_mode != legacy_mode:
            raise ValueError(
                "deployment.mode conflicts with legacy training.deployment_mode"
            )
        deployment["mode"] = legacy_mode
    elif canonical_mode is None and (legacy_urls or legacy_database):
        deployment["mode"] = "disaggregated"

    if legacy_urls is not None or legacy_database is not None:
        disaggregated = deployment.setdefault("disaggregated", {})
        if not isinstance(disaggregated, dict):
            return raw
        if legacy_urls is not None:
            canonical_urls = disaggregated.get("server_urls")
            if canonical_urls is not None and canonical_urls != legacy_urls:
                raise ValueError(
                    "deployment.disaggregated.server_urls conflicts with legacy "
                    "training.server_urls"
                )
            disaggregated["server_urls"] = legacy_urls
        if legacy_database:
            if os.path.basename(legacy_database) != "consumer.sqlite":
                raise ValueError(
                    "legacy training.metadata_db_path must end in consumer.sqlite; "
                    "use deployment.disaggregated.consumer_state_dir instead"
                )
            state_dir = os.path.dirname(legacy_database) or "."
            canonical_state_dir = disaggregated.get("consumer_state_dir")
            if canonical_state_dir is not None and canonical_state_dir != state_dir:
                raise ValueError(
                    "deployment.disaggregated.consumer_state_dir conflicts with "
                    "legacy training.metadata_db_path"
                )
            disaggregated["consumer_state_dir"] = state_dir
    return raw


class Config(StrictConfigModel):
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    run_id: str = "specforge-run"
    output_dir: str = "./output"

    @model_validator(mode="before")
    @classmethod
    def _default_role_for_deployment(cls, values):
        """Resolve the process-role default from the canonical topology."""
        if not isinstance(values, dict):
            return values
        raw = dict(values)
        deployment = dict(raw.get("deployment") or {})
        training = dict(raw.get("training") or {})
        mode = deployment.get("mode", "local_colocated")
        training.setdefault("role", "auto" if mode == "disaggregated" else "all")
        raw["training"] = training
        return raw

    @model_validator(mode="after")
    def _validate_run_structure(self):
        """Validate topology and cross-field shape without resolving algorithms."""
        mode = self.mode
        deployment = self.deployment.mode
        role = self.training.role

        if mode == "online" and deployment != "disaggregated":
            raise ValueError(
                "online training requires deployment.mode=disaggregated; "
                "colocated online training is no longer supported"
            )
        if mode == "online" and self.model.target_backend != "sglang":
            raise ValueError(
                "online training uses an external SGLang capture server and "
                "requires model.target_backend=sglang"
            )
        if role != "all" and deployment != "disaggregated":
            raise ValueError(
                "training.role=auto/producer/consumer requires "
                "deployment.mode=disaggregated"
            )
        if deployment == "disaggregated" and role == "all":
            raise ValueError(
                "deployment.mode=disaggregated requires "
                "training.role=auto, producer, or consumer"
            )

        if (
            self.model.draft_checkpoint_path is not None
            and self.training.resume_from is not None
        ):
            raise ValueError(
                "model.draft_checkpoint_path (weights-only warm start) and "
                "training.resume_from (full resume) are mutually exclusive"
            )
        if self.training.role == "producer" and self.profiling.enabled:
            raise ValueError(
                "profiling.enabled applies to trainer roles, not a capture-only "
                "producer"
            )
        if self.data.eval_data_path:
            raise ValueError(
                "data.eval_data_path is unsupported; online evaluation is not "
                "supported by the server-only capture path"
            )
        has_eval_source = bool(self.data.eval_hidden_states_path)
        has_eval_interval = self.training.eval_interval > 0
        if has_eval_source != has_eval_interval:
            raise ValueError(
                "an eval data source and training.eval_interval must be "
                "configured together"
            )
        if self.data.eval_hidden_states_path and mode != "offline":
            raise ValueError(
                "data.eval_hidden_states_path requires an offline training data "
                "source"
            )
        if (
            not self.training.compact_teacher
            and self.training.compact_teacher_chunk_size is not None
        ):
            raise ValueError(
                "training.compact_teacher_chunk_size requires "
                "training.compact_teacher=true"
            )
        if (
            mode == "online"
            and deployment == "disaggregated"
            and self.deployment.disaggregated is not None
            and self.deployment.disaggregated.backend != "mooncake"
        ):
            raise ValueError(
                "online disaggregated training requires "
                "deployment.disaggregated.backend=mooncake"
            )
        managed_local = (
            self.deployment.disaggregated.managed_local
            if self.deployment.disaggregated is not None
            else None
        )
        consumer_state_dir = (
            self.deployment.disaggregated.consumer_state_dir
            if self.deployment.disaggregated is not None
            else None
        )
        if consumer_state_dir is not None:
            if mode != "online" or deployment != "disaggregated":
                raise ValueError(
                    "deployment.disaggregated.consumer_state_dir is valid only "
                    "for online disaggregated training"
                )
        if (
            mode == "online"
            and deployment == "disaggregated"
            and managed_local is None
            and self.deployment.trainer.nnodes > 1
            and consumer_state_dir is None
        ):
            raise ValueError(
                "multi-node online consumers require an explicit node-local "
                "deployment.disaggregated.consumer_state_dir for SQLite/WAL"
            )
        if managed_local is not None:
            if mode != "online":
                raise ValueError("managed_local supports online capture only")
            if self.deployment.trainer.nnodes != 1:
                raise ValueError("managed_local requires deployment.trainer.nnodes=1")
            if self.training.role != "auto":
                raise ValueError(
                    "managed_local requires the persisted training role to be auto"
                )
            if self.training.resume_from is not None:
                raise ValueError("managed_local does not support resume")
            minimum_context_length = (
                self.data.max_length + SGLANG_CAPTURE_CONTEXT_HEADROOM
            )
            if (
                self.model.sglang_context_length is not None
                and self.model.sglang_context_length < minimum_context_length
            ):
                raise ValueError(
                    "managed_local model.sglang_context_length must be at least "
                    "data.max_length + "
                    f"{SGLANG_CAPTURE_CONTEXT_HEADROOM} for SGLang capture "
                    "request headroom "
                    f"({minimum_context_length})"
                )
            unsupported_dp_options = [
                name
                for name in (
                    "sglang_enable_dp_attention",
                    "sglang_enable_dp_lm_head",
                )
                if getattr(self.model, name)
            ]
            if unsupported_dp_options:
                raise ValueError(
                    "managed_local capture servers do not support SGLang DP "
                    f"options: {unsupported_dp_options}"
                )
            if (
                len(managed_local.trainer_cuda_visible_devices)
                != self.deployment.trainer.nproc_per_node
            ):
                raise ValueError(
                    "managed_local trainer_cuda_visible_devices count must equal "
                    "deployment.trainer.nproc_per_node"
                )
            ep_size = self.model.sglang_ep_size
            incompatible_tp_sizes = sorted(
                {
                    server.tp_size
                    for server in managed_local.capture_servers
                    if ep_size > server.tp_size or server.tp_size % ep_size
                }
            )
            if incompatible_tp_sizes:
                raise ValueError(
                    "model.sglang_ep_size must be no larger than and evenly "
                    "divide every managed capture-server tp_size; incompatible "
                    f"tp sizes: {incompatible_tp_sizes}"
                )
        if self.training.role == "producer" and self.training.resume_from is not None:
            raise ValueError("training.resume_from is valid only for a trainer role")
        if self.training.attention_backend == "usp":
            if mode != "offline":
                raise ValueError("USP attention currently requires offline features")
        if mode == "offline" and self.training.tp_size != 1:
            raise ValueError(
                "offline feature consumers do not implement trainer tensor "
                "parallelism; keep training.tp_size=1 so every non-SP rank "
                "receives its own data shard"
            )
        if (
            mode == "online"
            and deployment == "disaggregated"
            and (
                self.training.tp_size != 1
                or self.training.sp_ulysses_size != 1
                or self.training.sp_ring_size != 1
            )
        ):
            raise ValueError(
                "the disaggregated online consumer uses every trainer rank for "
                "data parallelism; configure target TP on the external server and "
                "keep training.tp_size/sp sizes at 1"
            )
        return self

    @property
    def mode(self) -> str:
        return "offline" if self.data.hidden_states_path else "online"

    def validate_world_size(self, world_size: int) -> None:
        if world_size < 1:
            raise ValueError(f"world_size must be positive, got {world_size}")
        tp_size = self.training.tp_size
        sp_size = self.training.sp_ulysses_size * self.training.sp_ring_size
        if world_size % tp_size:
            raise ValueError(
                f"world_size={world_size} must be divisible by "
                f"training.tp_size={tp_size}"
            )
        if world_size % sp_size:
            raise ValueError(
                f"world_size={world_size} must be divisible by draft sequence "
                f"parallel size {sp_size} "
                "(sp_ulysses_size * sp_ring_size)"
            )

    @classmethod
    def from_file(cls, path: str) -> "Config":
        with open(path, "r") as f:
            if path.endswith((".yaml", ".yml")):
                import yaml

                raw = yaml.safe_load(f)
            else:
                raw = json.load(f)
        return cls.model_validate(migrate_legacy_config(raw))


def apply_overrides(config: Config, overrides: List[str]) -> Config:
    """Apply dotted ``section.field=value`` overrides, re-validating the result."""
    raw = config.model_dump()
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not of the form path=value")
        path, value = item.split("=", 1)
        node = raw
        keys = path.split(".")
        for key in keys[:-1]:
            if not isinstance(node.get(key), dict):
                raise ValueError(f"override path {path!r} does not exist")
            node = node[key]
        if keys[-1] not in node:
            raise ValueError(f"override path {path!r} does not exist")
        current = node[keys[-1]]
        if isinstance(current, (dict, list)) and value.lstrip().startswith(("[", "{")):
            import yaml

            try:
                value = yaml.safe_load(value)
            except yaml.YAMLError as exc:
                raise ValueError(
                    f"override {path!r} contains an invalid structured value"
                ) from exc
        node[keys[-1]] = value  # pydantic coerces scalars on re-validation
    return Config.model_validate(raw)


def load_config(path: str, overrides: Optional[List[str]] = None) -> Config:
    config = Config.from_file(path)
    if overrides:
        config = apply_overrides(config, overrides)
    return config


__all__ = [
    "ModelConfig",
    "DataConfig",
    "TrackingConfig",
    "ProfilingConfig",
    "RuntimeConfig",
    "TrainerDeploymentConfig",
    "ManagedLocalMooncakeConfig",
    "ManagedLocalCaptureServerConfig",
    "ManagedLocalStackConfig",
    "DisaggregatedDeploymentConfig",
    "DeploymentConfig",
    "TrainingConfig",
    "Config",
    "migrate_legacy_config",
    "load_config",
    "apply_overrides",
]
