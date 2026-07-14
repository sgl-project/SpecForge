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

import json
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    target_backend: Literal["sglang", "hf", "custom"] = "sglang"
    #: Let colocated SGLang return only this target-TP rank's batch partition.
    #: HF/custom and multimodal targets capture the full batch and partition
    #: locally after the frozen-target forward.
    shard_target_output: bool = False
    #: Input family consumed by the frozen target.  Multimodal tensors are
    #: materialized inside rollout and never enter the control-plane payload.
    input_modality: Literal["text", "qwen2_5_vl"] = "text"
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


class DataConfig(StrictConfigModel):
    #: online mode — raw conversation JSON/JSONL.
    train_data_path: str = ""
    #: online mode — already-tokenized JSONL records.
    prompts_path: str = ""
    #: offline mode — directory of precomputed hidden-state .ckpt files.
    hidden_states_path: str = ""
    #: online evaluation — raw or pre-tokenized records prepared like training.
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
    #: Qwen2.5-VL image-resolution bounds (64 and 1024 28x28 patches).
    min_pixels: int = Field(default=50176, gt=0)
    max_pixels: int = Field(default=802816, gt=0)

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
        if self.max_pixels < self.min_pixels:
            raise ValueError("data.max_pixels must be >= data.min_pixels")
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


class TrainingConfig(StrictConfigModel):
    strategy: Literal["eagle3", "dflash", "domino", "dspark", "peagle"] = "eagle3"
    num_epochs: int = Field(default=1, gt=0)
    max_steps: Optional[int] = Field(default=None, gt=0)
    total_steps: Optional[int] = Field(default=None, gt=0)
    batch_size: int = Field(default=1, gt=0)
    accumulation_steps: int = Field(default=1, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    warmup_ratio: float = Field(default=0.015, ge=0.0, le=1.0)
    max_grad_norm: float = Field(default=0.5, gt=0.0)
    ttt_length: int = Field(default=7, gt=0)
    attention_backend: Literal["eager", "sdpa", "flex_attention", "fa", "usp"] = (
        "flex_attention"
    )
    #: Frozen target tensor parallelism and draft sequence-parallel topology.
    #: All groups are created by the one CLI process lifecycle.
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
    #: Colocated process or split producer/consumer deployment.
    deployment_mode: Literal["local_colocated", "disaggregated"] = "local_colocated"
    #: ``all`` is a colocated run. Disaggregated online runs launch producer
    #: and consumer as separate ``specforge train`` processes with the same
    #: config and different roles.
    role: Literal["all", "producer", "consumer"] = "all"
    server_urls: List[str] = Field(default_factory=list)
    metadata_db_path: Optional[str] = None
    seed: int = 42

    @model_validator(mode="after")
    def _validate_strategy_options(self):
        if self.strategy == "peagle" and self.batch_size != 1:
            raise ValueError("P-EAGLE currently requires training.batch_size=1")
        if self.strategy == "eagle3" and self.attention_backend == "eager":
            raise ValueError(
                "EAGLE3 attention_backend must be sdpa, flex_attention, fa, or usp"
            )
        if self.strategy == "peagle" and self.attention_backend != "flex_attention":
            raise ValueError(
                "P-EAGLE currently requires attention_backend=flex_attention"
            )
        if self.strategy in (
            "dflash",
            "domino",
            "dspark",
        ) and self.attention_backend in ("fa", "usp"):
            raise ValueError(
                "DFlash-family attention_backend must be eager, sdpa, or "
                "flex_attention"
            )
        if not 0.0 <= self.dpace_alpha <= 1.0:
            raise ValueError("training.dpace_alpha must be in [0, 1]")
        if not 0.0 < self.down_sample_ratio <= 1.0:
            raise ValueError("training.down_sample_ratio must be in (0, 1]")
        if not 0.0 < self.down_sample_ratio_min <= self.down_sample_ratio:
            raise ValueError(
                "training.down_sample_ratio_min must be in "
                "(0, training.down_sample_ratio]"
            )
        if self.role != "all" and self.deployment_mode != "disaggregated":
            raise ValueError(
                "training.role=producer/consumer requires "
                "training.deployment_mode=disaggregated"
            )
        if self.deployment_mode == "disaggregated" and self.role == "all":
            raise ValueError(
                "training.deployment_mode=disaggregated requires "
                "training.role=producer or consumer"
            )
        sp_size = self.sp_ulysses_size * self.sp_ring_size
        if self.attention_backend == "usp":
            if self.strategy != "eagle3":
                raise ValueError("USP attention is supported for EAGLE3 only")
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
        return self


class Config(StrictConfigModel):
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    run_id: str = "specforge-run"
    output_dir: str = "./output"

    @model_validator(mode="after")
    def _validate_capability_matrix(self):
        strategy = self.training.strategy
        mode = self.mode
        deployment = self.training.deployment_mode
        layers = self.model.aux_hidden_state_layer_ids
        modality = self.model.input_modality
        shard_target_output = self.model.shard_target_output
        eval_sources = [
            bool(self.data.eval_data_path),
            bool(self.data.eval_hidden_states_path),
        ]
        if (
            not self.model.draft_model_config
            and not self.model.draft_checkpoint_path
            and strategy not in ("eagle3", "peagle", "dflash")
        ):
            raise ValueError(
                f"training.strategy={strategy!r} requires "
                "model.draft_model_config; automatic target-derived configs are "
                "supported for EAGLE3, P-EAGLE, and DFlash"
            )
        if self.model.draft_num_hidden_layers is not None:
            if strategy not in ("eagle3", "peagle", "dflash"):
                raise ValueError(
                    "model.draft_num_hidden_layers supports EAGLE3, P-EAGLE, "
                    "and DFlash"
                )
            if strategy == "eagle3" and self.model.draft_num_hidden_layers != 1:
                raise ValueError(
                    "EAGLE3 has one draft decoder layer; use "
                    "model.draft_num_hidden_layers=1 or omit it"
                )
        if self.model.draft_block_size is not None and strategy != "dflash":
            raise ValueError("model.draft_block_size is an override for DFlash only")
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
        if sum(eval_sources) > 1:
            raise ValueError(
                "set at most one of data.eval_data_path (online eval) or "
                "data.eval_hidden_states_path (offline eval)"
            )
        has_eval_source = any(eval_sources)
        has_eval_interval = self.training.eval_interval > 0
        if has_eval_source != has_eval_interval:
            raise ValueError(
                "an eval data source and training.eval_interval must be "
                "configured together"
            )
        if self.data.eval_data_path and mode != "online":
            raise ValueError(
                "data.eval_data_path requires an online training data source"
            )
        if self.data.eval_hidden_states_path and mode != "offline":
            raise ValueError(
                "data.eval_hidden_states_path requires an offline training data "
                "source"
            )
        if has_eval_source and mode == "online" and deployment == "disaggregated":
            raise ValueError(
                "online disaggregated evaluation is not supported; use a "
                "colocated online eval stream"
            )
        if strategy in ("eagle3", "peagle"):
            if layers is not None and (len(layers) != 3 or any(i < 0 for i in layers)):
                raise ValueError(
                    "EAGLE-family model.aux_hidden_state_layer_ids must contain "
                    "exactly three non-negative layer ids"
                )
        elif layers is not None:
            raise ValueError(
                "DFlash-family capture layers come from draft_model_config; "
                "model.aux_hidden_state_layer_ids would be ignored"
            )
        if mode == "offline" and strategy not in ("eagle3", "dflash", "domino"):
            raise ValueError(
                "offline feature training supports EAGLE3, DFlash, and Domino"
            )
        if self.training.compact_teacher:
            if strategy != "eagle3" or mode != "offline" or modality != "text":
                raise ValueError(
                    "training.compact_teacher supports offline text EAGLE3 only"
                )
        elif self.training.compact_teacher_chunk_size is not None:
            raise ValueError(
                "training.compact_teacher_chunk_size requires "
                "training.compact_teacher=true"
            )
        if (
            strategy == "eagle3"
            and deployment == "disaggregated"
            and not self.model.vocab_mapping_path
        ):
            raise ValueError(
                "EAGLE3 disaggregated runs require model.vocab_mapping_path "
                "because producer and consumer cannot derive one shared mapping"
            )
        if strategy == "peagle" and deployment != "local_colocated":
            raise ValueError("P-EAGLE currently supports colocated online runs only")
        if strategy == "dspark" and deployment != "disaggregated":
            raise ValueError("DSpark requires disaggregated server capture")
        if (
            mode == "online"
            and deployment == "disaggregated"
            and self.model.target_backend != "sglang"
        ):
            raise ValueError(
                "disaggregated online capture uses an SGLang server and requires "
                "model.target_backend=sglang"
            )
        if (
            mode == "online"
            and deployment == "disaggregated"
            and self.training.total_steps is None
            and self.training.max_steps is None
        ):
            raise ValueError(
                "disaggregated streaming runs require training.total_steps or "
                "training.max_steps"
            )
        if self.training.role == "producer" and self.training.resume_from is not None:
            raise ValueError("training.resume_from is valid only for a trainer role")
        if self.training.metadata_db_path is not None and not (
            mode == "online"
            and deployment == "disaggregated"
            and self.training.role == "consumer"
        ):
            raise ValueError(
                "training.metadata_db_path belongs to the online consumer only"
            )
        if (
            mode == "online"
            and self.model.target_backend == "custom"
            and strategy not in ("eagle3", "peagle")
        ):
            raise ValueError(
                "the custom target backend currently supports EAGLE3 capture only"
            )
        if modality == "qwen2_5_vl":
            if strategy != "eagle3":
                raise ValueError("Qwen2.5-VL input is supported for EAGLE3 only")
            if mode != "online":
                raise ValueError("Qwen2.5-VL input requires online target capture")
            if deployment != "local_colocated":
                raise ValueError(
                    "Qwen2.5-VL currently requires deployment_mode=local_colocated"
                )
            if self.model.target_backend == "custom":
                raise ValueError("Qwen2.5-VL supports target_backend=sglang or hf")
            if self.data.prompts_path:
                raise ValueError(
                    "Qwen2.5-VL requires raw data.train_data_path so image metadata "
                    "can be re-materialized during rollout"
                )
            if self.training.batch_size != 1:
                raise ValueError("Qwen2.5-VL currently requires training.batch_size=1")
            if self.training.attention_backend == "usp":
                raise ValueError("Qwen2.5-VL does not support USP attention")
        if shard_target_output:
            if strategy != "eagle3" or mode != "online":
                raise ValueError(
                    "model.shard_target_output supports online EAGLE3 capture only"
                )
            if deployment != "local_colocated":
                raise ValueError(
                    "model.shard_target_output requires a colocated target"
                )
            if self.model.target_backend != "sglang":
                raise ValueError(
                    "model.shard_target_output requires target_backend=sglang"
                )
            if modality != "text":
                raise ValueError(
                    "model.shard_target_output is not supported for VLM capture"
                )
        if self.training.attention_backend == "usp":
            if mode != "offline":
                raise ValueError("USP attention currently requires offline features")
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
        return cls.model_validate(raw)


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
        node[keys[-1]] = value  # pydantic coerces the string on re-validation
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
    "TrainingConfig",
    "Config",
    "load_config",
    "apply_overrides",
]
