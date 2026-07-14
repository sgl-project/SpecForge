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
    draft_model_config: str
    target_backend: Literal["sglang", "hf", "custom"] = "sglang"
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
    max_length: int = Field(default=2048, gt=0)
    chat_template: str = "llama3"
    is_preformatted: bool = False
    train_only_last_turn: bool = False
    build_dataset_num_proc: int = Field(default=8, gt=0)
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
    attention_backend: Literal["eager", "sdpa", "flex_attention", "fa"] = (
        "flex_attention"
    )
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
    log_interval: int = Field(default=50, gt=0)
    #: CheckpointManager rotation: keep the newest N checkpoints (0 = keep all).
    max_checkpoints: int = Field(default=0, ge=0)
    #: resume target: a checkpoint dir / file:// URI / run root.
    resume_from: Optional[str] = None
    #: control-plane selection for the offline builder.
    deployment_mode: Literal[
        "local_colocated", "dataflow_colocated", "disaggregated"
    ] = "local_colocated"
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
                "EAGLE3 attention_backend must be sdpa, flex_attention, or fa"
            )
        if self.strategy == "peagle" and self.attention_backend != "flex_attention":
            raise ValueError(
                "P-EAGLE currently requires attention_backend=flex_attention"
            )
        if (
            self.strategy in ("dflash", "domino", "dspark")
            and self.attention_backend == "fa"
        ):
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
        return self


class Config(StrictConfigModel):
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    run_id: str = "specforge-run"
    output_dir: str = "./output"

    @model_validator(mode="after")
    def _validate_capability_matrix(self):
        strategy = self.training.strategy
        mode = self.mode
        deployment = self.training.deployment_mode
        layers = self.model.aux_hidden_state_layer_ids
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
        if mode == "offline" and strategy != "eagle3":
            raise ValueError(
                "offline feature training is currently supported for EAGLE3 only"
            )
        if (
            strategy == "eagle3"
            and (mode == "offline" or deployment == "disaggregated")
            and not self.model.vocab_mapping_path
        ):
            raise ValueError(
                "EAGLE3 offline and disaggregated runs require "
                "model.vocab_mapping_path because those roles cannot derive a "
                "shared mapping from the online prompt stream"
            )
        if strategy == "peagle" and deployment != "local_colocated":
            raise ValueError("P-EAGLE currently supports colocated online runs only")
        if strategy == "eagle3" and mode == "online" and self.training.batch_size != 1:
            raise ValueError(
                "EAGLE3 online capture currently requires training.batch_size=1"
            )
        if strategy == "dspark" and deployment != "disaggregated":
            raise ValueError("DSpark requires disaggregated server capture")
        if mode == "online" and deployment == "dataflow_colocated":
            raise ValueError(
                "online runs use local_colocated or disaggregated deployment"
            )
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
        if mode == "online" and self.training.resume_from is not None:
            raise ValueError(
                "online resume is not yet supported by the unified entry because "
                "streamed prompt/ref reconciliation is incomplete"
            )
        if self.training.role == "producer" and self.training.resume_from is not None:
            raise ValueError("training.resume_from is valid only for a trainer role")
        if self.model.target_backend == "custom" and strategy not in (
            "eagle3",
            "peagle",
        ):
            raise ValueError(
                "the custom target backend currently supports EAGLE3 capture only"
            )
        return self

    @property
    def mode(self) -> str:
        return "offline" if self.data.hidden_states_path else "online"

    def validate_world_size(self, world_size: int) -> None:
        supports_data_parallel = (
            self.mode == "online"
            and self.training.deployment_mode == "disaggregated"
            and self.training.role == "consumer"
        )
        if world_size > 1 and not supports_data_parallel:
            raise ValueError(
                "multi-rank training is currently supported only by an online "
                "disaggregated consumer; colocated and offline inputs are not yet "
                "data-parallel sharded"
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
    "TrainingConfig",
    "Config",
    "load_config",
    "apply_overrides",
]
