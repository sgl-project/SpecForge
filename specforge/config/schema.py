# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The typed run config behind the ``specforge`` CLI.

One validated :class:`Config` replaces the argparse-style launch knobs: it maps
1:1 onto what the DataFlow launch builders (``specforge.launch``) and the domain
``Trainer`` actually accept — no aspirational fields. YAML or JSON on disk;
dotted CLI overrides (``training.learning_rate=1e-4``) re-validate through the
same schema.
"""

from __future__ import annotations

import json
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
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


class DataConfig(BaseModel):
    #: online mode — pre-tokenized prompts jsonl ({"input_ids": [...], "loss_mask": [...]}).
    prompts_path: str = ""
    #: offline mode — directory of precomputed hidden-state .ckpt files.
    hidden_states_path: str = ""
    max_length: int = 2048

    @model_validator(mode="after")
    def _exactly_one_source(self):
        if bool(self.prompts_path) == bool(self.hidden_states_path):
            raise ValueError(
                "set exactly one of data.prompts_path (online) or "
                "data.hidden_states_path (offline)"
            )
        return self


class TrainingConfig(BaseModel):
    strategy: Literal["eagle3", "dflash", "domino"] = "eagle3"
    num_epochs: int = 1
    max_steps: Optional[int] = None
    total_steps: Optional[int] = None
    batch_size: int = 1
    accumulation_steps: int = 1
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.015
    max_grad_norm: float = 0.5
    ttt_length: int = 7
    attention_backend: str = "flex_attention"
    tp_size: int = 1
    sp_ulysses_size: int = 1
    sp_ring_size: int = 1
    save_interval: int = 0
    eval_interval: int = 0
    log_interval: int = 50
    #: CheckpointManager rotation: keep the newest N checkpoints (0 = keep all).
    max_checkpoints: int = 0
    #: resume target: a checkpoint dir / file:// URI / run root.
    resume_from: Optional[str] = None
    #: control-plane selection for the offline builder.
    deployment_mode: Literal[
        "local_colocated", "dataflow_colocated", "disaggregated"
    ] = "local_colocated"
    metadata_db_path: Optional[str] = None
    seed: int = 42


class Config(BaseModel):
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    run_id: str = "specforge-run"
    output_dir: str = "./output"

    @property
    def mode(self) -> str:
        return "offline" if self.data.hidden_states_path else "online"

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
