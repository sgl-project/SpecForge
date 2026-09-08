# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Colocated online run assembly used by ``specforge train``.

Every trainer rank loads one SGLang target shard next to its FSDP draft shard
and captures hidden states in process; no producer role, feature transport, or
second GPU pool is involved. This module is the colocated counterpart of
:mod:`specforge.training.disaggregated`: it turns a validated config into a
``TrainingRun`` and leaves trainer assembly to ``specforge.launch``.

Topology vocabulary:

* ``training.tp_size`` is the target-TP *island* width. Contiguous ranks form
  one island, capture the same TP-wide prompt batch, and each trains its own
  contiguous slice of it (``TargetBatchPartition``).
* Islands are target-DP replicas. Each takes a disjoint, equally sized shard of
  the same per-epoch prompt permutation the disaggregated producer streams
  (:mod:`specforge.training.prompt_plan`), so the two topologies are comparable
  at matched samples.
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple

from specforge.algorithms.contracts import FeatureMode
from specforge.algorithms.registry import AlgorithmRegistration
from specforge.config import SGLANG_CAPTURE_CONTEXT_HEADROOM, Config
from specforge.training import prompt_plan

#: Persisted with every checkpoint and compared on resume; bump when the
#: meaning of the planned stream changes.
PROMPT_PLAN_VERSION = "epoch-shard-v1"
CAPTURE_CONTRACT = "sglang_hidden_state_v1"


def _target_dp_layout() -> Tuple[int, int]:
    """Return ``(island_rank, island_count)`` for this process."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return 0, 1
    from specforge.distributed import get_dp_group

    group = get_dp_group()
    return dist.get_rank(group), dist.get_world_size(group)


def _sglang_engine_kwargs(cfg: Config) -> Dict[str, Any]:
    """Resolve ``model.sglang_*`` into in-process SGLang engine arguments.

    Sizing defaults follow the training shape: one island captures
    ``tp_size * batch_size`` requests of at most ``data.max_length`` tokens plus
    SGLang's request headroom. Explicit config values win; the schema already
    rejects values below these minimums.
    """
    from specforge.launch_plan import resolve_sglang_engine_args

    model, training = cfg.model, cfg.training
    context_length = model.sglang_context_length or (
        cfg.data.max_length + SGLANG_CAPTURE_CONTEXT_HEADROOM
    )
    target_batch_size = training.tp_size * training.batch_size
    return resolve_sglang_engine_args(
        model,
        overrides={
            "sglang_context_length": context_length,
            "sglang_max_running_requests": (
                model.sglang_max_running_requests or target_batch_size
            ),
            "sglang_max_total_tokens": (
                model.sglang_max_total_tokens or target_batch_size * context_length
            ),
            # Islands are the target replicas; each runs one SGLang replica.
            "sglang_dp_size": 1,
        },
    )


@dataclass
class _PromptPlan:
    """This island's deterministic prompt stream and its bookkeeping."""

    #: Lazy prompts in training order across all epochs.
    stream: Iterator[dict]
    #: Rank-local samples over all epochs (``Trainer`` uses it for resume math).
    dataset_size: int
    #: Persisted in checkpoints and compared on resume.
    metadata: Dict[str, Any] = field(default_factory=dict)


def _plan_prompts(cfg: Config, prompts: Sequence[dict]) -> _PromptPlan:
    t = cfg.training
    seed = prompt_plan.online_prompt_seed(cfg)
    island_rank, island_count = _target_dp_layout()
    batch_multiple = t.tp_size * t.batch_size
    per_epoch = prompt_plan.shard_epoch_size(
        len(prompts), shard_count=island_count, batch_multiple=batch_multiple
    )
    if per_epoch == 0:
        raise ValueError(
            "online prompt planning produced no complete target batch; provide "
            "at least tp_size * batch_size prompts per target-DP island"
        )
    stream = prompt_plan.iter_sharded_online_prompts(
        prompts,
        num_epochs=t.num_epochs,
        seed=seed,
        shard_rank=island_rank,
        shard_count=island_count,
        batch_multiple=batch_multiple,
    )
    return _PromptPlan(
        stream=stream,
        dataset_size=per_epoch * t.num_epochs // t.tp_size,
        metadata={
            "prompt_plan": PROMPT_PLAN_VERSION,
            "prompt_source_size": len(prompts),
            "prompt_seed": seed,
            "prompt_epochs": t.num_epochs,
            "target_dp_size": island_count,
            "colocated_capture": CAPTURE_CONTRACT,
        },
    )


def _resume_position(
    cfg: Config, plan: _PromptPlan
) -> Tuple[Optional[dict], Iterator[dict]]:
    """Validate a checkpoint against *plan* and skip the consumed prefix.

    The checkpoint carries the plan metadata plus the rank-local sample count.
    Resuming under a different plan (other prompts, seed, epochs, islands, TP
    or batch size) would silently train the wrong slice, so any recorded value
    that differs raises. A checkpoint written after the stream was consumed
    resumes with an empty stream; ``fit`` then exits at the persisted step.
    """
    t = cfg.training
    if t.resume_from is None:
        return None, plan.stream

    from specforge.training.checkpoint import CheckpointManager

    state = CheckpointManager.read_resume_state(t.resume_from)
    expected = {
        "dataset_size": plan.dataset_size,
        "batch_size": t.batch_size,
        "tp_size": t.tp_size,
        **plan.metadata,
    }
    mismatched = {
        key: (state[key], value)
        for key, value in expected.items()
        if key in state and state[key] is not None and state[key] != value
    }
    if mismatched:
        raise ValueError(
            "training.resume_from checkpoint does not match this colocated "
            "prompt plan; resuming would train the wrong prompt slice: "
            f"{mismatched} (checkpoint value, current value)"
        )
    if int(state.get("epoch", 0)) > 0:
        return state, iter(())
    # Every rank of an island consumed ``epoch_samples`` local samples, i.e.
    # ``epoch_samples * tp_size`` prompts of the island's stream.
    skip = int(state.get("epoch_samples", 0)) * t.tp_size
    return state, itertools.islice(plan.stream, skip, None)


def _ensure_streaming_vocab_mapping(
    cfg: Config,
    bundle,
    algorithm: AlgorithmRegistration,
    prompts: Sequence[dict],
    *,
    dataset_identity: str,
) -> None:
    """Derive the EAGLE3 draft-vocabulary map from the prepared prompts."""
    if FeatureMode.STREAMING not in algorithm.providers.vocab_mapping_modes:
        return
    from specforge.training.assembly import _install_dataset_vocab_mapping

    def count_loss_tokens() -> Counter:
        counts: Counter = Counter()
        for task in prompts:
            payload = task["payload"]
            counts.update(
                int(token)
                for token, keep in zip(payload["input_ids"], payload["loss_mask"])
                if keep
            )
        return counts

    _install_dataset_vocab_mapping(
        cfg,
        bundle,
        dataset_identity=dataset_identity,
        count_tokens=count_loss_tokens,
    )


def build_colocated_run(
    cfg: Config,
    *,
    algorithm: AlgorithmRegistration,
    build_model_bundle: Callable,
    prepare_prompts: Callable,
    logger: Optional[Callable],
):
    """Assemble one colocated online run (role ``all`` on every rank)."""
    if cfg.mode != "online" or cfg.deployment.mode != "local_colocated":
        raise ValueError("build_colocated_run assembles online local_colocated runs")
    if cfg.training.role != "all":
        raise ValueError("colocated online training runs every rank as role 'all'")

    import torch

    from specforge.inference.adapters import LocalSGLangCaptureAdapter
    from specforge.launch import build_colocated_online_runtime
    from specforge.offline_capture import load_offline_capture
    from specforge.torch_compat import configure_flex_attention_inductor
    from specforge.training.assembly import (
        TrainingRun,
        _common_launch_kwargs,
        _training_prompt_cache_key,
    )
    from specforge.training.schedule import resolve_total_steps

    t = cfg.training
    bundle = build_model_bundle(cfg)
    configure_flex_attention_inductor(t.attention_backend)

    prompts = prepare_prompts(cfg, bundle.input_tools, draft_config=bundle.draft_config)
    if not prompts:
        raise ValueError("online data preparation produced no trainable prompts")
    _ensure_streaming_vocab_mapping(
        cfg,
        bundle,
        algorithm,
        prompts,
        dataset_identity=_training_prompt_cache_key(cfg, bundle.input_tools),
    )
    plan = _plan_prompts(cfg, prompts)
    resume_state, stream = _resume_position(cfg, plan)
    total_steps = resolve_total_steps(
        total_steps=t.total_steps,
        max_steps=t.max_steps,
        num_samples=plan.dataset_size,
        batch_size=t.batch_size,
        accumulation_steps=t.accumulation_steps,
        num_epochs=1,
    )

    streaming = algorithm.providers.server_streaming_for(cfg.model.input_modality)
    target = load_offline_capture(
        cfg.model.target_model_path,
        torch_dtype=getattr(torch, cfg.model.torch_dtype),
        trust_remote_code=cfg.model.trust_remote_code,
        **_sglang_engine_kwargs(cfg),
    )
    target.set_capture_layers(
        bundle.capture_layers, capture_method=streaming.capture_method
    )

    launch_kwargs = _common_launch_kwargs(cfg, bundle, algorithm, logger=logger)
    launch_kwargs["total_steps"] = total_steps
    trainer = build_colocated_online_runtime(
        prompts=stream,
        feature_source=LocalSGLangCaptureAdapter(target, provider=streaming),
        draft_model=bundle.model,
        target_head=bundle.target_head,
        target_hidden_size=bundle.target_hidden_size,
        target_vocab_size=bundle.target_vocab_size,
        draft_vocab_size=bundle.draft_vocab_size,
        target_repr=streaming.target_representation,
        aux_hidden_state_layer_ids=bundle.capture_layers,
        resume_from=t.resume_from,
        resume_state=resume_state,
        dataset_size=plan.dataset_size,
        checkpoint_extra=plan.metadata,
        **launch_kwargs,
    )
    return TrainingRun(trainer=trainer)


__all__ = ["build_colocated_run"]
