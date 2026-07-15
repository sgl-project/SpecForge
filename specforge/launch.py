# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Launch helpers that wire the DataFlow runtime from a RunConfig.

The named builders span the topology axis (offline vs online, colocated vs
disaggregated); the draft model is the ``strategy=`` parameter, resolved to a
:class:`StrategySpec` (``specforge.training.strategies.registry``) — adding a model
is a registry entry, not a new ``build_*`` family.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from specforge.runtime.contracts import DeploymentMode, SampleRef
from specforge.runtime.control_plane import (
    DataFlowController,
    build_control_plane_for_mode,
)
from specforge.runtime.control_plane.metadata_store import (
    InMemoryMetadataStore,
    MetadataStore,
    SQLiteMetadataStore,
)
from specforge.runtime.data_plane import (
    FeatureStore,
    LocalFeatureStore,
    LocalRolloutStream,
)
from specforge.training.strategies.registry import StrategySpec, resolve_strategy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared assemblers — strategy- and topology-agnostic.
# ---------------------------------------------------------------------------


def _assemble_trainer(
    *,
    spec: StrategySpec,
    controller: DataFlowController,
    store: FeatureStore,
    ref_source: dict,  # {"refs": [...]} re-iterable (offline) | {"queue": q} stream (online)
    model,
    target_head,
    optimizer_factory,
    run_id: str,
    output_dir: str,
    batch_size: int,
    accumulation_steps: int,
    num_epochs: int,
    max_steps: Optional[int],
    total_steps: Optional[int] = None,
    save_interval: int,
    eval_interval: int,
    tp_size: int,
    sp_ulysses_size: int,
    sp_ring_size: int,
    logger,
    log_interval: int,
    collate_fn,
    strategy_kwargs: Optional[dict] = None,
    per_sample_transform=None,
    durable_ack: bool = True,
    resume_from: Optional[str] = None,
    max_checkpoints: int = 0,
):
    """Delegate to the domain ``Trainer`` (``specforge.training``) — the one
    assembly (FSDP wrap, optimizer-after-wrap, per-step strategy, loader, acks)
    shared by every builder; returns ``(trainer.controller, trainer.loader)``.
    """
    from specforge.training import Trainer

    trainer = Trainer(
        spec=spec,
        controller=controller,
        store=store,
        ref_source=ref_source,
        model=model,
        target_head=target_head,
        optimizer_factory=optimizer_factory,
        run_id=run_id,
        output_dir=output_dir,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        num_epochs=num_epochs,
        max_steps=max_steps,
        total_steps=total_steps,
        save_interval=save_interval,
        eval_interval=eval_interval,
        tp_size=tp_size,
        sp_ulysses_size=sp_ulysses_size,
        sp_ring_size=sp_ring_size,
        logger=logger,
        log_interval=log_interval,
        collate_fn=collate_fn,
        strategy_kwargs=strategy_kwargs,
        per_sample_transform=per_sample_transform,
        durable_ack=durable_ack,
        resume_from=resume_from,
        max_checkpoints=max_checkpoints,
    )
    return trainer.controller, trainer.loader


def _offline_io(spec: StrategySpec, max_len: int):
    """Resolve (collate_fn, per_sample_transform); raise if not wired for offline."""
    if spec.make_offline_collate is None or spec.make_offline_transform is None:
        raise NotImplementedError(
            f"offline data path for strategy {spec.name!r} is not wired yet: its "
            f"StrategySpec needs make_offline_transform + make_offline_collate "
            f"(DFlash/Domino use their own feature schema). See "
            f"specforge.training.strategies.registry."
        )
    return spec.make_offline_collate(), spec.make_offline_transform(max_len)


def _online_collate(spec: StrategySpec, collate_fn):
    """Resolve the online/streamed collate; raise if the strategy has none wired."""
    if collate_fn is not None:
        return collate_fn
    if spec.make_online_collate is None:
        raise NotImplementedError(
            f"online data path for strategy {spec.name!r} is not wired yet: its "
            f"StrategySpec needs make_online_collate (or pass an explicit collate_fn). "
            f"See specforge.training.strategies.registry."
        )
    return spec.make_online_collate()


def _resolve_metadata_store(
    metadata_store: Optional[MetadataStore],
    metadata_db_path: Optional[str],
) -> Optional[MetadataStore]:
    """Pick the online-disagg controller's metadata store.

    Producer and consumer must open the SAME durable store for cross-process
    commit/dedup/ack: an explicit ``metadata_store``, or a shared
    ``metadata_db_path`` (SQLite). ``None`` = private in-process store, un-shared.
    """
    if metadata_store is not None:
        return metadata_store
    if metadata_db_path is not None:
        return SQLiteMetadataStore(metadata_db_path)
    return None


def _dp_consumer_layout(
    dp_rank: Optional[int],
    dp_size: Optional[int],
    tp_size: int,
    sp_ulysses_size: int,
    sp_ring_size: int,
) -> Tuple[int, int]:
    """Resolve the online consumer's (dp_rank, dp_size), defaulting from dist.

    The DP online consumer shards DATA across ranks (one inbox each), so its DP
    width is the whole trainer world; tp/sp replication inside a shard is not
    wired yet and is rejected rather than silently double-training.
    """
    import torch.distributed as dist

    initialized = dist.is_available() and dist.is_initialized()
    if dp_size is None:
        dp_size = dist.get_world_size() if initialized else 1
    if dp_rank is None:
        dp_rank = dist.get_rank() if initialized else 0
    if dp_size > 1 and (tp_size != 1 or sp_ulysses_size != 1 or sp_ring_size != 1):
        raise NotImplementedError(
            "DP online consumer shards refs across dp ranks; tp/sp inside a DP "
            f"shard is not wired yet (got tp={tp_size}, sp_ulysses="
            f"{sp_ulysses_size}, sp_ring={sp_ring_size})"
        )
    if not 0 <= dp_rank < dp_size:
        raise ValueError(f"dp_rank {dp_rank} out of range for dp_size {dp_size}")
    return dp_rank, dp_size


def _dp_barrier() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()


def _normalize_prompt_epochs(prompt_epochs: int) -> int:
    prompt_epochs = int(prompt_epochs or 1)
    if prompt_epochs < 1:
        raise ValueError(f"prompt_epochs must be >= 1, got {prompt_epochs}")
    return prompt_epochs


def _epoch_online_prompts(prompts, epoch: int, prompt_epochs: int):
    """Build one epoch's prompt tasks without expanding the full run upfront."""
    if prompt_epochs == 1:
        return prompts

    out = []
    for idx, prompt in enumerate(prompts):
        item = dict(prompt)
        metadata = dict(prompt.get("metadata") or {})
        if "task_id" in prompt:
            metadata.setdefault("base_task_id", str(prompt["task_id"]))
        metadata["prompt_index"] = idx
        metadata["epoch"] = epoch
        metadata["prompt_epochs"] = prompt_epochs
        item["metadata"] = metadata
        # The online feature store is consume-once and commit dedups by
        # sample_id, so every epoch pass must mint distinct task/sample ids.
        item["task_id"] = f"epoch{epoch:04d}-prompt{idx:012d}"
        out.append(item)
    return out


def _checkpoint_global_step(resume_from: str) -> int:
    """Read ``global_step`` from the shared payload of the checkpoint at
    ``resume_from`` (a checkpoint dir, its state file, or a ``file://`` URI)."""
    import os

    import torch

    from specforge.training.checkpoint import STATE_FILE

    path = str(resume_from)
    if path.startswith("file://"):
        path = path[len("file://") :]
    if os.path.basename(path) != STATE_FILE:
        path = os.path.join(path, STATE_FILE)
    state = torch.load(path, map_location="cpu", weights_only=False)
    return int(state.get("global_step", 0) or 0)


def _assemble_rollout_workers(
    *,
    spec: StrategySpec,
    controller: DataFlowController,
    store: FeatureStore,
    run_id: str,
    target_hidden_size: int,
    target_vocab_size: Optional[int],
    draft_vocab_size: Optional[int],
    target_repr: str,
    aux_hidden_state_layer_ids,
    vocab_map_version: Optional[str],
    t2d,
    num_rollout_workers: int,
    device: str,
    target_model=None,
    feature_source=None,
):
    """Build the rollout producer workers (colocated-online + disagg producer).

    ``feature_source`` overrides the in-process adapter with a pre-built
    FeatureSource/RefSource (e.g. the server-capture transport, whose live
    SGLang server writes features straight to the store — no ``target_model``
    is loaded here). Otherwise an in-process adapter is built over
    ``target_model``.

    A *sequence* of feature sources fans out to one worker per source (the
    multi-server topology: 1 server : 1 adapter : 1 worker). All workers share
    the one controller, whose per-``worker_id`` leases keep their prompt slices
    disjoint. A single source keeps the legacy shape: ``num_rollout_workers``
    workers sharing it.
    """
    if not spec.supports_online:
        raise NotImplementedError(
            f"online capture for strategy {spec.name!r} is not wired yet: it needs "
            f"a {spec.name} capture path. Set feature_schema (or make_adapter) + "
            f"supports_online=True on its StrategySpec."
        )
    from specforge.inference.capture import CaptureConfig
    from specforge.inference.rollout_worker import RolloutWorker

    if aux_hidden_state_layer_ids is None:
        aux_hidden_state_layer_ids = tuple(
            getattr(target_model, "aux_hidden_states_layers", ()) or ()
        )
    if isinstance(feature_source, (list, tuple)):
        if not feature_source:
            raise ValueError("feature_source sequence is empty")
        if num_rollout_workers not in (1, len(feature_source)):
            raise ValueError(
                f"num_rollout_workers={num_rollout_workers} conflicts with "
                f"{len(feature_source)} feature sources (one worker per source)"
            )
        adapters = list(feature_source)
    elif feature_source is not None:
        adapters = [feature_source] * num_rollout_workers
    elif spec.make_adapter is not None:
        adapters = [
            spec.make_adapter(target_model, device=device, t2d=t2d)
        ] * num_rollout_workers
    else:
        from specforge.inference.adapters.policy import (
            EAGLE3_FEATURE_SCHEMA,
            PolicyFeatureAdapter,
        )

        adapters = [
            PolicyFeatureAdapter(
                target_model,
                schema=spec.feature_schema or EAGLE3_FEATURE_SCHEMA,
                device=device,
                t2d=t2d,
            )
        ] * num_rollout_workers
    capture_config = CaptureConfig.from_strategy(
        required_features=spec.required_features,
        aux_hidden_state_layer_ids=tuple(aux_hidden_state_layer_ids),
        target_repr=target_repr,
        target_hidden_size=target_hidden_size,
        target_vocab_size=target_vocab_size,
        draft_vocab_size=draft_vocab_size,
        vocab_map_version=vocab_map_version,
    )
    # strategy=spec.name: committed refs must pass the loader's strategy check
    return [
        RolloutWorker(
            controller,
            store,
            adapter,
            capture_config,
            run_id=run_id,
            worker_id=f"rollout-{i}",
            strategy=spec.name,
        )
        for i, adapter in enumerate(adapters)
    ]


# ---------------------------------------------------------------------------
# Offline (colocated + disaggregated).
# ---------------------------------------------------------------------------


def build_offline_runtime(
    *,
    strategy: str = "eagle3",
    hidden_states_path: str,
    eagle3_model,
    target_head,
    optimizer_factory,
    run_id: str,
    output_dir: str,
    ttt_length: int = 7,
    max_len: int = 2048,
    batch_size: int = 1,
    accumulation_steps: int = 1,
    num_epochs: int = 1,
    max_steps: Optional[int] = None,
    total_steps: Optional[int] = None,
    save_interval: int = 0,
    eval_interval: int = 0,
    tp_size: int = 1,
    sp_ulysses_size: int = 1,
    sp_ring_size: int = 1,
    logger=None,
    log_interval: int = 50,
    deployment_mode: DeploymentMode = "local_colocated",
    metadata_db_path: Optional[str] = None,
    resume_from: Optional[str] = None,
    max_checkpoints: int = 0,
    strategy_kwargs: Optional[dict] = None,
):
    """Assemble the colocated offline dataflow (``LocalFeatureStore``).

    ``eagle3_model`` is really "the composite draft model for ``strategy``"
    (back-compat name; must expose ``.draft_model``). ``deployment_mode`` selects
    the control plane: ``local_colocated`` skips the durable store/ack; other
    modes keep both on the same code path (training result is mode-independent).
    """
    spec = resolve_strategy(strategy)
    if spec.make_offline_reader is None:
        raise NotImplementedError(
            f"offline data path for strategy {spec.name!r} is not wired yet: its "
            f"StrategySpec needs make_offline_reader. See "
            f"specforge.training.strategies.registry."
        )
    collate_fn, per_sample_transform = _offline_io(spec, max_len)
    controller, durable_ack = build_control_plane_for_mode(
        deployment_mode, run_id, metadata_db_path=metadata_db_path
    )
    refs = spec.make_offline_reader(
        hidden_states_path, run_id=run_id, ttt_length=ttt_length, max_len=max_len
    ).read()
    store = LocalFeatureStore(run_id)
    return _assemble_trainer(
        spec=spec,
        controller=controller,
        store=store,
        ref_source={"refs": refs},
        model=eagle3_model,
        target_head=target_head if spec.uses_target_head else None,
        optimizer_factory=optimizer_factory,
        run_id=run_id,
        output_dir=output_dir,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        num_epochs=num_epochs,
        max_steps=max_steps,
        total_steps=total_steps,
        save_interval=save_interval,
        eval_interval=eval_interval,
        tp_size=tp_size,
        sp_ulysses_size=sp_ulysses_size,
        sp_ring_size=sp_ring_size,
        logger=logger,
        log_interval=log_interval,
        collate_fn=collate_fn,
        strategy_kwargs=strategy_kwargs,
        per_sample_transform=per_sample_transform,
        durable_ack=durable_ack,
        resume_from=resume_from,
        max_checkpoints=max_checkpoints,
    )


def build_disagg_offline_runtime(
    *,
    strategy: str = "eagle3",
    feature_store: FeatureStore,
    refs: List[SampleRef],
    eagle3_model,
    target_head,
    optimizer_factory,
    run_id: str,
    output_dir: str,
    max_len: int = 2048,
    batch_size: int = 1,
    accumulation_steps: int = 1,
    num_epochs: int = 1,
    max_steps: Optional[int] = None,
    total_steps: Optional[int] = None,
    save_interval: int = 0,
    eval_interval: int = 0,
    tp_size: int = 1,
    sp_ulysses_size: int = 1,
    sp_ring_size: int = 1,
    logger=None,
    log_interval: int = 50,
    resume_from: Optional[str] = None,
    max_checkpoints: int = 0,
    strategy_kwargs: Optional[dict] = None,
):
    """Consumer side of a disaggregated OFFLINE run.

    Trains from a caller-supplied cross-process ``feature_store`` and the
    ``disagg://`` refs its producer published. Same trainer assembly as the
    colocated offline path, so results match within determinism tolerance.
    """
    spec = resolve_strategy(strategy)
    collate_fn, per_sample_transform = _offline_io(spec, max_len)
    controller = DataFlowController(run_id)
    return _assemble_trainer(
        spec=spec,
        controller=controller,
        store=feature_store,
        ref_source={"refs": refs},
        model=eagle3_model,
        target_head=target_head if spec.uses_target_head else None,
        optimizer_factory=optimizer_factory,
        run_id=run_id,
        output_dir=output_dir,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        num_epochs=num_epochs,
        max_steps=max_steps,
        total_steps=total_steps,
        save_interval=save_interval,
        eval_interval=eval_interval,
        tp_size=tp_size,
        sp_ulysses_size=sp_ulysses_size,
        sp_ring_size=sp_ring_size,
        logger=logger,
        log_interval=log_interval,
        collate_fn=collate_fn,
        strategy_kwargs=strategy_kwargs,
        per_sample_transform=per_sample_transform,
        resume_from=resume_from,
        max_checkpoints=max_checkpoints,
    )


# ---------------------------------------------------------------------------
# Online (colocated + disaggregated producer/consumer + one-process interleaved).
# ---------------------------------------------------------------------------


def build_online_runtime(
    *,
    strategy: str = "eagle3",
    target_model,
    prompts,
    eagle3_model,
    optimizer_factory,
    run_id: str,
    output_dir: str,
    target_hidden_size: int,
    target_vocab_size: Optional[int] = None,
    draft_vocab_size: Optional[int] = None,
    target_repr: str = "logits",
    aux_hidden_state_layer_ids=None,
    vocab_map_version: Optional[str] = None,
    t2d=None,
    num_rollout_workers: int = 1,
    device: str = "cuda",
    ttt_length: int = 7,
    batch_size: int = 1,
    accumulation_steps: int = 1,
    num_epochs: int = 1,
    max_steps: Optional[int] = None,
    total_steps: Optional[int] = None,
    save_interval: int = 0,
    eval_interval: int = 0,
    tp_size: int = 1,
    sp_ulysses_size: int = 1,
    sp_ring_size: int = 1,
    collate_fn=None,
    logger=None,
    log_interval: int = 50,
    resume_from: Optional[str] = None,
    max_checkpoints: int = 0,
    strategy_kwargs: Optional[dict] = None,
    prompt_epochs: int = 1,
):
    """Assemble the colocated online dataflow; return
    ``(trainer, loader, workers, controller, run_interleaved)``.

    ``run_interleaved()`` owns the complete local lifecycle.  The loader pulls
    one bounded rollout batch at a time and trains it before asking for more, so
    the prompt pool is never materialized into GPU-resident features all at
    once. ``target_head`` is ``None`` on purpose: online rollout already
    materialized the target distribution.
    """
    spec = resolve_strategy(strategy)
    # Keep colocated online on a private queue; durable online runs use the
    # disagg builders with a shared store and streaming ref channel.
    controller, durable_ack = build_control_plane_for_mode("local_colocated", run_id)
    prompt_epochs = _normalize_prompt_epochs(prompt_epochs)
    if prompt_epochs == 1:
        controller.ingest_prompts(prompts)
    else:
        prompts = list(prompts)
        for epoch in range(prompt_epochs):
            controller.ingest_prompts(
                _epoch_online_prompts(prompts, epoch, prompt_epochs)
            )
    store = LocalFeatureStore(run_id)

    workers = _assemble_rollout_workers(
        spec=spec,
        target_model=target_model,
        controller=controller,
        store=store,
        run_id=run_id,
        target_hidden_size=target_hidden_size,
        target_vocab_size=target_vocab_size,
        draft_vocab_size=draft_vocab_size,
        target_repr=target_repr,
        aux_hidden_state_layer_ids=aux_hidden_state_layer_ids,
        vocab_map_version=vocab_map_version,
        t2d=t2d,
        num_rollout_workers=num_rollout_workers,
        device=device,
    )
    rollout_stream = LocalRolloutStream(
        controller=controller,
        workers=workers,
        feature_store=store,
        max_resident_samples=batch_size,
    )

    trainer, loader = _assemble_trainer(
        spec=spec,
        controller=controller,
        store=store,
        ref_source={"queue": rollout_stream},
        model=eagle3_model,
        target_head=None,
        optimizer_factory=optimizer_factory,
        run_id=run_id,
        output_dir=output_dir,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        num_epochs=num_epochs,
        max_steps=max_steps,
        total_steps=total_steps,
        save_interval=save_interval,
        eval_interval=eval_interval,
        tp_size=tp_size,
        sp_ulysses_size=sp_ulysses_size,
        sp_ring_size=sp_ring_size,
        logger=logger,
        log_interval=log_interval,
        collate_fn=_online_collate(spec, collate_fn),
        strategy_kwargs=strategy_kwargs,
        per_sample_transform=None,
        durable_ack=durable_ack,
        resume_from=resume_from,
        max_checkpoints=max_checkpoints,
    )

    def run_interleaved() -> int:
        """Train while rollout is pulled batch-by-batch; always stop workers."""
        with rollout_stream:
            return trainer.fit(loader)

    return trainer, loader, workers, controller, run_interleaved


def build_disagg_online_producer(
    *,
    strategy: str = "eagle3",
    target_model=None,
    prompts,
    feature_store: FeatureStore,
    channel,
    run_id: str,
    target_hidden_size: int,
    target_vocab_size: Optional[int] = None,
    draft_vocab_size: Optional[int] = None,
    target_repr: str = "logits",
    aux_hidden_state_layer_ids=None,
    vocab_map_version: Optional[str] = None,
    t2d=None,
    num_rollout_workers: int = 1,
    device: str = "cuda",
    feature_source=None,
    lease: int = 8,
    in_flight_high_watermark: int = 256,
    backpressure_poll_s: float = 0.2,
    peer_wait_timeout_s: Optional[float] = None,
    max_worker_failures: int = 3,
    max_prompt_attempts: Optional[int] = 5,
    metadata_store: Optional[MetadataStore] = None,
    metadata_db_path: Optional[str] = None,
    sleep=None,
    prompt_epochs: int = 1,
):
    """Producer side of an ONLINE disaggregated run (rollout pool).

    Workers put() into a cross-node ``feature_store``; committed refs stream to
    the consumer via ``channel``. Pass ``feature_source`` for the server-capture
    transport (live SGLang server writes to the store; ``target_model`` stays
    None) — a *sequence* of sources fans out to one worker per source (the
    multi-server topology; each worker drives its own server concurrently).
    ``metadata_store``/``metadata_db_path`` must be the same durable store
    the consumer opens. Returns ``(workers, drive_producer)``:
    ``drive_producer(should_stop=...)`` runs until the prompt pool drains and
    pauses above ``in_flight_high_watermark``. A successful or cooperative stop
    closes the channel; a failure publishes a distinct failure sentinel.
    ``peer_wait_timeout_s`` bounds a backpressure wait when the consumer dies
    before it can publish its stop sentinel.

    ``prompt_epochs`` repeats the prompt stream on the producer side by minting
    epoch-tagged task/sample ids. This keeps the consumer path consume-once while
    preserving ``num_epochs`` semantics for online streams.

    Failure semantics: a worker whose source raises (dead/unreachable server)
    has already failed its leases retryable — the surviving workers re-lease
    those prompts. After ``max_worker_failures`` *consecutive* failures the
    worker is dropped from rotation (its health is logged); if every worker is
    dropped while prompts remain, ``drive_producer`` raises instead of silently
    truncating the run. Per-task retryable failures are bounded by
    ``max_prompt_attempts`` (a poisoned prompt goes terminal, not infinite).
    The pool counts as drained only when no prompt is pending *or leased* —
    an all-failed round no longer reads as end-of-data. With N workers the
    watermark can overshoot by up to N * lease (each worker checks it
    independently before leasing).
    """
    import os
    import threading
    import time

    def producer_timing(message: str) -> None:
        print(
            f"[producer-timing] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}",
            flush=True,
        )

    def elapsed(start: float) -> str:
        return f"{time.perf_counter() - start:.3f}s"

    spec = resolve_strategy(strategy)
    sleep = sleep or time.sleep
    build_start = time.perf_counter()
    prompt_epochs = _normalize_prompt_epochs(prompt_epochs)
    if prompt_epochs > 1:
        prompts = list(prompts)
    base_prompt_count = len(prompts) if hasattr(prompts, "__len__") else "unknown"
    producer_timing(
        "build_disagg_online_producer enter "
        f"strategy={strategy} base_prompts={base_prompt_count} "
        f"prompt_epochs={prompt_epochs} "
        f"lease={lease} workers={num_rollout_workers} "
        f"watermark={in_flight_high_watermark}"
    )
    phase = time.perf_counter()
    controller = DataFlowController(
        run_id,
        metadata_store=_resolve_metadata_store(metadata_store, metadata_db_path),
        max_prompt_attempts=max_prompt_attempts,
    )
    producer_timing(f"DataFlowController created elapsed={elapsed(phase)}")

    phase = time.perf_counter()
    producer_timing("assemble rollout workers start")
    workers = _assemble_rollout_workers(
        spec=spec,
        target_model=target_model,
        feature_source=feature_source,
        controller=controller,
        store=feature_store,
        run_id=run_id,
        target_hidden_size=target_hidden_size,
        target_vocab_size=target_vocab_size,
        draft_vocab_size=draft_vocab_size,
        target_repr=target_repr,
        aux_hidden_state_layer_ids=aux_hidden_state_layer_ids,
        vocab_map_version=vocab_map_version,
        t2d=t2d,
        num_rollout_workers=num_rollout_workers,
        device=device,
    )
    producer_timing(
        "assemble rollout workers done "
        f"workers={len(workers)} elapsed={elapsed(phase)} "
        f"total_build_elapsed={elapsed(build_start)}"
    )

    def drive_producer(max_rounds: int = 1_000_000, should_stop=None) -> int:
        """Drive all workers until the pool drains; returns refs published.

        One worker runs inline; N workers run one thread each (the blocking
        HTTP prefill call releases the GIL, so servers genuinely overlap).
        The controller and feature store are lock-protected; the channel is
        not, so publishes serialize through ``publish_lock``.
        """
        drive_start = time.perf_counter()
        progress_interval = float(
            os.environ.get("DISAGG_PRODUCER_PROGRESS_INTERVAL", 30.0)
        )
        producer_timing(
            "drive_producer enter "
            f"workers={len(workers)} lease={lease} max_rounds={max_rounds} "
            f"watermark={in_flight_high_watermark} "
            f"progress_interval={progress_interval}"
        )
        for w in workers:
            producer_timing(f"rollout worker start worker_id={w.worker_id}")
            w.start()
        publish_lock = threading.Lock()
        state = {"produced": 0, "first_ref_logged": False}
        last_publish_log = {"t": time.perf_counter()}
        dead: dict = {}  # worker_id -> last failure reason

        def pool_drained() -> bool:
            st = controller.status()
            # leased counts too: a peer's in-flight lease may fail retryable
            # and come back — leaving then would strand it.
            return st["prompts_pending"] == 0 and st["prompts_leased"] == 0

        def run_worker(w) -> None:
            import os as _os

            # PROFILE_PRODUCER=N -> every N rounds print one [prod] line
            # splitting the round into backpressure-park / run_once / publish.
            _prof = int(_os.environ.get("PROFILE_PRODUCER", "0"))
            _ps = {
                "rounds": 0,
                "refs": 0,
                "bp": 0.0,
                "once": 0.0,
                "pub": 0.0,
                "t0": time.monotonic(),
                "infl": 0,
                "infl_max": 0,
            }
            failures = 0
            last_backpressure_log = 0.0
            for _ in range(max_rounds):
                if should_stop is not None and should_stop():
                    return
                _t = time.monotonic()
                _infl = channel.in_flight_remote()
                # backpressure: in_flight = published - consumer-acked
                backpressure_started = time.monotonic()
                while _infl >= in_flight_high_watermark:
                    now = time.perf_counter()
                    if (
                        progress_interval > 0
                        and now - last_backpressure_log >= progress_interval
                    ):
                        st = controller.status()
                        producer_timing(
                            "backpressure wait "
                            f"worker={w.worker_id} produced={state['produced']} "
                            f"in_flight={channel.in_flight_remote()} "
                            f"pending={st['prompts_pending']} "
                            f"leased={st['prompts_leased']} "
                            f"elapsed={elapsed(drive_start)}"
                        )
                        last_backpressure_log = now
                    if should_stop is not None and should_stop():
                        return
                    if (
                        peer_wait_timeout_s is not None
                        and time.monotonic() - backpressure_started
                        > peer_wait_timeout_s
                    ):
                        raise TimeoutError(
                            "producer backpressure timed out after "
                            f"{peer_wait_timeout_s:.0f}s waiting for consumer "
                            f"progress (in_flight={_infl})"
                        )
                    sleep(backpressure_poll_s)
                    _infl = channel.in_flight_remote()
                if _prof:
                    _ps["bp"] += time.monotonic() - _t
                    _ps["infl"] = _infl
                    _ps["infl_max"] = max(_ps["infl_max"], _infl)
                _t = time.monotonic()
                try:
                    run_once_start = time.perf_counter()
                    refs = w.run_once(max_tasks=lease)
                except Exception as exc:
                    # the worker already failed its leases retryable; peers
                    # (or this worker, next round) will re-lease them.
                    failures += 1
                    logger.warning(
                        "rollout worker %s failed (%d/%d): %s",
                        w.worker_id,
                        failures,
                        max_worker_failures,
                        exc,
                    )
                    if failures >= max_worker_failures:
                        dead[w.worker_id] = str(exc)
                        logger.error(
                            "dropping rollout worker %s after %d consecutive "
                            "failures; health=%s",
                            w.worker_id,
                            failures,
                            w.health(),
                        )
                        return
                    sleep(backpressure_poll_s)
                    continue
                failures = 0
                if _prof:
                    _ps["once"] += time.monotonic() - _t
                if refs:
                    _t = time.monotonic()
                    with publish_lock:
                        channel.publish_many(refs)
                        state["produced"] += len(refs)
                        now = time.perf_counter()
                        should_log = not state["first_ref_logged"]
                        should_log = should_log or (
                            progress_interval > 0
                            and now - last_publish_log["t"] >= progress_interval
                        )
                        if should_log:
                            st = controller.status()
                            producer_timing(
                                "published refs "
                                f"worker={w.worker_id} batch={len(refs)} "
                                f"produced={state['produced']} "
                                f"in_flight={channel.in_flight_remote()} "
                                f"pending={st['prompts_pending']} "
                                f"leased={st['prompts_leased']} "
                                f"run_once_elapsed={elapsed(run_once_start)} "
                                f"elapsed={elapsed(drive_start)}"
                            )
                            state["first_ref_logged"] = True
                            last_publish_log["t"] = now
                elif pool_drained():
                    producer_timing(
                        f"pool drained worker={w.worker_id} produced={state['produced']} "
                        f"elapsed={elapsed(drive_start)}"
                    )
                    return
                else:
                    # leased nothing: peers hold the remaining prompts (their
                    # leases may yet fail back into the pool) — wait, retry.
                    sleep(backpressure_poll_s)

        def ingest_epoch(epoch: int) -> None:
            epoch_prompts = _epoch_online_prompts(prompts, epoch, prompt_epochs)
            epoch_count = (
                len(epoch_prompts) if hasattr(epoch_prompts, "__len__") else "unknown"
            )
            phase = time.perf_counter()
            producer_timing(
                "controller.ingest_prompts start "
                f"epoch={epoch + 1}/{prompt_epochs} prompts={epoch_count}"
            )
            task_ids = controller.ingest_prompts(epoch_prompts)
            status = controller.status()
            producer_timing(
                "controller.ingest_prompts done "
                f"epoch={epoch + 1}/{prompt_epochs} tasks={len(task_ids)} "
                f"pending={status['prompts_pending']} elapsed={elapsed(phase)}"
            )

        def run_epoch_workers(live_workers) -> None:
            fatal: list = []  # non-transport errors escaping a worker thread

            def run_worker_guarded(w) -> None:
                try:
                    run_worker(w)
                except BaseException as exc:  # e.g. a channel publish failure
                    fatal.append((w.worker_id, exc))

            if len(live_workers) == 1:
                run_worker(live_workers[0])
            else:
                threads = [
                    threading.Thread(
                        target=run_worker_guarded,
                        args=(w,),
                        name=f"drive-{w.worker_id}",
                        daemon=True,
                    )
                    for w in live_workers
                ]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
            if fatal:
                raise fatal[0][1]

        try:
            live_workers = list(workers)
            for epoch in range(prompt_epochs):
                if should_stop is not None and should_stop():
                    break
                ingest_epoch(epoch)
                if not live_workers:
                    raise RuntimeError(
                        f"all rollout workers were already dropped before "
                        f"epoch {epoch + 1}/{prompt_epochs} could run — "
                        f"dead workers: {dead}"
                    )
                run_epoch_workers(live_workers)
                stopped = should_stop is not None and should_stop()
                live_workers = [w for w in live_workers if w.worker_id not in dead]
                if dead and not stopped and not pool_drained():
                    raise RuntimeError(
                        f"all rollout workers exited with {len(dead)} dropped as "
                        f"dead and prompts remaining — dead workers: {dead}"
                    )
                if stopped:
                    break
                st = controller.status()
                producer_timing(
                    "epoch drained "
                    f"epoch={epoch + 1}/{prompt_epochs} "
                    f"produced={state['produced']} "
                    f"prompts_failed={st['prompts_failed']} "
                    f"pending={st['prompts_pending']} leased={st['prompts_leased']} "
                    f"elapsed={elapsed(drive_start)}"
                )
            st = controller.status()
            stopped = should_stop is not None and should_stop()
            if st["prompts_failed"] and not stopped:
                raise RuntimeError(
                    "producer finished with "
                    f"{st['prompts_failed']} terminally failed prompt(s); "
                    "refusing to publish a successful EOF for partial data"
                )
            if not stopped and (st["prompts_pending"] or st["prompts_leased"]):
                raise RuntimeError(
                    "producer exhausted max_rounds before draining the prompt "
                    f"pool: pending={st['prompts_pending']} "
                    f"leased={st['prompts_leased']}"
                )
            producer_timing(
                "drive_producer returning "
                f"produced={state['produced']} prompts_failed={st['prompts_failed']} "
                f"pending={st['prompts_pending']} leased={st['prompts_leased']} "
                f"elapsed={elapsed(drive_start)}"
            )
            produced = state["produced"]
        except BaseException as exc:
            producer_timing(
                f"drive_producer failing channel produced={state['produced']} "
                f"elapsed={elapsed(drive_start)}"
            )
            try:
                channel.fail(f"{type(exc).__name__}: {exc}")
            except Exception:
                logger.exception("failed to publish producer failure sentinel")
            try:
                channel.close()
            except Exception:
                logger.exception("failed to close channel after producer failure")
            raise
        producer_timing(
            f"drive_producer closing channel produced={produced} "
            f"elapsed={elapsed(drive_start)}"
        )
        channel.close()
        return produced

    return workers, drive_producer


def build_disagg_online_consumer(
    *,
    strategy: str = "eagle3",
    feature_store: FeatureStore,
    channel,
    eagle3_model,
    optimizer_factory,
    run_id: str,
    output_dir: str,
    target_head=None,
    batch_size: int = 1,
    accumulation_steps: int = 1,
    num_epochs: int = 1,
    max_steps: Optional[int] = None,
    total_steps: Optional[int] = None,
    save_interval: int = 0,
    eval_interval: int = 0,
    tp_size: int = 1,
    sp_ulysses_size: int = 1,
    sp_ring_size: int = 1,
    collate_fn=None,
    idle_timeout_s: Optional[float] = None,
    metadata_store: Optional[MetadataStore] = None,
    metadata_db_path: Optional[str] = None,
    resume: bool = False,
    logger=None,
    log_interval: int = 50,
    strategy_kwargs: Optional[dict] = None,
    resume_from: Optional[str] = None,
    max_checkpoints: int = 0,
    dp_rank: Optional[int] = None,
    dp_size: Optional[int] = None,
    inbox_dir: Optional[str] = None,
):
    """Consumer (trainer) side of an ONLINE disaggregated run.

    Trains from a streamed ref ``channel`` + a consume-once ``feature_store``
    produced by another pool. Online resume needs BOTH knobs: ``resume=True``
    reconciles the durable marker (already-trained refs are skipped on the
    channel re-read) and ``resume_from`` restores the trainer state those acks
    correspond to. ``resume_from`` alone raises; a marker ahead of the
    checkpoint raises (the skipped samples' weight updates were rolled back —
    silent data loss).

    **Data-parallel trainer** (``dp_size > 1``, defaulting to the torchrun
    world): rank 0 runs the :class:`RefDistributor` — the run's single
    book-keeper. It alone reads ``channel``, commits into the ONE durable
    ``metadata_store``/``metadata_db_path`` (required on rank 0; the producer
    must NOT share this db — the distributor's commit-dedup would drop its
    rows), and round-robin dispatches aligned windows to per-rank inboxes under
    ``inbox_dir`` (default ``<channel>.inboxes``, trainer-local). Every rank
    consumes only its own inbox; durable acks gather to rank 0
    (:class:`DPAckController`) while the distributor mirrors the per-rank inbox
    acks onto the producer's backpressure counter. Passing ``inbox_dir``
    explicitly opts a single-rank run into the same distributor path;
    otherwise ``dp_size == 1`` keeps the original direct-channel path.
    """
    from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefQueue

    if resume_from and not resume:
        raise ValueError(
            "resume_from is set but resume is not: online resume requires BOTH — "
            "resume=True reconciles the durable metadata store (skips "
            "already-trained refs on the channel re-read) and resume_from restores "
            "the trainer state those acks correspond to; a checkpoint restore "
            "alone would re-train the whole re-streamed channel"
        )
    spec = resolve_strategy(strategy)
    dp_rank, dp_size = _dp_consumer_layout(
        dp_rank, dp_size, tp_size, sp_ulysses_size, sp_ring_size
    )

    def _reconcile(controller, resolved_store):
        """resume=True -> skip already-released refs; guard marker vs checkpoint."""
        if resolved_store is None:
            raise ValueError(
                "resume=True needs a durable metadata_store/metadata_db_path; an "
                "in-process store has no committed/ack history to reconcile against"
            )
        # Released == durably acked AND optimizer-step committed: skip exactly
        # those on the channel re-read; the committed-unacked tail re-trains.
        reconciled = controller.reconcile_on_restart(feature_store)
        if resume_from:
            marker_step = reconciled["global_step"]
            ckpt_step = _checkpoint_global_step(resume_from)
            if marker_step is not None and marker_step > ckpt_step:
                raise RuntimeError(
                    f"durable marker is at global_step={marker_step} but checkpoint "
                    f"{resume_from!r} is at global_step={ckpt_step}: samples acked "
                    f"after that checkpoint would be skipped on resume while the "
                    f"weight updates they produced were rolled back (data loss); "
                    f"resume from a checkpoint at step >= {marker_step} (the run's "
                    f"latest), or start a fresh run_id + metadata store to re-train "
                    f"from scratch"
                )
        return set(reconciled["released"])

    distributor = None
    if dp_size == 1 and inbox_dir is None:
        # Legacy direct-channel path, unchanged.
        store = _resolve_metadata_store(metadata_store, metadata_db_path)
        controller = DataFlowController(run_id, metadata_store=store)
        skip_ids = _reconcile(controller, store) if resume else None
        queue = StreamingRefQueue(
            channel, idle_timeout_s=idle_timeout_s, skip_ids=skip_ids
        )
    else:
        import torch.distributed as dist

        from specforge.runtime.control_plane.dp_ack import DPAckController
        from specforge.runtime.control_plane.metadata_store import NoOpMetadataStore
        from specforge.runtime.data_plane.ref_distributor import (
            InboxChannel,
            RefDistributor,
        )

        if inbox_dir is None:
            inbox_dir = channel.path + ".inboxes"
        # Symmetric preconditions (ALL ranks) — a rank-0-only raise would strand
        # the other ranks in the barrier below.
        if metadata_store is None and metadata_db_path is None:
            raise ValueError(
                "DP online consumer needs a durable metadata_store/"
                "metadata_db_path — the rank-0 distributor is the run's single "
                "ledger"
            )
        if dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
            if world != dp_size:
                raise ValueError(
                    f"DP online consumer: dp_size={dp_size} but the process "
                    f"group has {world} ranks — every rank must own exactly one "
                    f"inbox"
                )
        # Liveness default: a dead producer/distributor must never hang the
        # ranks silently. Generous enough for a cold server load before the
        # first ref.
        if idle_timeout_s is None:
            idle_timeout_s = 1800.0
        if dp_rank == 0:
            store = _resolve_metadata_store(metadata_store, metadata_db_path)
            if isinstance(store, NoOpMetadataStore):
                raise ValueError(
                    "DP online consumer needs a RETAINING metadata store: the "
                    "ledger is what dedups commits and reconciles restarts"
                )
            controller = DPAckController(
                run_id, is_authority=True, metadata_store=store
            )
            skip_ids = _reconcile(controller, store) if resume else None
            if not resume and store.committed_count() > 0:
                raise ValueError(
                    f"metadata store already holds {store.committed_count()} "
                    f"committed samples from a previous run; the distributor's "
                    f"commit-dedup would silently drop the whole re-streamed "
                    f"channel. Pass resume=True (+ resume_from) to reconcile, or "
                    f"start fresh (new metadata_db_path / delete the db)"
                )
            distributor = RefDistributor(
                channel,
                controller,
                inbox_dir,
                dp_size,
                skip_ids=skip_ids,
                idle_timeout_s=idle_timeout_s,
            )
        else:
            # Gather-participant only: throwaway store, records nothing.
            controller = DPAckController(
                run_id, is_authority=False, metadata_store=InMemoryMetadataStore()
            )
        # Inboxes must be re-created (rank 0, in RefDistributor.__init__) before
        # any rank opens a reader on a stale previous-attempt file.
        _dp_barrier()
        inbox = InboxChannel(RefDistributor.inbox_path(inbox_dir, dp_rank))
        queue = StreamingRefQueue(inbox, idle_timeout_s=idle_timeout_s)
        if distributor is not None:
            distributor.start()

    trainer, loader = _assemble_trainer(
        spec=spec,
        controller=controller,
        store=feature_store,
        ref_source={"queue": queue},
        model=eagle3_model,
        target_head=target_head if spec.uses_target_head else None,
        optimizer_factory=optimizer_factory,
        run_id=run_id,
        output_dir=output_dir,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        num_epochs=num_epochs,
        max_steps=max_steps,
        total_steps=total_steps,
        save_interval=save_interval,
        eval_interval=eval_interval,
        tp_size=tp_size,
        sp_ulysses_size=sp_ulysses_size,
        sp_ring_size=sp_ring_size,
        logger=logger,
        log_interval=log_interval,
        collate_fn=_online_collate(spec, collate_fn),
        strategy_kwargs=strategy_kwargs,
        per_sample_transform=None,
        resume_from=resume_from,
        max_checkpoints=max_checkpoints,
    )
    #: rank 0's RefDistributor handle in DP mode (None elsewhere) — callers may
    #: stop() it after fit for a clean early (max_steps) shutdown.
    trainer.ref_distributor = distributor
    return trainer, loader


def run_disagg_online_interleaved(
    *,
    trainer,
    loader,
    drive_producer,
    channel,
    producer_max_rounds: int = 1_000_000,
    join_timeout_s: Optional[float] = 30.0,
) -> int:
    """Run an online producer and the trainer CONCURRENTLY (O1.2).

    Shutdown is symmetric and hang-free: trainer finishes -> ``should_stop`` winds
    the producer down; producer finishes -> it closes the channel so ``fit``
    drains the tail and returns; producer raises -> the channel is closed and the
    exception re-raised here as root cause. Returns the trainer's final step.
    """
    import threading

    stop = threading.Event()
    err: dict = {}

    def _produce() -> None:
        try:
            drive_producer(producer_max_rounds, should_stop=stop.is_set)
        except BaseException as exc:  # surfaced to the main thread below
            err["exc"] = exc
            channel.close()  # never leave the consumer blocked on a dead producer

    thread = threading.Thread(
        target=_produce, name="disagg-online-producer", daemon=True
    )
    thread.start()
    trainer_exc: Optional[BaseException] = None
    try:
        step = trainer.fit(loader)
    except BaseException as exc:  # noqa: BLE001 - re-raised below, chained w/ producer
        trainer_exc = exc
    finally:
        stop.set()  # trainer done (or failed) -> tell the producer to wind down
        thread.join(timeout=join_timeout_s)

    producer_exc = err.get("exc")
    if thread.is_alive():
        # Fail loudly rather than "succeed" while leaking a live daemon producer.
        msg = (
            f"disagg online producer did not wind down within {join_timeout_s}s of "
            "trainer exit (still alive); abandoning it would leak an active rollout"
        )
        if trainer_exc is not None:
            raise RuntimeError(msg) from trainer_exc
        raise RuntimeError(msg)
    # A producer failure usually caused the trainer failure: producer = root cause.
    if producer_exc is not None:
        if trainer_exc is not None:
            raise producer_exc from trainer_exc
        raise producer_exc
    if trainer_exc is not None:
        raise trainer_exc
    return step


def build_disagg_online_runtime(
    *,
    strategy: str = "eagle3",
    target_model,
    prompts,
    eagle3_model,
    optimizer_factory,
    feature_store: FeatureStore,
    run_id: str,
    output_dir: str,
    target_hidden_size: int,
    channel=None,
    ref_channel_path: Optional[str] = None,
    target_vocab_size: Optional[int] = None,
    draft_vocab_size: Optional[int] = None,
    target_repr: str = "logits",
    aux_hidden_state_layer_ids=None,
    vocab_map_version: Optional[str] = None,
    t2d=None,
    num_rollout_workers: int = 1,
    device: str = "cuda",
    lease: int = 8,
    in_flight_high_watermark: int = 256,
    batch_size: int = 1,
    accumulation_steps: int = 1,
    num_epochs: int = 1,
    max_steps: Optional[int] = None,
    total_steps: Optional[int] = None,
    save_interval: int = 0,
    eval_interval: int = 0,
    tp_size: int = 1,
    sp_ulysses_size: int = 1,
    sp_ring_size: int = 1,
    collate_fn=None,
    idle_timeout_s: Optional[float] = None,
    metadata_store: Optional[MetadataStore] = None,
    metadata_db_path: Optional[str] = None,
    resume: bool = False,
    join_timeout_s: Optional[float] = 30.0,
    logger=None,
    log_interval: int = 50,
    resume_from: Optional[str] = None,
    max_checkpoints: int = 0,
):
    """One-process online disaggregated runtime (O1.2).

    Producer + consumer over ONE shared metadata store, ONE consume-once
    ``feature_store``, and one streaming-ref path; returns ``(trainer, loader,
    run)`` where ``run()`` drives both concurrently. Pass ``channel`` or
    ``ref_channel_path``; pass ``metadata_store``/``metadata_db_path`` for a
    durable, restart-reconcilable run (``resume=True`` + ``resume_from`` — see
    :func:`build_disagg_online_consumer`).
    """
    from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel

    if resume_from and not resume:
        raise ValueError(
            "resume_from is set but resume is not: online resume requires both "
            "knobs (see build_disagg_online_consumer)"
        )
    if channel is not None:
        producer_channel = channel
        path = channel.path
    elif ref_channel_path is not None:
        path = ref_channel_path
        producer_channel = StreamingRefChannel(path)
    else:
        raise ValueError("provide either `channel` or `ref_channel_path`")
    # Producer writes / consumer reads through SEPARATE channel views over one
    # path (each holds its own offset) — the cross-process split, in one process.
    consumer_channel = StreamingRefChannel(path)

    # Both halves must share ONE store instance: commits and acks land together.
    shared_store = metadata_store
    if shared_store is None and metadata_db_path is None:
        shared_store = InMemoryMetadataStore()

    _workers, drive_producer = build_disagg_online_producer(
        strategy=strategy,
        target_model=target_model,
        prompts=prompts,
        feature_store=feature_store,
        channel=producer_channel,
        run_id=run_id,
        target_hidden_size=target_hidden_size,
        target_vocab_size=target_vocab_size,
        draft_vocab_size=draft_vocab_size,
        target_repr=target_repr,
        aux_hidden_state_layer_ids=aux_hidden_state_layer_ids,
        vocab_map_version=vocab_map_version,
        t2d=t2d,
        num_rollout_workers=num_rollout_workers,
        device=device,
        lease=lease,
        in_flight_high_watermark=in_flight_high_watermark,
        metadata_store=shared_store,
        metadata_db_path=metadata_db_path,
        prompt_epochs=num_epochs,
    )
    trainer, loader = build_disagg_online_consumer(
        strategy=strategy,
        feature_store=feature_store,
        channel=consumer_channel,
        eagle3_model=eagle3_model,
        optimizer_factory=optimizer_factory,
        run_id=run_id,
        output_dir=output_dir,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        num_epochs=1,
        max_steps=max_steps,
        total_steps=total_steps,
        save_interval=save_interval,
        eval_interval=eval_interval,
        tp_size=tp_size,
        sp_ulysses_size=sp_ulysses_size,
        sp_ring_size=sp_ring_size,
        collate_fn=collate_fn,
        idle_timeout_s=idle_timeout_s,
        metadata_store=shared_store,
        metadata_db_path=metadata_db_path,
        resume=resume,
        logger=logger,
        log_interval=log_interval,
        resume_from=resume_from,
        max_checkpoints=max_checkpoints,
    )

    def run() -> int:
        return run_disagg_online_interleaved(
            trainer=trainer,
            loader=loader,
            drive_producer=drive_producer,
            channel=producer_channel,
            join_timeout_s=join_timeout_s,
        )

    return trainer, loader, run


# Back-compat aliases: the model used to live in the function name; it is now
# the ``strategy=`` parameter, so these are plain aliases.

build_offline_eagle3_runtime = build_offline_runtime
build_offline_eagle3_controller = build_offline_runtime
build_disagg_eagle3_runtime = build_disagg_offline_runtime
build_online_eagle3_runtime = build_online_runtime
build_disagg_online_eagle3_runtime = build_disagg_online_runtime


__all__ = [
    # strategy-neutral (preferred)
    "build_offline_runtime",
    "build_disagg_offline_runtime",
    "build_online_runtime",
    "build_disagg_online_producer",
    "build_disagg_online_consumer",
    "build_disagg_online_runtime",
    "run_disagg_online_interleaved",
    # back-compat aliases
    "build_offline_eagle3_runtime",
    "build_offline_eagle3_controller",
    "build_disagg_eagle3_runtime",
    "build_online_eagle3_runtime",
    "build_disagg_online_eagle3_runtime",
]
