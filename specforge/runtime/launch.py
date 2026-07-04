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
:class:`StrategySpec` (``specforge.runtime.training.registry``) — adding a model
is a registry entry, not a new ``build_*`` family.
"""

from __future__ import annotations

from typing import List, Optional

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
from specforge.runtime.data_plane import FeatureStore, LocalFeatureStore
from specforge.runtime.training.registry import StrategySpec, resolve_strategy

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
            f"specforge.runtime.training.registry."
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
            f"See specforge.runtime.training.registry."
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
    target_model,
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
):
    """Build the rollout producer workers (colocated-online + disagg producer).

    The capture contract derives from ``spec.required_features``; strategies with
    ``supports_online=False`` raise here rather than emit the wrong feature schema.
    """
    if not spec.supports_online:
        raise NotImplementedError(
            f"online capture for strategy {spec.name!r} is not wired yet: it needs "
            f"a {spec.name} capture adapter. Set make_adapter + supports_online=True "
            f"on its StrategySpec."
        )
    from specforge.runtime.inference.capture import CaptureConfig
    from specforge.runtime.inference.rollout_worker import RolloutWorker

    if aux_hidden_state_layer_ids is None:
        aux_hidden_state_layer_ids = tuple(
            getattr(target_model, "aux_hidden_states_layers", ()) or ()
        )
    if spec.make_adapter is not None:
        adapter = spec.make_adapter(target_model, device=device, t2d=t2d)
    else:
        from specforge.runtime.inference.sglang_adapter import SGLangAdapter

        adapter = SGLangAdapter(target_model, device=device, t2d=t2d)
    capture = CaptureConfig.from_strategy(
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
            capture,
            run_id=run_id,
            worker_id=f"rollout-{i}",
            strategy=spec.name,
        )
        for i in range(num_rollout_workers)
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
            f"specforge.runtime.training.registry."
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
    resume_from: Optional[str] = None,
    max_checkpoints: int = 0,
):
    """Assemble the colocated online dataflow; return
    ``(trainer, loader, workers, controller, drive_rollout)``.

    Call ``drive_rollout()`` (runs the RolloutWorkers until the prompt pool
    drains) before ``trainer.fit(loader)``. ``target_head`` is ``None`` on
    purpose: online rollout already materialized the target distribution.
    """
    spec = resolve_strategy(strategy)
    # Keep colocated online on a private queue; durable online runs use the
    # disagg builders with a shared store and streaming ref channel.
    controller, durable_ack = build_control_plane_for_mode("local_colocated", run_id)
    controller.ingest_prompts(prompts)
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

    trainer, loader = _assemble_trainer(
        spec=spec,
        controller=controller,
        store=store,
        ref_source={"queue": controller.sample_queue},
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
        log_interval=50,
        collate_fn=_online_collate(spec, collate_fn),
        per_sample_transform=None,
        durable_ack=durable_ack,
        resume_from=resume_from,
        max_checkpoints=max_checkpoints,
    )

    def drive_rollout(max_rounds: int = 100_000) -> int:
        """Run the workers until the prompt pool drains; returns refs produced."""
        for w in workers:
            w.start()
        produced = 0
        lease = max(batch_size * 8, 8)
        for _ in range(max_rounds):
            got = sum(len(w.run_once(max_tasks=lease)) for w in workers)
            if got == 0:
                break
            produced += got
        return produced

    return trainer, loader, workers, controller, drive_rollout


def build_disagg_online_producer(
    *,
    strategy: str = "eagle3",
    target_model,
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
    lease: int = 8,
    in_flight_high_watermark: int = 256,
    backpressure_poll_s: float = 0.2,
    metadata_store: Optional[MetadataStore] = None,
    metadata_db_path: Optional[str] = None,
    sleep=None,
):
    """Producer side of an ONLINE disaggregated run (rollout pool).

    Workers put() into a cross-node ``feature_store``; committed refs stream to
    the consumer via ``channel``. ``metadata_store``/``metadata_db_path`` must be
    the same durable store the consumer opens. Returns ``(workers,
    drive_producer)``: ``drive_producer(should_stop=...)`` runs until the prompt
    pool drains, pauses above ``in_flight_high_watermark``, and always closes the
    channel on exit (EOF terminates the consumer's loader).
    """
    import time

    spec = resolve_strategy(strategy)
    sleep = sleep or time.sleep
    controller = DataFlowController(
        run_id,
        metadata_store=_resolve_metadata_store(metadata_store, metadata_db_path),
    )
    controller.ingest_prompts(prompts)

    workers = _assemble_rollout_workers(
        spec=spec,
        target_model=target_model,
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

    def drive_producer(max_rounds: int = 1_000_000, should_stop=None) -> int:
        for w in workers:
            w.start()
        produced = 0
        try:
            for _ in range(max_rounds):
                if should_stop is not None and should_stop():
                    break  # caller asked us to wind down (e.g. trainer finished)
                # backpressure: in_flight = published - consumer-acked
                while channel.in_flight_remote() >= in_flight_high_watermark:
                    if should_stop is not None and should_stop():
                        return produced  # don't block on the watermark forever
                    sleep(backpressure_poll_s)
                refs = []
                for w in workers:
                    refs.extend(w.run_once(max_tasks=lease))
                if not refs:
                    break  # prompt pool drained
                channel.publish_many(refs)
                produced += len(refs)
            return produced
        finally:
            channel.close()  # EOF -> the consumer's loader terminates once drained

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
    resume_from: Optional[str] = None,
    max_checkpoints: int = 0,
):
    """Consumer (trainer) side of an ONLINE disaggregated run.

    Trains from a streamed ref ``channel`` + a consume-once ``feature_store``
    produced by another pool; ``metadata_store``/``metadata_db_path`` must be the
    durable store the producer commits to. Online resume needs BOTH knobs:
    ``resume=True`` reconciles the durable marker (already-trained refs are
    skipped on the channel re-read) and ``resume_from`` restores the trainer
    state those acks correspond to. ``resume_from`` alone raises; a marker ahead
    of the checkpoint raises (the skipped samples' weight updates were rolled
    back — silent data loss).
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
    store = _resolve_metadata_store(metadata_store, metadata_db_path)
    controller = DataFlowController(run_id, metadata_store=store)

    skip_ids = None
    if resume:
        if store is None:
            raise ValueError(
                "resume=True needs a durable metadata_store/metadata_db_path; an "
                "in-process store has no committed/ack history to reconcile against"
            )
        # Released == durably acked AND optimizer-step committed: skip exactly
        # those on the channel re-read; the committed-unacked tail re-trains.
        reconciled = controller.reconcile_on_restart(feature_store)
        skip_ids = set(reconciled["released"])
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

    queue = StreamingRefQueue(channel, idle_timeout_s=idle_timeout_s, skip_ids=skip_ids)
    return _assemble_trainer(
        spec=spec,
        controller=controller,
        store=feature_store,
        ref_source={"queue": queue},
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
        per_sample_transform=None,
        resume_from=resume_from,
        max_checkpoints=max_checkpoints,
    )


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
        num_epochs=num_epochs,
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
