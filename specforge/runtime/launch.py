# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Launch helpers that wire the DataFlow runtime from a RunConfig.

The training *script* becomes a thin launcher: it parses args, calls one of
these builders, and runs ``TrainerController.fit``. All training logic lives in
the runtime components, not the script.

Two orthogonal axes shape a run, and they are deliberately kept separate:

* **topology / data-source** — offline manifest vs online rollout, colocated vs
  disaggregated (cross-process or one-process). This is what the *named* builders
  below span; each carries genuinely different control flow (producer vs consumer
  process, drain-then-fit vs streamed vs interleaved) and so earns its function.
* **draft model / strategy** — eagle3 vs dflash vs ... This is NOT a per-builder
  axis. Every builder takes ``strategy=`` and resolves a :class:`StrategySpec`
  (``specforge.runtime.training.registry``) for the model-specific bits — the
  per-step strategy, the required-feature contract, the offline reader/transform/
  collate, the online collate, the capture adapter. Adding a model is a spec entry
  beside its model code, not a new ``build_*`` family, so this file does not grow
  as (topologies x models).

The shared spine every path funnels through:

    ref source (OfflineManifestReader | RolloutWorker | StreamingRefQueue)
        -> DataFlowController -> FeatureDataLoader(transform, collate)
        -> TrainBatch -> spec.make_strategy -> TrainerCore/Controller -> FSDP
"""

from __future__ import annotations

from typing import List, Optional

from specforge.runtime.contracts import SampleRef
from specforge.runtime.control_plane import DataFlowController, resolve_control_plane
from specforge.runtime.control_plane.metadata_store import (
    InMemoryMetadataStore,
    MetadataStore,
    SQLiteMetadataStore,
)
from specforge.runtime.data_plane import FeatureStore, LocalFeatureStore
from specforge.runtime.training.registry import StrategySpec, resolve_strategy

# The trainer/loader assembly (FeatureDataLoader + FSDPTrainingBackend +
# TrainerCore + TrainerController) now lives in the domain ``Trainer``
# (``specforge.training``); ``_assemble_trainer`` below delegates to it.

# ---------------------------------------------------------------------------
# Shared assemblers — strategy- and topology-agnostic. Every builder is a thin
# wrapper that resolves a spec, obtains a (store, ref source), and calls these.
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
):
    """The trainer+loader assembly shared by offline / disagg / online / interleaved.

    Only the (store, ref source, collate/transform, target_head) differ between
    topologies; the FSDP wrap, optimizer-after-wrap, per-step strategy
    (``spec.make_strategy``), ``TrainerCore`` and ``TrainerController`` — including
    the optimizer-step ack — are identical. ``optimizer_factory`` runs AFTER
    FSDP-wrap, over the wrapped module's inner draft.
    """
    # Delegates to the domain Trainer (``specforge.training``) — the canonical
    # assembler for this seam since Phase B3. It performs the exact composition
    # this function used to inline; we return the same
    # (TrainerController, FeatureDataLoader) tuple so every build_* path is
    # unchanged. New code can build a ``Trainer`` directly and call ``.fit()``.
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
    )
    return trainer.controller, trainer.loader


def _offline_io(spec: StrategySpec, max_len: int):
    """Resolve (collate_fn, per_sample_transform) for an offline-shaped run.

    Raises if the strategy has no offline data path wired yet.
    """
    if spec.make_offline_collate is None or spec.make_offline_transform is None:
        raise NotImplementedError(
            f"offline data path for strategy {spec.name!r} is not wired yet: its "
            f"StrategySpec needs make_offline_transform + make_offline_collate "
            f"(DFlash/Domino use their own feature schema). See "
            f"specforge.runtime.training.registry."
        )
    return spec.make_offline_collate(), spec.make_offline_transform(max_len)


def _online_collate(spec: StrategySpec, collate_fn):
    """Resolve the online/streamed collate, or raise if the strategy has none wired.

    Mirrors ``_offline_io``: a strategy registered ``supports_online`` but with no
    ``make_online_collate`` (and no explicit ``collate_fn``) fails with an
    actionable error instead of ``TypeError: 'NoneType' object is not callable``.
    """
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
    """Pick the controller's metadata store for an online disaggregated process.

    O1.1: producer and consumer run in *separate* processes, so a shared, durable
    store is what makes commit/dedup/ack cross-process (an in-process
    ``InMemoryMetadataStore`` per process shares nothing). Pass an explicit
    ``metadata_store``, or a ``metadata_db_path`` and both processes open a
    ``SQLiteMetadataStore`` over the same file. ``None`` keeps the pre-O1.1
    behaviour: a private in-process store, control state un-shared.
    """
    if metadata_store is not None:
        return metadata_store
    if metadata_db_path is not None:
        return SQLiteMetadataStore(metadata_db_path)
    return None


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
    """Build the rollout producer workers shared by colocated-online and the
    disaggregated-online producer.

    The capture contract is derived from ``spec.required_features`` and verified
    at the ``FeatureStore.put`` boundary (``verify_capture``). The adapter is
    ``spec.make_adapter`` (default ``SGLangAdapter`` for eagle3). Strategies
    without an online capture path (``supports_online=False``) raise here rather
    than quietly emitting the wrong feature schema.
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
    # strategy=spec.name so committed SampleRefs carry the right strategy tag and
    # the loader's strategy check passes (the RolloutWorker default is "eagle3").
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
# Offline (colocated + disaggregated). The named builders span the topology
# axis; `strategy=` selects the model.
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
):
    """Assemble the colocated offline dataflow (``LocalFeatureStore``).

    The model object is passed as ``eagle3_model`` for backward compatibility; it
    is really "the composite draft model for ``strategy``" (it must expose an
    inner ``.draft_model`` for the optimizer target).
    """
    spec = resolve_strategy(strategy)
    if spec.make_offline_reader is None:
        raise NotImplementedError(
            f"offline data path for strategy {spec.name!r} is not wired yet: its "
            f"StrategySpec needs make_offline_reader. See "
            f"specforge.runtime.training.registry."
        )
    collate_fn, per_sample_transform = _offline_io(spec, max_len)
    controller, durable_ack = resolve_control_plane("local_colocated", run_id)
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
):
    """Consumer side of a disaggregated OFFLINE run.

    Trains from a caller-supplied ``feature_store`` (tensors produced by a
    *different process* — typically a :class:`SharedDirFeatureStore` on a shared
    mount) and ``disagg://`` ``SampleRef``s the producer published to the manifest.
    The trainer assembly is identical to the colocated offline path — only the
    (store, refs) source differs — so results match within determinism tolerance.
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
):
    """Assemble the colocated online dataflow and return
    ``(trainer, loader, workers, controller, drive_rollout)``.

    Mirror of :func:`build_offline_runtime`; the only difference is the *producer*
    of ``SampleRef``s: a ``RolloutWorker`` leases ``PromptTask``s, asks the
    ``target_model`` (any backend exposing the strategy's ``generate_*_data`` — HF,
    SGLang, or custom; **sglang is not required**) for per-sample features via the
    strategy's adapter, writes them to a ``mem://`` ``FeatureStore``, and commits
    ``SampleRef``s onto the controller's ``SampleRefQueue``. From ``SampleRef``
    down the path is identical to offline.

    The returned ``drive_rollout()`` runs the workers until the prompt pool drains,
    populating the queue the loader consumes; the launcher calls it before
    ``trainer.fit(loader)``. ``target_head`` is ``None`` on purpose: online rollout
    already materialized the target distribution.
    """
    spec = resolve_strategy(strategy)
    controller, durable_ack = resolve_control_plane("local_colocated", run_id)
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

    Mirrors the producer half of :func:`build_online_runtime`, but the
    RolloutWorkers put() into a cross-node ``feature_store`` (a consume-once
    :class:`MooncakeFeatureStore`) and committed SampleRefs are streamed to the
    consumer through a :class:`StreamingRefChannel` instead of an in-process queue.

    ``metadata_store`` / ``metadata_db_path`` (O1.1) point the controller at the
    *shared, durable* store the consumer also opens, so the producer's
    ``commit_samples`` lands where the consumer reconciles on restart.

    Returns ``(workers, drive_producer)``. ``drive_producer(should_stop=...)`` runs
    the workers until the prompt pool drains, publishing refs to the channel and
    applying backpressure (pauses while ``channel.in_flight_remote()`` exceeds
    ``in_flight_high_watermark``), then closes the channel so the consumer's loader
    terminates. ``should_stop`` lets the interleaved driver wind the producer down
    once the trainer hits ``max_steps``. The channel is always closed on exit.
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
                # backpressure: don't let the producer outrun the consumer (and
                # the Mooncake segment). in_flight = published - consumer-acked.
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
):
    """Consumer (trainer) side of an ONLINE disaggregated run.

    Trains from a streamed ``SampleRef`` channel + a consume-once
    ``feature_store`` produced by a *different pool*. The trainer assembly is the
    online one (``target_head=None``); refs arrive from a :class:`StreamingRefQueue`
    over the channel rather than the in-process ``SampleRefQueue``, and features
    are fetched cross-node from Mooncake.

    ``metadata_store`` / ``metadata_db_path`` (O1.1) point the controller at the
    *shared, durable* store the producer commits to. ``resume=True`` reconciles
    against that store before training: already-durably-trained samples are handed
    to the queue as ``skip_ids`` (dropped on the channel re-read); the
    committed-but-unacked tail re-streams and re-trains. Requires a durable store.
    """
    from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefQueue

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
        # Released == durably acked AND optimizer-step committed: the samples
        # already trained on a prior run. Skip exactly those on the channel
        # re-read; the committed-unacked tail re-streams and re-trains.
        reconciled = controller.reconcile_on_restart(feature_store)
        skip_ids = set(reconciled["released"])

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

    The producer streams refs on a background thread while ``trainer.fit`` consumes
    them on this thread. The consumer's :class:`StreamingRefQueue` blocks until the
    channel is closed-and-drained, so the trainer tracks the producer instead of
    ending the instant the stream is momentarily empty.

    Shutdown is symmetric and hang-free: trainer finishes first -> ``should_stop``
    is set so the producer stops generating; producer finishes first -> it closes
    the channel so the loader drains the tail and ``fit`` returns; producer raises
    -> the channel is closed so the consumer cannot hang, and the exception is
    re-raised here once ``fit`` has unwound. Returns the trainer's final step.
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
        # The producer overran join_timeout_s and is still running: a daemon thread
        # that would keep publishing into a store no consumer drains. Fail loudly
        # instead of returning "success" with a leaked, still-live producer.
        msg = (
            f"disagg online producer did not wind down within {join_timeout_s}s of "
            "trainer exit (still alive); abandoning it would leak an active rollout"
        )
        if trainer_exc is not None:
            raise RuntimeError(msg) from trainer_exc
        raise RuntimeError(msg)
    # A producer failure closes the channel, which is usually what makes trainer.fit
    # fail downstream, so surface the producer exception as the root cause and chain
    # the trainer error so neither is silently lost.
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
):
    """One-process online disaggregated runtime (O1.2).

    Wires the producer (rollout pool) and the consumer (FSDP trainer) over ONE
    shared metadata store, ONE consume-once ``feature_store``, and ONE streaming-ref
    channel (two :class:`StreamingRefChannel` views over the same path), and returns
    ``(trainer, loader, run)``. Calling ``run()`` drives both concurrently
    (:func:`run_disagg_online_interleaved`). Pass a ``channel`` or a
    ``ref_channel_path``. The metadata store defaults to a shared in-process store;
    pass ``metadata_store`` / ``metadata_db_path`` for a durable, restart-reconcilable
    run (``resume=True`` then skips already-trained refs on the channel re-read).
    """
    from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel

    if channel is not None:
        producer_channel = channel
        path = channel.path
    elif ref_channel_path is not None:
        path = ref_channel_path
        producer_channel = StreamingRefChannel(path)
    else:
        raise ValueError("provide either `channel` or `ref_channel_path`")
    # The producer writes / the consumer reads through SEPARATE channel views over
    # the same path (each holds its own read/write offset) — the same split the
    # cross-process disagg path uses, here colocated in one process.
    consumer_channel = StreamingRefChannel(path)

    # One process: share a single metadata store instance across both halves so
    # the producer's commits and the consumer's acks land in the same store.
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


# ---------------------------------------------------------------------------
# Backward-compatible names. The model used to live in the function name; it is
# now the ``strategy=`` parameter (default "eagle3"), so these are plain aliases
# of the strategy-neutral builders above. Existing callers are unchanged.
# ---------------------------------------------------------------------------

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
