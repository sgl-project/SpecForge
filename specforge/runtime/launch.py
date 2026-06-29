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
the runtime components, not the script. This module wires the **offline EAGLE3**
path end to end:

    OfflineManifestReader -> DataFlowController -> SampleRefQueue
        -> FeatureDataLoader(process_data, DataCollatorWithPadding)
        -> TrainBatch -> Eagle3TrainStrategy -> TrainerCore/Controller -> FSDP

Online wiring (RolloutWorker + SGLangAdapter) composes the same control/data
plane; see ``inference/`` for the equivalent assembly.
"""

from __future__ import annotations

from typing import List, Optional

from specforge.runtime.contracts import SampleRef
from specforge.runtime.control_plane import DataFlowController
from specforge.runtime.control_plane.metadata_store import (
    InMemoryMetadataStore,
    MetadataStore,
    SQLiteMetadataStore,
)
from specforge.runtime.data_plane import (
    FeatureDataLoader,
    FeatureStore,
    LocalFeatureStore,
    OfflineManifestReader,
)
from specforge.runtime.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.runtime.training.strategy import Eagle3TrainStrategy
from specforge.runtime.training.trainer import TrainerController, TrainerCore


def _assemble_offline_eagle3(
    *,
    controller: DataFlowController,
    store: FeatureStore,
    refs: List[SampleRef],
    eagle3_model,
    target_head,
    optimizer_factory,
    run_id: str,
    output_dir: str,
    max_len: int,
    batch_size: int,
    accumulation_steps: int,
    num_epochs: int,
    max_steps: Optional[int],
    save_interval: int,
    eval_interval: int,
    tp_size: int,
    sp_ulysses_size: int,
    sp_ring_size: int,
    logger,
    log_interval: int,
):
    """Shared trainer/loader assembly for the offline-shaped EAGLE3 dataflow.

    Identical for the colocated (``LocalFeatureStore``) and disaggregated
    (``SharedDirFeatureStore``) paths — only the (store, refs) source differs, so
    both produce byte-identical batches and training. ``optimizer_factory`` runs
    AFTER FSDP-wrap, over the wrapped module's inner draft.
    """
    from specforge.data.preprocessing import OfflineEagle3Dataset
    from specforge.data.utils import DataCollatorWithPadding

    controller.enqueue_offline_refs(refs)  # record committed state (enables ack lookup)
    trainer_id = controller.register_trainer({"role": "trainer", "run_id": run_id})
    # Offline = a fixed, re-iterable ref set (so num_epochs > 1 actually trains
    # multiple epochs). The trainer acks at the optimizer-step boundary via ack_fn.
    loader = FeatureDataLoader(
        store,
        refs=refs,
        batch_size=batch_size,
        collate_fn=DataCollatorWithPadding(),
        per_sample_transform=lambda raw: OfflineEagle3Dataset.process_data(
            raw, max_len
        ),
        drop_last=True,
        strategy="eagle3",
    )

    parallel = ParallelConfig.from_distributed(
        tp_size=tp_size, sp_ulysses_size=sp_ulysses_size, sp_ring_size=sp_ring_size
    )
    backend = FSDPTrainingBackend(parallel, optimizer_factory=optimizer_factory)
    # FSDP-wrap the composite model and build the optimizer over the inner draft
    # AFTER wrapping; the strategy MUST run forward through the wrapped module so
    # FSDP is actually in the forward/backward path (not bypassed at >1 rank).
    wrapped = backend.prepare_model(
        eagle3_model, optimizer_target=eagle3_model.draft_model
    )
    strategy = Eagle3TrainStrategy(wrapped, target_head=target_head)
    core = TrainerCore(strategy, backend, accumulation_steps=accumulation_steps)
    trainer = TrainerController(
        core,
        run_id=run_id,
        output_dir=output_dir,
        num_epochs=num_epochs,
        max_steps=max_steps,
        save_interval=save_interval,
        eval_interval=eval_interval,
        log_interval=log_interval,
        logger=logger,
        ack_fn=lambda ids, step: controller.ack_train_refs(
            trainer_id, ids, global_step=step, optimizer_durable=True
        ),
    )
    return trainer, loader


def build_offline_eagle3_runtime(
    *,
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
    save_interval: int = 0,
    eval_interval: int = 0,
    tp_size: int = 1,
    sp_ulysses_size: int = 1,
    sp_ring_size: int = 1,
    logger=None,
    log_interval: int = 50,
):
    """Assemble the offline-EAGLE3 dataflow (colocated ``LocalFeatureStore``)."""
    controller = DataFlowController(run_id)
    refs = OfflineManifestReader(
        hidden_states_path,
        run_id=run_id,
        ttt_length=ttt_length,
        max_len=max_len,
        target_repr="hidden_state",
    ).read()
    store = LocalFeatureStore(run_id)
    return _assemble_offline_eagle3(
        controller=controller,
        store=store,
        refs=refs,
        eagle3_model=eagle3_model,
        target_head=target_head,
        optimizer_factory=optimizer_factory,
        run_id=run_id,
        output_dir=output_dir,
        max_len=max_len,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        num_epochs=num_epochs,
        max_steps=max_steps,
        save_interval=save_interval,
        eval_interval=eval_interval,
        tp_size=tp_size,
        sp_ulysses_size=sp_ulysses_size,
        sp_ring_size=sp_ring_size,
        logger=logger,
        log_interval=log_interval,
    )


def build_disagg_eagle3_runtime(
    *,
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
    save_interval: int = 0,
    eval_interval: int = 0,
    tp_size: int = 1,
    sp_ulysses_size: int = 1,
    sp_ring_size: int = 1,
    logger=None,
    log_interval: int = 50,
):
    """Consumer side of a disaggregated EAGLE3 run.

    Trains from a ``feature_store`` whose tensors were produced by a *different
    process* (the rollout/ingest pool) on a shared mount — typically a
    :class:`SharedDirFeatureStore`. ``refs`` are the ``disagg://`` ``SampleRef``s
    the producer published to the manifest
    (:func:`data_plane.disagg_ingest.read_ref_manifest`). The trainer assembly is
    identical to the colocated offline path, so results match within determinism
    tolerance — the only difference is where the feature tensors live.
    """
    controller = DataFlowController(run_id)
    return _assemble_offline_eagle3(
        controller=controller,
        store=feature_store,
        refs=refs,
        eagle3_model=eagle3_model,
        target_head=target_head,
        optimizer_factory=optimizer_factory,
        run_id=run_id,
        output_dir=output_dir,
        max_len=max_len,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        num_epochs=num_epochs,
        max_steps=max_steps,
        save_interval=save_interval,
        eval_interval=eval_interval,
        tp_size=tp_size,
        sp_ulysses_size=sp_ulysses_size,
        sp_ring_size=sp_ring_size,
        logger=logger,
        log_interval=log_interval,
    )


def build_online_eagle3_runtime(
    *,
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
    save_interval: int = 0,
    eval_interval: int = 0,
    tp_size: int = 1,
    sp_ulysses_size: int = 1,
    sp_ring_size: int = 1,
    collate_fn=None,
    logger=None,
):
    """Assemble the online-EAGLE3 dataflow and return
    ``(trainer, loader, workers, controller, drive_rollout)``.

    Mirror of :func:`build_offline_eagle3_runtime`; the only difference is the
    *producer* of ``SampleRef``s. Instead of an ``OfflineManifestReader`` reading
    ``.ckpt`` files, a ``RolloutWorker`` leases ``PromptTask``s, asks the
    ``target_model`` (any backend exposing ``generate_eagle3_data`` — HF, SGLang,
    or custom; **sglang is not required**) for per-sample features via
    ``SGLangAdapter``, writes them to the ``mem://`` ``FeatureStore``, and commits
    ``SampleRef``s onto the controller's ``SampleRefQueue``. From ``SampleRef``
    down (loader -> strategy -> trainer) the code path is identical to offline.

    ``prompts`` is the metadata-only PromptTask source (e.g.
    ``[{"payload": {"input_ids": [...], "loss_mask": [...]}}]``). The returned
    ``drive_rollout()`` runs the workers until the prompt pool is exhausted,
    populating the queue the loader consumes; the launcher script calls it before
    ``trainer.fit(loader)``. (Fully-async rollout/train interleaving with
    backpressure is the control-plane's job — a follow-up, not this seam.)

    ``target_head`` is ``None`` on purpose: online rollout already materialized the
    ``target`` distribution, so the strategy consumes it directly rather than
    re-running an lm-head (that is the offline ``hidden_state`` path's job).
    """
    import torch

    from specforge.runtime.inference.capture import CaptureConfig
    from specforge.runtime.inference.rollout_worker import RolloutWorker
    from specforge.runtime.inference.sglang_adapter import SGLangAdapter

    controller = DataFlowController(run_id)
    controller.ingest_prompts(prompts)
    # PR8 colocated store has no residency cap (max_resident_bytes is the M5
    # backpressure follow-up); mirror the offline launcher's plain construction.
    store = LocalFeatureStore(run_id)

    if aux_hidden_state_layer_ids is None:
        aux_hidden_state_layer_ids = tuple(
            getattr(target_model, "aux_hidden_states_layers", ()) or ()
        )

    adapter = SGLangAdapter(target_model, device=device, t2d=t2d)
    capture = CaptureConfig.from_strategy(
        required_features=Eagle3TrainStrategy.required_features,
        aux_hidden_state_layer_ids=tuple(aux_hidden_state_layer_ids),
        target_repr=target_repr,
        target_hidden_size=target_hidden_size,
        target_vocab_size=target_vocab_size,
        draft_vocab_size=draft_vocab_size,
        vocab_map_version=vocab_map_version,
    )
    workers = [
        RolloutWorker(
            controller,
            store,
            adapter,
            capture,
            run_id=run_id,
            worker_id=f"rollout-{i}",
        )
        for i in range(num_rollout_workers)
    ]

    # Queue mode (online consume-once stream). Online features arrive from the
    # adapter already in train form (input_ids/attention_mask/loss_mask/
    # hidden_state/target), so there is no per_sample_transform (unlike offline).
    def _cat_collate(feats):
        # Concatenate per-sample features along the batch dim. The offline
        # ``DataCollatorWithPadding`` assumes 2D (B,n) inputs and would choke on
        # the 3D hidden_state/target tensors; online features are pre-formed, so
        # a plain cat is correct for equal-length / batch_size=1 batches (the
        # adapter already groups equal-length prompts). Variable-length padded
        # batching is a follow-up; pass ``collate_fn`` to override.
        return {k: torch.cat([f[k] for f in feats], dim=0) for k in feats[0]}

    loader = FeatureDataLoader(
        store,
        controller.sample_queue,
        batch_size=batch_size,
        collate_fn=collate_fn or _cat_collate,
        drop_last=True,
        strategy="eagle3",
    )

    parallel = ParallelConfig.from_distributed(
        tp_size=tp_size, sp_ulysses_size=sp_ulysses_size, sp_ring_size=sp_ring_size
    )
    backend = FSDPTrainingBackend(parallel, optimizer_factory=optimizer_factory)
    wrapped = backend.prepare_model(
        eagle3_model, optimizer_target=eagle3_model.draft_model
    )
    strategy = Eagle3TrainStrategy(wrapped, target_head=None)
    core = TrainerCore(strategy, backend, accumulation_steps=accumulation_steps)
    trainer_id = controller.register_trainer({"role": "trainer", "run_id": run_id})
    trainer = TrainerController(
        core,
        run_id=run_id,
        output_dir=output_dir,
        num_epochs=num_epochs,
        max_steps=max_steps,
        save_interval=save_interval,
        eval_interval=eval_interval,
        logger=logger,
        ack_fn=lambda ids, step: controller.ack_train_refs(
            trainer_id, ids, global_step=step, optimizer_durable=True
        ),
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


def _online_cat_collate(feats):
    """Concatenate pre-formed online features along the batch dim (see
    build_online_eagle3_runtime); online features are already in train form, so a
    plain cat is correct for equal-length / batch_size=1 batches."""
    import torch

    return {k: torch.cat([f[k] for f in feats], dim=0) for k in feats[0]}


def _resolve_metadata_store(
    metadata_store: Optional[MetadataStore],
    metadata_db_path: Optional[str],
) -> Optional[MetadataStore]:
    """Pick the controller's metadata store for an online disaggregated process.

    O1.1: the producer and consumer run in *separate* processes, so a shared,
    durable store is what makes commit/dedup/ack cross-process (an in-process
    ``InMemoryMetadataStore`` per process shares nothing). Pass an explicit
    ``metadata_store`` (e.g. a future ``RedisMetadataStore`` for multi-node O2),
    or a ``metadata_db_path`` and both processes open a ``SQLiteMetadataStore``
    over the same file (enough for single-host O1; SQLite WAL serializes the
    producer's commits against the consumer's ack transaction). ``None`` keeps the
    pre-O1.1 behaviour: a private in-process store, control state un-shared.
    """
    if metadata_store is not None:
        return metadata_store
    if metadata_db_path is not None:
        return SQLiteMetadataStore(metadata_db_path)
    return None


def build_disagg_online_producer(
    *,
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

    Mirrors the producer half of :func:`build_online_eagle3_runtime`, but the
    RolloutWorkers put() into a cross-node ``feature_store`` (a consume-once
    :class:`MooncakeFeatureStore`) and the committed SampleRefs are streamed to
    the consumer through a :class:`StreamingRefChannel` instead of an in-process
    queue. The feature tensors never touch a shared mount.

    ``metadata_store`` / ``metadata_db_path`` (O1.1) point the controller at the
    *shared, durable* metadata store the consumer also opens, so the producer's
    ``commit_samples`` (dedup, at-least-once) lands in the same store the consumer
    reconciles against on restart. Omit both for the pre-O1.1 in-process store
    (commit/ack state stays private to this process). See
    :func:`_resolve_metadata_store`.

    Returns ``(workers, drive_producer)``. ``drive_producer(should_stop=...)`` runs
    the workers until the prompt pool drains, publishing refs to the channel and
    applying backpressure (it pauses while ``channel.in_flight_remote()`` exceeds
    ``in_flight_high_watermark`` so a lagging trainer can't overrun the Mooncake
    segment), then closes the channel so the consumer's loader terminates.
    ``should_stop`` (a zero-arg predicate) lets a caller wind the producer down
    early — e.g. the interleaved driver sets it once the trainer hits ``max_steps``
    so the producer doesn't block forever on the watermark after the consumer
    stops draining (O1.2). The channel is always closed on exit so the consumer
    never hangs on a finished producer.
    """
    import time

    from specforge.runtime.inference.capture import CaptureConfig
    from specforge.runtime.inference.rollout_worker import RolloutWorker
    from specforge.runtime.inference.sglang_adapter import SGLangAdapter

    sleep = sleep or time.sleep
    controller = DataFlowController(
        run_id,
        metadata_store=_resolve_metadata_store(metadata_store, metadata_db_path),
    )
    controller.ingest_prompts(prompts)

    if aux_hidden_state_layer_ids is None:
        aux_hidden_state_layer_ids = tuple(
            getattr(target_model, "aux_hidden_states_layers", ()) or ()
        )
    adapter = SGLangAdapter(target_model, device=device, t2d=t2d)
    capture = CaptureConfig.from_strategy(
        required_features=Eagle3TrainStrategy.required_features,
        aux_hidden_state_layer_ids=tuple(aux_hidden_state_layer_ids),
        target_repr=target_repr,
        target_hidden_size=target_hidden_size,
        target_vocab_size=target_vocab_size,
        draft_vocab_size=draft_vocab_size,
        vocab_map_version=vocab_map_version,
    )
    workers = [
        RolloutWorker(
            controller,
            feature_store,
            adapter,
            capture,
            run_id=run_id,
            worker_id=f"rollout-{i}",
        )
        for i in range(num_rollout_workers)
    ]

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
    online one (``target_head=None``: rollout already materialized the target
    distribution); the only difference from colocated online is that refs arrive
    from a :class:`StreamingRefQueue` over the channel rather than the in-process
    ``SampleRefQueue``, and the features are fetched cross-node from Mooncake. The
    loader frees each sample on read (consume-once) and acks the channel (the
    producer's backpressure signal).

    ``metadata_store`` / ``metadata_db_path`` (O1.1) point the controller at the
    *shared, durable* store the producer commits to, so the per-optimizer-step
    ack transaction (``ack_train_refs``) is recorded against the same committed
    set the producer wrote — the precondition for a correct restart.

    ``resume=True`` reconciles against that durable store before training: the
    channel JSONL is append-only and a restarted consumer re-reads it from the
    start, so ``reconcile_on_restart`` derives the already-durably-trained samples
    and they are handed to the :class:`StreamingRefQueue` as ``skip_ids`` (dropped
    on re-read, no duplicate train). The committed-but-unacked tail is *not*
    skipped, so it is re-trained — that requeue is realized by the channel re-read
    itself, which is why only the released set matters here. Requires a durable
    ``metadata_store`` / ``metadata_db_path`` (an in-process store has no history
    to reconcile).
    """
    from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefQueue

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
    loader = FeatureDataLoader(
        feature_store,
        queue,
        batch_size=batch_size,
        collate_fn=collate_fn or _online_cat_collate,
        drop_last=True,
        strategy="eagle3",
    )

    parallel = ParallelConfig.from_distributed(
        tp_size=tp_size, sp_ulysses_size=sp_ulysses_size, sp_ring_size=sp_ring_size
    )
    backend = FSDPTrainingBackend(parallel, optimizer_factory=optimizer_factory)
    wrapped = backend.prepare_model(
        eagle3_model, optimizer_target=eagle3_model.draft_model
    )
    strategy = Eagle3TrainStrategy(wrapped, target_head=None)
    core = TrainerCore(strategy, backend, accumulation_steps=accumulation_steps)
    trainer_id = controller.register_trainer({"role": "trainer", "run_id": run_id})
    trainer = TrainerController(
        core,
        run_id=run_id,
        output_dir=output_dir,
        num_epochs=num_epochs,
        max_steps=max_steps,
        save_interval=save_interval,
        eval_interval=eval_interval,
        log_interval=log_interval,
        logger=logger,
        ack_fn=lambda ids, step: controller.ack_train_refs(
            trainer_id, ids, global_step=step, optimizer_durable=True
        ),
    )
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

    Replaces the synchronous drain-then-fit shape (generate the whole prompt
    pool, *then* train) with a live loop: the producer streams refs on a
    background thread while ``trainer.fit`` consumes them on this thread. The
    consumer's :class:`StreamingRefQueue` blocks until the channel is
    closed-and-drained, so the trainer tracks the producer instead of ending the
    instant the stream is momentarily empty.

    Shutdown is symmetric and hang-free:

    * trainer finishes first (e.g. ``max_steps``) -> ``should_stop`` is set, so
      the producer stops generating instead of blocking on the in-flight
      watermark after the consumer quit draining;
    * producer finishes first (prompts drained) -> it closes the channel, so the
      loader drains the tail and ``fit`` returns;
    * producer raises -> the channel is closed (``drive_producer``'s ``finally``)
      so the consumer cannot hang, and the exception is re-raised here once
      ``fit`` has unwound.

    Returns the trainer's final optimizer step. Single process, in-process
    generator stub — no Ray, no live SGLang server (those are O1.3 / O2).
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


def build_disagg_online_eagle3_runtime(
    *,
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
    """One-process online disaggregated EAGLE3 runtime (O1.2).

    The single named builder the roadmap calls for: it wires the producer
    (rollout pool, in-process ``generate_eagle3_data`` stub via ``SGLangAdapter``)
    and the consumer (FSDP trainer) over ONE shared metadata store, ONE
    consume-once ``feature_store``, and ONE streaming-ref channel (two
    :class:`StreamingRefChannel` views over the same path — the proven
    producer/consumer split), and returns ``(trainer, loader, run)``. Calling
    ``run()`` drives both concurrently (:func:`run_disagg_online_interleaved`) —
    the live loop that replaces drain-then-fit. No live SGLang server and no Ray
    yet (O1.3 / O2); this proves the data + control paths live with a stubbed
    generator.

    Pass a ``channel`` or a ``ref_channel_path`` (a ``StreamingRefChannel`` is
    built over it). The metadata store defaults to a shared in-process store
    (enough for one process); pass ``metadata_store`` / ``metadata_db_path`` for a
    durable, restart-reconcilable run (``resume=True`` then skips already-trained
    refs on the channel re-read).
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
    # (A metadata_db_path instead opens one SQLite connection per half over the
    # same file — durable, and what a restart-reconcilable run needs.)
    shared_store = metadata_store
    if shared_store is None and metadata_db_path is None:
        shared_store = InMemoryMetadataStore()

    _workers, drive_producer = build_disagg_online_producer(
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


# Backward-compatible alias for early branch users.
build_offline_eagle3_controller = build_offline_eagle3_runtime


__all__ = [
    "build_offline_eagle3_controller",
    "build_offline_eagle3_runtime",
    "build_disagg_eagle3_runtime",
    "build_online_eagle3_runtime",
    "build_disagg_online_producer",
    "build_disagg_online_consumer",
    "build_disagg_online_eagle3_runtime",
    "run_disagg_online_interleaved",
]
