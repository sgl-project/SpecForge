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

from specforge.runtime.contracts import SampleRef
from specforge.runtime.control_plane import DataFlowController
from specforge.runtime.control_plane.metadata_store import (
    InMemoryMetadataStore,
    MetadataStore,
    NoOpMetadataStore,
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
    """Pick the online consumer's single rank-shared ledger."""
    if metadata_store is not None:
        return metadata_store
    if metadata_db_path is not None:
        return SQLiteMetadataStore(metadata_db_path)
    return None


def _dp_consumer_layout(
    dp_rank: Optional[int],
    dp_size: Optional[int],
) -> Tuple[int, int]:
    """Resolve the online consumer's (dp_rank, dp_size), defaulting from dist.

    The DP online consumer shards DATA across ranks (one inbox each), so its DP
    width is the whole trainer world.
    """
    import torch.distributed as dist

    initialized = dist.is_available() and dist.is_initialized()
    if dp_size is None:
        dp_size = dist.get_world_size() if initialized else 1
    if dp_rank is None:
        dp_rank = dist.get_rank() if initialized else 0
    if not 0 <= dp_rank < dp_size:
        raise ValueError(f"dp_rank {dp_rank} out of range for dp_size {dp_size}")
    return dp_rank, dp_size


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
    disjoint. A single source may be shared by ``num_rollout_workers`` workers.
    """
    if not spec.supports_online:
        raise NotImplementedError(
            f"online capture for strategy {spec.name!r} is not wired yet: it needs "
            f"a {spec.name} capture path. Set feature_schema + "
            f"supports_online=True on its StrategySpec."
        )
    from specforge.inference.capture import CaptureConfig
    from specforge.inference.rollout_worker import RolloutWorker

    if aux_hidden_state_layer_ids is None:
        aux_hidden_state_layer_ids = tuple(
            getattr(target_model, "capture_layers", ()) or ()
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
    else:
        from specforge.inference.adapters.policy import PolicyFeatureAdapter

        if spec.feature_schema is None:
            raise NotImplementedError(
                f"{spec.name} online capture requires a server feature source; "
                "colocated target capture is not supported"
            )
        adapters = [
            PolicyFeatureAdapter(
                target_model,
                schema=spec.feature_schema,
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
    draft_model,
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
    logger=None,
    log_interval: int = 50,
    resume_from: Optional[str] = None,
    max_checkpoints: int = 0,
    strategy_kwargs: Optional[dict] = None,
):
    """Assemble the colocated offline dataflow (``LocalFeatureStore``).

    ``draft_model`` is the composite model for ``strategy`` and must expose its
    trainable module as ``.draft_model``. Colocated offline refs are fixed and
    re-iterable, so this path does not allocate a training ledger or ref queue.
    """
    spec = resolve_strategy(strategy)
    if spec.make_offline_reader is None:
        raise NotImplementedError(
            f"offline data path for strategy {spec.name!r} is not wired yet: its "
            f"StrategySpec needs make_offline_reader. See "
            f"specforge.training.strategies.registry."
        )
    collate_fn, per_sample_transform = _offline_io(spec, max_len)
    controller = DataFlowController(
        run_id,
        metadata_store=NoOpMetadataStore(),
        enable_sample_queue=False,
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
        model=draft_model,
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
        logger=logger,
        log_interval=log_interval,
        collate_fn=collate_fn,
        strategy_kwargs=strategy_kwargs,
        per_sample_transform=per_sample_transform,
        durable_ack=False,
        resume_from=resume_from,
        max_checkpoints=max_checkpoints,
    )


def build_disagg_offline_runtime(
    *,
    strategy: str = "eagle3",
    feature_store: FeatureStore,
    refs: List[SampleRef],
    draft_model,
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
    controller = DataFlowController(
        run_id,
        metadata_store=NoOpMetadataStore(),
        enable_sample_queue=False,
    )
    return _assemble_trainer(
        spec=spec,
        controller=controller,
        store=feature_store,
        ref_source={"refs": refs},
        model=draft_model,
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
        logger=logger,
        log_interval=log_interval,
        collate_fn=collate_fn,
        strategy_kwargs=strategy_kwargs,
        per_sample_transform=per_sample_transform,
        durable_ack=False,
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
    draft_model,
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
    batch_size: int = 1,
    accumulation_steps: int = 1,
    max_steps: Optional[int] = None,
    total_steps: Optional[int] = None,
    save_interval: int = 0,
    collate_fn=None,
    logger=None,
    log_interval: int = 50,
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
    controller = DataFlowController(
        run_id,
        metadata_store=NoOpMetadataStore(),
    )
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
        model=draft_model,
        target_head=None,
        optimizer_factory=optimizer_factory,
        run_id=run_id,
        output_dir=output_dir,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        num_epochs=1,
        max_steps=max_steps,
        total_steps=total_steps,
        save_interval=save_interval,
        logger=logger,
        log_interval=log_interval,
        collate_fn=_online_collate(spec, collate_fn),
        strategy_kwargs=strategy_kwargs,
        per_sample_transform=None,
        durable_ack=False,
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
    sleep=None,
    prompt_epochs: int = 1,
):
    """Producer side of an ONLINE disaggregated run (rollout pool).

    Workers put() into a cross-node ``feature_store``; committed refs stream to
    the consumer via ``channel``. Pass ``feature_source`` for the server-capture
    transport (live SGLang server writes to the store; ``target_model`` stays
    None) — a *sequence* of sources fans out to one worker per source (the
    multi-server topology; each worker drives its own server concurrently).
    The producer has no training ledger or local ref queue; the consumer owns
    deduplication and durable acknowledgements. Returns ``(workers,
    drive_producer)``:
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
    if in_flight_high_watermark < 1:
        raise ValueError("in_flight_high_watermark must be >= 1")
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
        metadata_store=NoOpMetadataStore(),
        max_prompt_attempts=max_prompt_attempts,
        enable_sample_queue=False,
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
        quantum_wait_start = time.monotonic()
        try:
            while True:
                consumer_quantum = channel.consumer_quantum()
                if consumer_quantum is not None:
                    break
                consumer_failure = channel.consumer_failure()
                if consumer_failure is not None:
                    raise RuntimeError(
                        "consumer failed before publishing its optimizer window: "
                        f"{consumer_failure}"
                    )
                if (
                    peer_wait_timeout_s is not None
                    and time.monotonic() - quantum_wait_start > peer_wait_timeout_s
                ):
                    raise TimeoutError(
                        "producer timed out waiting for the consumer optimizer "
                        f"window after {peer_wait_timeout_s:.0f}s"
                    )
                sleep(backpressure_poll_s)
            if in_flight_high_watermark < consumer_quantum:
                raise ValueError(
                    "producer in-flight high watermark "
                    f"{in_flight_high_watermark} is smaller than the consumer's "
                    f"global optimizer-step quantum {consumer_quantum}; set "
                    "DISAGG_IN_FLIGHT_HIGH_WATERMARK to at least that value"
                )
            producer_timing(
                f"consumer optimizer window ready quantum={consumer_quantum}"
            )
        except BaseException as exc:
            try:
                channel.fail(f"{type(exc).__name__}: {exc}")
            except Exception:
                logger.exception("failed to publish producer setup failure")
            raise
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
            raise
        producer_timing(
            f"drive_producer closing channel produced={produced} "
            f"elapsed={elapsed(drive_start)}"
        )
        channel.close()  # successful EOF; a failure uses channel.fail()
        return produced

    return workers, drive_producer


def build_disagg_online_consumer(
    *,
    strategy: str = "eagle3",
    feature_store: FeatureStore,
    channel,
    draft_model,
    optimizer_factory,
    run_id: str,
    output_dir: str,
    target_head=None,
    batch_size: int = 1,
    accumulation_steps: int = 1,
    max_steps: Optional[int] = None,
    total_steps: Optional[int] = None,
    save_interval: int = 0,
    collate_fn=None,
    idle_timeout_s: Optional[float] = None,
    metadata_store: Optional[MetadataStore] = None,
    metadata_db_path: Optional[str] = None,
    logger=None,
    log_interval: int = 50,
    strategy_kwargs: Optional[dict] = None,
    max_checkpoints: int = 0,
    dp_rank: Optional[int] = None,
    dp_size: Optional[int] = None,
    inbox_dir: Optional[str] = None,
):
    """Consumer (trainer) side of an ONLINE disaggregated run.

    Rank 0 always runs the :class:`RefDistributor`, including for ``dp_size=1``.
    It is the only reader of ``channel`` and the only writer to the fresh
    ``metadata_store``/``metadata_db_path`` ledger, then dispatches refs into one
    inbox per rank. Every rank consumes the same inbox-based path and durable
    acknowledgements gather to rank 0 through :class:`DPAckController`.
    """
    import torch.distributed as dist

    from specforge.runtime.control_plane.dp_ack import DPAckController
    from specforge.runtime.data_plane.ref_distributor import (
        InboxChannel,
        RefDistributor,
    )
    from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefQueue

    spec = resolve_strategy(strategy)
    distributed = dist.is_available() and dist.is_initialized()
    world = dist.get_world_size() if distributed else 1
    actual_rank = dist.get_rank() if distributed else 0
    preflight_exc = None
    try:
        dp_rank, dp_size = _dp_consumer_layout(dp_rank, dp_size)
        if metadata_store is None and metadata_db_path is None:
            raise ValueError(
                "online consumer needs a fresh metadata_store/metadata_db_path "
                "for its single rank-0 ledger"
            )
        if distributed and world != dp_size:
            raise ValueError(
                f"online consumer: dp_size={dp_size} but the process group has "
                f"{world} ranks — every rank must own exactly one inbox"
            )
        if distributed and actual_rank != dp_rank:
            raise ValueError(
                f"online consumer: dp_rank={dp_rank} but process-group rank is "
                f"{actual_rank}"
            )
    except BaseException as exc:
        preflight_exc = exc

    preflight_error = (
        f"{type(preflight_exc).__name__}: {preflight_exc}" if preflight_exc else None
    )
    if distributed and world > 1:
        gathered_errors = [None] * world
        dist.all_gather_object(gathered_errors, preflight_error)
        preflight_error = next((error for error in gathered_errors if error), None)
    if preflight_error is not None:
        if not distributed or world == 1:
            raise preflight_exc
        raise RuntimeError(f"online consumer preflight failed: {preflight_error}")

    if inbox_dir is None:
        inbox_dir = channel.path + ".inboxes"
    if idle_timeout_s is None:
        idle_timeout_s = 1800.0

    distributor = None
    store = None
    setup_exc = None
    if dp_rank == 0:
        try:
            store = _resolve_metadata_store(metadata_store, metadata_db_path)
            if store is None or isinstance(store, NoOpMetadataStore):
                raise ValueError("online consumer requires a retaining metadata ledger")
            committed = store.committed_count()
            if committed > 0:
                raise ValueError(
                    f"metadata store already holds {committed} committed samples; "
                    "every online attempt requires a fresh ledger"
                )
            controller = DPAckController(
                run_id, is_authority=True, metadata_store=store
            )
            distributor = RefDistributor(
                channel,
                controller,
                inbox_dir,
                dp_size,
                feature_store=feature_store,
                refs_per_rank_step=batch_size * accumulation_steps,
                idle_timeout_s=idle_timeout_s,
            )
            channel.publish_consumer_quantum(dp_size * batch_size * accumulation_steps)
        except BaseException as exc:
            setup_exc = exc

    setup_error = f"{type(setup_exc).__name__}: {setup_exc}" if setup_exc else None
    if distributed and world > 1:
        payload = [setup_error]
        dist.broadcast_object_list(payload, src=0)
        setup_error = payload[0]
    if setup_error is not None:
        if dp_rank == 0 and store is not None and hasattr(store, "close"):
            store.close()
        if not distributed or world == 1:
            raise setup_exc
        raise RuntimeError(f"online consumer rank-0 setup failed: {setup_error}")

    if dp_rank != 0:
        controller = DPAckController(
            run_id, is_authority=False, metadata_store=InMemoryMetadataStore()
        )

    # The successful rank-0 setup broadcast guarantees inbox recreation and the
    # optimizer-window sidecar are visible before any rank opens its reader.
    inbox = InboxChannel(RefDistributor.inbox_path(inbox_dir, dp_rank))
    queue = StreamingRefQueue(inbox, idle_timeout_s=idle_timeout_s)

    trainer, loader = _assemble_trainer(
        spec=spec,
        controller=controller,
        store=feature_store,
        ref_source={"queue": queue},
        model=draft_model,
        target_head=target_head if spec.uses_target_head else None,
        optimizer_factory=optimizer_factory,
        run_id=run_id,
        output_dir=output_dir,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        num_epochs=1,
        max_steps=max_steps,
        total_steps=total_steps,
        save_interval=save_interval,
        logger=logger,
        log_interval=log_interval,
        collate_fn=_online_collate(spec, collate_fn),
        strategy_kwargs=strategy_kwargs,
        per_sample_transform=None,
        max_checkpoints=max_checkpoints,
    )
    #: rank 0's RefDistributor handle (None elsewhere) — callers may
    #: stop() it after fit for a clean early (max_steps) shutdown.
    trainer.ref_distributor = distributor
    if distributor is not None:
        distributor.start()
    return trainer, loader


__all__ = [
    "build_offline_runtime",
    "build_disagg_offline_runtime",
    "build_online_runtime",
    "build_disagg_online_producer",
    "build_disagg_online_consumer",
]
