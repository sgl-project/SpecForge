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
        per_sample_transform=lambda raw: OfflineEagle3Dataset.process_data(raw, max_len),
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


# Backward-compatible alias for early branch users.
build_offline_eagle3_controller = build_offline_eagle3_runtime


__all__ = [
    "build_offline_eagle3_controller",
    "build_offline_eagle3_runtime",
    "build_disagg_eagle3_runtime",
    "build_online_eagle3_runtime",
]
