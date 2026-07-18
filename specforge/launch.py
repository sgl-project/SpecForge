# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Internal wiring helpers used by the application composition root."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from specforge.algorithms.registry import AlgorithmRegistration
from specforge.runtime.contracts import SampleRef
from specforge.runtime.control_plane import DataFlowController
from specforge.runtime.control_plane.metadata_store import (
    InMemoryMetadataStore,
    MetadataStore,
    NoOpMetadataStore,
    SQLiteMetadataStore,
)
from specforge.runtime.data_plane import (
    FeatureDataLoader,
    FeatureStore,
    LocalFeatureStore,
    drain_feature_store_removals,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared assemblers — strategy- and topology-agnostic.
# ---------------------------------------------------------------------------


def _assemble_trainer(
    *,
    algorithm: AlgorithmRegistration,
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
    eval_interval: int = 0,
    eval_data_factory=None,
    logger,
    log_interval: int,
    collate_fn,
    strategy_kwargs: Optional[Mapping[str, Any]] = None,
    per_sample_transform=None,
    durable_ack: bool = True,
    resume_from: Optional[str] = None,
    resume_state: Optional[dict] = None,
    dataset_size: Optional[int] = None,
    checkpoint_extra: Optional[dict] = None,
    max_checkpoints: int = 0,
    tp_size: int = 1,
    sp_ulysses_size: int = 1,
    sp_ring_size: int = 1,
    dataloader_num_workers: int = 0,
    profiling_options=None,
    fit_context=None,
    on_fit_success: Optional[Callable[[int], None]] = None,
    on_fit_failure: Optional[Callable[[BaseException], None]] = None,
    on_fit_finally: Optional[Callable[[], None]] = None,
):
    """Delegate to the domain ``Trainer`` (``specforge.training``) — the one
    assembly (FSDP wrap, optimizer-after-wrap, per-step strategy, loader, acks)
    shared by every trainer-bearing builder.
    """
    from specforge.algorithms.common.providers import (
        MODEL_PROVENANCE_CONTRACT_KEY,
        StepRuntimeConfig,
    )
    from specforge.training import Trainer

    if not isinstance(strategy_kwargs, StepRuntimeConfig):
        if resume_from is not None:
            raise ValueError(
                "direct builder resume requires a provider-bound "
                "StepRuntimeConfig so algorithm options and reconstructed "
                "frozen state can be validated"
            )
        strategy_kwargs = StepRuntimeConfig(
            options=strategy_kwargs or {},
            # A plain direct-builder run may train from scratch, but its
            # checkpoint intentionally cannot be resumed without a provider-
            # bound runtime contract on the subsequent call.
            resume_contract={},
            allowed_missing_checkpoint_keys=frozenset(),
        )
    elif (
        resume_from is not None
        and MODEL_PROVENANCE_CONTRACT_KEY not in strategy_kwargs.resume_contract
    ):
        raise ValueError(
            "direct builder resume requires provider-bound model provenance in "
            f"the {MODEL_PROVENANCE_CONTRACT_KEY!r} resume contract"
        )

    trainer = Trainer(
        algorithm_name=algorithm.name,
        make_step_strategy=algorithm.providers.step.build,
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
        eval_data_factory=eval_data_factory,
        logger=logger,
        log_interval=log_interval,
        collate_fn=collate_fn,
        strategy_kwargs=strategy_kwargs,
        per_sample_transform=per_sample_transform,
        durable_ack=durable_ack,
        resume_from=resume_from,
        resume_state=resume_state,
        dataset_size=dataset_size,
        checkpoint_extra=checkpoint_extra,
        max_checkpoints=max_checkpoints,
        tp_size=tp_size,
        sp_ulysses_size=sp_ulysses_size,
        sp_ring_size=sp_ring_size,
        dataloader_num_workers=dataloader_num_workers,
        profiling_options=profiling_options,
        fit_context=fit_context,
        on_fit_success=on_fit_success,
        on_fit_failure=on_fit_failure,
        on_fit_finally=on_fit_finally,
    )
    return trainer


def _offline_io(
    algorithm: AlgorithmRegistration,
    modality: str,
    max_len: int,
    *,
    ttt_length: int,
    use_usp_preprocess: bool,
):
    """Resolve the algorithm-owned normalizer and collator for one modality."""
    provider = algorithm.providers.offline_for(modality)
    return provider.build_collator(), provider.build_normalizer(
        max_len,
        ttt_length=ttt_length,
        use_usp_preprocess=use_usp_preprocess,
    )


def _shard_offline_refs(
    refs,
    *,
    use_usp_preprocess: bool,
    seed=0,
    epoch=0,
    shuffle=True,
    dp_rank=None,
    dp_size=None,
):
    """Match ``DistributedSampler`` over metadata-only refs for one epoch.

    Draft sequence-parallel peers must see the same sample, while independent
    draft-DP replicas see disjoint samples. Without USP, trainer TP is fixed at
    one and every rank is a distinct DP replica. Padding to an even number of
    refs keeps every FSDP rank on the same number of collectives. The
    caller rebuilds this plan with ``seed + epoch`` before applying a resume
    seek, exactly like the legacy ``DistributedSampler.set_epoch`` lifecycle.
    """
    import torch.distributed as dist

    refs = list(refs)
    if (dp_rank is None) != (dp_size is None):
        raise ValueError("dp_rank and dp_size must be provided together")
    if dp_rank is None:
        if dist.is_available() and dist.is_initialized():
            from specforge.distributed import get_dp_group, get_draft_dp_group

            group = get_draft_dp_group() if use_usp_preprocess else get_dp_group()
            dp_rank, dp_size = dist.get_rank(group), dist.get_world_size(group)
        else:
            dp_rank, dp_size = 0, 1
    if dp_size < 1 or not 0 <= dp_rank < dp_size:
        raise ValueError(f"invalid data-DP layout rank={dp_rank}, size={dp_size}")
    indices = _distributed_sampler_indices(
        len(refs),
        dp_rank=dp_rank,
        dp_size=dp_size,
        seed=seed,
        epoch=epoch,
        shuffle=shuffle,
    )
    return [refs[index] for index in indices]


def _distributed_sampler_indices(size, *, dp_rank, dp_size, seed, epoch, shuffle=True):
    """Reproduce ``DistributedSampler(drop_last=False)`` index generation."""
    import math

    import torch

    if size <= 0:
        return []
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed) + int(epoch))
        indices = torch.randperm(size, generator=generator).tolist()
    else:
        indices = list(range(size))

    total_size = math.ceil(size / dp_size) * dp_size
    padding_size = total_size - len(indices)
    if padding_size:
        repeats = math.ceil(padding_size / len(indices))
        indices.extend((indices * repeats)[:padding_size])
    return indices[dp_rank:total_size:dp_size]


def _make_offline_eval_data_factory(
    *,
    algorithm: AlgorithmRegistration,
    modality: str,
    hidden_states_path: str,
    run_id: str,
    batch_size: int,
    max_len: int,
    ttt_length: int,
    use_usp_preprocess: bool,
    dataloader_num_workers: int,
):
    """Build a fresh re-iterable eval loader over the offline feature path."""
    provider = algorithm.providers.offline_for(modality)
    collate_fn, per_sample_transform = _offline_io(
        algorithm,
        modality,
        max_len,
        ttt_length=ttt_length,
        use_usp_preprocess=use_usp_preprocess,
    )
    eval_run_id = f"{run_id}-eval"
    refs = provider.build_reader(
        hidden_states_path,
        run_id=eval_run_id,
        ttt_length=ttt_length,
        max_len=max_len,
    ).read()
    refs = _shard_offline_refs(
        refs, use_usp_preprocess=use_usp_preprocess, shuffle=False
    )
    store = LocalFeatureStore(eval_run_id)

    def build_loader():
        return FeatureDataLoader(
            store,
            refs=refs,
            batch_size=batch_size,
            collate_fn=collate_fn,
            per_sample_transform=per_sample_transform,
            drop_last=False,
            strategy=algorithm.name,
            num_workers=dataloader_num_workers,
        )

    return build_loader


def _streaming_collate(
    algorithm: AlgorithmRegistration,
    modality: str,
    collate_fn,
):
    """Resolve an algorithm-owned server-streaming collator."""
    if collate_fn is not None:
        return collate_fn
    return algorithm.providers.server_streaming_for(modality).build_collator()


def _resolve_metadata_store(
    metadata_store: Optional[MetadataStore],
    metadata_db_path: Optional[str],
) -> Optional[MetadataStore]:
    """Pick the online consumer's single rank-shared ledger."""
    if metadata_store is not None:
        return metadata_store
    if metadata_db_path is not None:
        import os

        # SQLite creates the database file but not its parent.  The typed
        # consumer_state_dir is intentionally allowed to name a fresh
        # node-local directory, so rank 0 must materialize it before opening
        # the single authority ledger.
        os.makedirs(os.path.dirname(os.path.abspath(metadata_db_path)), exist_ok=True)
        return SQLiteMetadataStore(metadata_db_path)
    return None


def _checkpoint_global_step(resume_from: str) -> int:
    """Read the shared checkpoint step without requiring a rank-state file."""
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


def _dp_consumer_layout(
    dp_rank: Optional[int],
    dp_size: Optional[int],
    tp_size: int,
    sp_ulysses_size: int,
    sp_ring_size: int,
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
    if dp_size > 1 and (tp_size != 1 or sp_ulysses_size != 1 or sp_ring_size != 1):
        raise NotImplementedError(
            "the online disaggregated consumer assigns one inbox to every "
            "trainer rank; nested target TP or draft SP is not supported "
            f"(tp={tp_size}, sp_ulysses={sp_ulysses_size}, "
            f"sp_ring={sp_ring_size})"
        )
    if not 0 <= dp_rank < dp_size:
        raise ValueError(f"dp_rank {dp_rank} out of range for dp_size {dp_size}")
    return dp_rank, dp_size


def _normalize_prompt_epochs(prompt_epochs: int) -> int:
    prompt_epochs = int(prompt_epochs or 1)
    if prompt_epochs < 1:
        raise ValueError(f"prompt_epochs must be >= 1, got {prompt_epochs}")
    return prompt_epochs


def _publish_refs_with_cleanup(
    *, channel, feature_store, refs: List[SampleRef]
) -> None:
    """Publish one captured batch and abort only its untouched suffix on error."""

    transaction = channel.begin_publish(refs)
    try:
        transaction.commit()
    except BaseException as publish_exc:
        cleanup_errors = []
        for ref in transaction.unpublished_refs:
            try:
                feature_store.abort(
                    ref.sample_id,
                    reason="producer-ref-publication-failed",
                )
            except Exception as cleanup_exc:
                cleanup_errors.append(
                    f"{ref.sample_id}: {type(cleanup_exc).__name__}: {cleanup_exc}"
                )
        if cleanup_errors:
            raise RuntimeError(
                "reference publication failed "
                f"({type(publish_exc).__name__}: {publish_exc}) and cleanup of "
                "unpublished server-captured refs also failed: "
                f"{cleanup_errors}"
            ) from publish_exc
        raise


def _epoch_online_prompts(
    prompts,
    epoch: int,
    prompt_epochs: int,
    *,
    seed: int = 0,
):
    """Build one deterministic, epoch-specific online prompt plan."""
    import random

    indexed_prompts = list(enumerate(prompts))
    random.Random(int(seed) + int(epoch)).shuffle(indexed_prompts)
    if prompt_epochs == 1:
        return [prompt for _idx, prompt in indexed_prompts]

    out = []
    for idx, prompt in indexed_prompts:
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


def _assemble_server_rollout_workers(
    *,
    algorithm: AlgorithmRegistration,
    modality: str,
    controller: DataFlowController,
    store: FeatureStore,
    run_id: str,
    target_hidden_size: int,
    target_vocab_size: Optional[int],
    draft_vocab_size: Optional[int],
    target_repr: Optional[str],
    aux_hidden_state_layer_ids,
    vocab_map_version: Optional[str],
    num_rollout_workers: int,
    feature_source,
):
    """Build workers over injected SGLang-server ref sources only."""

    if feature_source is None:
        raise ValueError(
            "online rollout requires an injected SGLang server feature source"
        )
    if isinstance(feature_source, (list, tuple)):
        if not feature_source:
            raise ValueError("feature_source sequence is empty")
        if num_rollout_workers not in (1, len(feature_source)):
            raise ValueError(
                f"num_rollout_workers={num_rollout_workers} conflicts with "
                f"{len(feature_source)} feature sources (one worker per source)"
            )
        sources = list(feature_source)
    else:
        sources = [feature_source] * num_rollout_workers

    from specforge.inference.capture import CaptureConfig
    from specforge.inference.rollout_worker import RolloutWorker

    contract = algorithm.spec.feature_contract("streaming", modality)
    capture_config = CaptureConfig.from_strategy(
        required_features=contract.required_tensors,
        aux_hidden_state_layer_ids=tuple(aux_hidden_state_layer_ids or ()),
        target_repr=target_repr,
        target_hidden_size=target_hidden_size,
        target_vocab_size=target_vocab_size,
        draft_vocab_size=draft_vocab_size,
        vocab_map_version=vocab_map_version,
    )
    return [
        RolloutWorker(
            controller,
            store,
            source,
            capture_config,
            run_id=run_id,
            worker_id=f"rollout-{index}",
            strategy=algorithm.name,
        )
        for index, source in enumerate(sources)
    ]


# ---------------------------------------------------------------------------
# Offline (colocated + disaggregated).
# ---------------------------------------------------------------------------


def _validate_offline_trainer_tp(tp_size: int) -> None:
    if tp_size != 1:
        raise ValueError(
            "offline feature consumers do not implement trainer tensor "
            "parallelism; keep tp_size=1 so every non-SP rank receives its "
            "own data shard"
        )


def build_offline_runtime(
    *,
    algorithm: AlgorithmRegistration,
    modality: str = "text",
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
    eval_interval: int = 0,
    eval_hidden_states_path: Optional[str] = None,
    eval_data_factory=None,
    tp_size: int = 1,
    sp_ulysses_size: int = 1,
    sp_ring_size: int = 1,
    use_usp_preprocess: bool = False,
    seed: int = 0,
    logger=None,
    log_interval: int = 50,
    resume_from: Optional[str] = None,
    max_checkpoints: int = 0,
    strategy_kwargs: Optional[Mapping[str, Any]] = None,
    dataloader_num_workers: int = 0,
    profiling_options=None,
):
    """Assemble the colocated offline dataflow (``LocalFeatureStore``).

    ``draft_model`` is the composite model for ``algorithm`` and must expose its
    trainable module as ``.draft_model``. Colocated offline refs are fixed and
    re-iterable, so this path does not allocate a training ledger or ref queue.
    """
    _validate_offline_trainer_tp(tp_size)
    provider = algorithm.providers.offline_for(modality)
    collate_fn, per_sample_transform = _offline_io(
        algorithm,
        modality,
        max_len,
        ttt_length=ttt_length,
        use_usp_preprocess=use_usp_preprocess,
    )
    controller = DataFlowController(
        run_id,
        metadata_store=NoOpMetadataStore(),
        enable_sample_queue=False,
    )
    source_refs = provider.build_reader(
        hidden_states_path, run_id=run_id, ttt_length=ttt_length, max_len=max_len
    ).read()

    def refs_for_epoch(epoch):
        return _shard_offline_refs(
            source_refs,
            use_usp_preprocess=use_usp_preprocess,
            seed=seed,
            epoch=epoch,
        )

    refs = refs_for_epoch(0)
    store = LocalFeatureStore(run_id)
    if eval_hidden_states_path:
        if eval_data_factory is not None:
            raise ValueError(
                "pass either eval_hidden_states_path or eval_data_factory, not both"
            )
        eval_data_factory = _make_offline_eval_data_factory(
            algorithm=algorithm,
            modality=modality,
            hidden_states_path=eval_hidden_states_path,
            run_id=run_id,
            batch_size=batch_size,
            max_len=max_len,
            ttt_length=ttt_length,
            use_usp_preprocess=use_usp_preprocess,
            dataloader_num_workers=dataloader_num_workers,
        )
    return _assemble_trainer(
        algorithm=algorithm,
        controller=controller,
        store=store,
        ref_source={"refs": refs, "refs_for_epoch": refs_for_epoch},
        model=draft_model,
        target_head=(
            target_head if algorithm.providers.step.uses_external_target_head else None
        ),
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
        eval_data_factory=eval_data_factory,
        logger=logger,
        log_interval=log_interval,
        collate_fn=collate_fn,
        strategy_kwargs=strategy_kwargs,
        per_sample_transform=per_sample_transform,
        durable_ack=False,
        resume_from=resume_from,
        checkpoint_extra={
            "offline_sampler_version": 1,
            "sampler_seed": seed,
            "source_dataset_size": len(source_refs),
        },
        max_checkpoints=max_checkpoints,
        tp_size=tp_size,
        sp_ulysses_size=sp_ulysses_size,
        sp_ring_size=sp_ring_size,
        dataloader_num_workers=dataloader_num_workers,
        profiling_options=profiling_options,
    )


def build_disagg_offline_runtime(
    *,
    algorithm: AlgorithmRegistration,
    modality: str = "text",
    feature_store: FeatureStore,
    refs: List[SampleRef],
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
    eval_interval: int = 0,
    eval_hidden_states_path: Optional[str] = None,
    eval_data_factory=None,
    tp_size: int = 1,
    sp_ulysses_size: int = 1,
    sp_ring_size: int = 1,
    use_usp_preprocess: bool = False,
    seed: int = 0,
    logger=None,
    log_interval: int = 50,
    resume_from: Optional[str] = None,
    max_checkpoints: int = 0,
    strategy_kwargs: Optional[Mapping[str, Any]] = None,
    dataloader_num_workers: int = 0,
    profiling_options=None,
):
    """Consumer side of a disaggregated OFFLINE run.

    Trains from a caller-supplied cross-process ``feature_store`` and the
    ``disagg://`` refs its producer published. Same trainer assembly as the
    colocated offline path, so results match within determinism tolerance.
    """
    _validate_offline_trainer_tp(tp_size)
    collate_fn, per_sample_transform = _offline_io(
        algorithm,
        modality,
        max_len,
        ttt_length=ttt_length,
        use_usp_preprocess=use_usp_preprocess,
    )
    source_refs = list(refs)

    def refs_for_epoch(epoch):
        return _shard_offline_refs(
            source_refs,
            use_usp_preprocess=use_usp_preprocess,
            seed=seed,
            epoch=epoch,
        )

    refs = refs_for_epoch(0)
    controller = DataFlowController(
        run_id,
        metadata_store=NoOpMetadataStore(),
        enable_sample_queue=False,
    )
    if eval_hidden_states_path:
        if eval_data_factory is not None:
            raise ValueError(
                "pass either eval_hidden_states_path or eval_data_factory, not both"
            )
        eval_data_factory = _make_offline_eval_data_factory(
            algorithm=algorithm,
            modality=modality,
            hidden_states_path=eval_hidden_states_path,
            run_id=run_id,
            batch_size=batch_size,
            max_len=max_len,
            ttt_length=ttt_length,
            use_usp_preprocess=use_usp_preprocess,
            dataloader_num_workers=dataloader_num_workers,
        )
    return _assemble_trainer(
        algorithm=algorithm,
        controller=controller,
        store=feature_store,
        ref_source={"refs": refs, "refs_for_epoch": refs_for_epoch},
        model=draft_model,
        target_head=(
            target_head if algorithm.providers.step.uses_external_target_head else None
        ),
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
        eval_data_factory=eval_data_factory,
        logger=logger,
        log_interval=log_interval,
        collate_fn=collate_fn,
        strategy_kwargs=strategy_kwargs,
        per_sample_transform=per_sample_transform,
        durable_ack=False,
        resume_from=resume_from,
        checkpoint_extra={
            "offline_sampler_version": 1,
            "sampler_seed": seed,
            "source_dataset_size": len(source_refs),
        },
        max_checkpoints=max_checkpoints,
        tp_size=tp_size,
        sp_ulysses_size=sp_ulysses_size,
        sp_ring_size=sp_ring_size,
        dataloader_num_workers=dataloader_num_workers,
        profiling_options=profiling_options,
    )


# ---------------------------------------------------------------------------
# Online disaggregated producer/consumer.  Capture is always delegated to an
# externally managed SGLang server source.
# ---------------------------------------------------------------------------


def build_disagg_online_producer(
    *,
    algorithm: AlgorithmRegistration,
    modality: str = "text",
    prompts,
    feature_store: FeatureStore,
    channel,
    run_id: str,
    target_hidden_size: int,
    target_vocab_size: Optional[int] = None,
    draft_vocab_size: Optional[int] = None,
    target_repr: Optional[str] = None,
    aux_hidden_state_layer_ids=None,
    vocab_map_version: Optional[str] = None,
    num_rollout_workers: int = 1,
    feature_source=None,
    lease: int = 8,
    in_flight_high_watermark: int = 256,
    in_flight_low_watermark: Optional[int] = None,
    resident_high_watermark_bytes: Optional[int] = None,
    resident_low_watermark_bytes: Optional[int] = None,
    feature_store_max_resident_bytes: Optional[int] = None,
    backpressure_poll_s: float = 0.2,
    peer_wait_timeout_s: Optional[float] = None,
    max_worker_failures: int = 3,
    max_prompt_attempts: Optional[int] = 5,
    sleep=None,
    prompt_epochs: int = 1,
    prompt_seed: int = 0,
):
    """Producer side of an ONLINE disaggregated run (rollout pool).

    Workers put() into a cross-node ``feature_store``; committed refs stream to
    the consumer via ``channel``. ``feature_source`` is an injected SGLang
    server-capture transport; a *sequence* of sources fans out to one worker
    per source (the multi-server topology; each worker drives its own server
    concurrently).  There is intentionally no in-process target-model path.
    The producer has no training ledger or local ref queue; the consumer owns
    deduplication and durable acknowledgements. Returns ``(workers,
    drive_producer)``:
    ``drive_producer(should_stop=...)`` runs until the prompt pool drains and
    pauses above the reference or resident-byte high watermark and resumes only
    below the corresponding low watermarks. A successful or cooperative stop
    closes the channel; a failure publishes a distinct failure sentinel.
    ``peer_wait_timeout_s`` bounds a backpressure wait when the consumer dies
    before it can publish its stop sentinel.

    ``prompt_epochs`` repeats the prompt stream on the producer side by minting
    epoch-tagged task/sample ids. Each pass uses the deterministic
    ``prompt_seed + epoch`` order, matching sampler-style epoch semantics while
    keeping a reconstructed plan stable across restarts.

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

    from specforge.runtime.control_plane.flow_control import (
        FlowControlLimits,
        ProducerFlowControl,
    )

    def producer_timing(message: str) -> None:
        print(
            f"[producer-timing] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}",
            flush=True,
        )

    def elapsed(start: float) -> str:
        return f"{time.perf_counter() - start:.3f}s"

    streaming = algorithm.providers.server_streaming_for(modality)
    contract = algorithm.spec.feature_contract("streaming", modality)
    if target_repr is None:
        target_repr = streaming.target_representation
    allowed_target_representations = contract.allowed_target_representations
    if allowed_target_representations and (
        target_repr not in allowed_target_representations
    ):
        raise ValueError(
            f"target representation {target_repr!r} is not supported by "
            f"algorithm {algorithm.name!r} for modality {modality!r}; "
            f"expected one of {sorted(allowed_target_representations)}"
        )
    if not allowed_target_representations and target_repr is not None:
        raise ValueError(
            f"algorithm {algorithm.name!r} for modality {modality!r} does not "
            f"consume a target representation, got {target_repr!r}"
        )
    sleep = sleep or time.sleep
    flow_control = ProducerFlowControl(
        FlowControlLimits(
            high_watermark_refs=in_flight_high_watermark,
            low_watermark_refs=in_flight_low_watermark,
            high_watermark_bytes=resident_high_watermark_bytes,
            low_watermark_bytes=resident_low_watermark_bytes,
            max_prompt_lease_per_worker=lease,
        )
    )
    if (
        feature_store_max_resident_bytes is not None
        and feature_store_max_resident_bytes < 1
    ):
        raise ValueError("feature_store_max_resident_bytes must be >= 1")
    if (
        feature_store_max_resident_bytes is not None
        and resident_high_watermark_bytes is not None
        and feature_store_max_resident_bytes < resident_high_watermark_bytes
    ):
        raise ValueError(
            "feature_store_max_resident_bytes must be >= "
            "resident_high_watermark_bytes"
        )
    worker_lease = flow_control.prompt_lease(lease)
    build_start = time.perf_counter()
    prompt_epochs = _normalize_prompt_epochs(prompt_epochs)
    if prompt_epochs > 1:
        prompts = list(prompts)
    base_prompt_count = len(prompts) if hasattr(prompts, "__len__") else "unknown"
    producer_timing(
        "build_disagg_online_producer enter "
        f"algorithm={algorithm.name} modality={modality} "
        f"base_prompts={base_prompt_count} "
        f"prompt_epochs={prompt_epochs} "
        f"lease={worker_lease} workers={num_rollout_workers} "
        f"watermarks={in_flight_high_watermark}/"
        f"{flow_control.limits.resolved_low_watermark_refs}"
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
    workers = _assemble_server_rollout_workers(
        algorithm=algorithm,
        modality=modality,
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
        num_rollout_workers=num_rollout_workers,
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
        from collections import deque

        drive_start = time.perf_counter()
        progress_interval = float(
            os.environ.get("DISAGG_PRODUCER_PROGRESS_INTERVAL", 30.0)
        )
        producer_timing(
            "drive_producer enter "
            f"workers={len(workers)} lease={worker_lease} max_rounds={max_rounds} "
            f"watermarks={in_flight_high_watermark}/"
            f"{flow_control.limits.resolved_low_watermark_refs} "
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
            resolved_low_watermark = flow_control.limits.resolved_low_watermark_refs
            if resolved_low_watermark < consumer_quantum:
                # The consumer dispatches only whole optimizer windows, so a
                # paused producer must be resumable while the consumer still
                # needs up to one full window: a low watermark below the
                # quantum could leave both sides waiting on each other.
                raise ValueError(
                    "producer in-flight low watermark "
                    f"{resolved_low_watermark} is smaller than the consumer's "
                    f"global optimizer-step quantum {consumer_quantum}; set "
                    "DISAGG_IN_FLIGHT_LOW_WATERMARK to at least that value"
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
        state = {
            "produced": 0,
            "first_ref_logged": False,
            "accounted_consumed": channel.consumed_remote(),
            "resident_bytes": 0,
        }
        published_sizes = deque()
        last_publish_log = {"t": time.perf_counter()}
        dead: dict = {}  # worker_id -> last failure reason

        def pool_drained() -> bool:
            st = controller.status()
            # leased counts too: a peer's in-flight lease may fail retryable
            # and come back — leaving then would strand it.
            return st["prompts_pending"] == 0 and st["prompts_leased"] == 0

        def reconcile_consumed_locked() -> int:
            consumed = channel.consumed_remote()
            delta = consumed - state["accounted_consumed"]
            if delta < 0 or delta > len(published_sizes):
                raise RuntimeError(
                    "producer byte accounting does not match the channel: "
                    f"consumed advanced by {delta} with "
                    f"{len(published_sizes)} published refs tracked"
                )
            for _ in range(delta):
                state["resident_bytes"] -= published_sizes.popleft()
            state["accounted_consumed"] = consumed
            return state["resident_bytes"]

        def resident_bytes() -> int:
            # Mooncake health is process-local and cannot observe deletes made by
            # a remote consumer. Track the published-but-unacknowledged prefix
            # from SampleRef.estimated_bytes and the channel's durable counter.
            with publish_lock:
                return reconcile_consumed_locked()

        def run_worker(w) -> None:
            failures = 0
            last_backpressure_log = 0.0
            for _ in range(max_rounds):
                if should_stop is not None and should_stop():
                    return
                _infl = channel.in_flight_remote()
                _resident = resident_bytes()
                # backpressure: in_flight = published - consumer-acked
                backpressure_started = time.monotonic()
                while flow_control.should_pause(
                    in_flight_refs=_infl,
                    resident_bytes=_resident,
                ):
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
                            f"resident_bytes={resident_bytes()} "
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
                    _resident = resident_bytes()
                try:
                    run_once_start = time.perf_counter()
                    refs = w.run_once(max_tasks=worker_lease)
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
                if refs:
                    with publish_lock:
                        current_bytes = reconcile_consumed_locked()
                        ref_sizes = [
                            max(0, int(ref.estimated_bytes or 0)) for ref in refs
                        ]
                        projected_bytes = current_bytes + sum(ref_sizes)
                        if (
                            feature_store_max_resident_bytes is not None
                            and projected_bytes > feature_store_max_resident_bytes
                        ):
                            cleanup_errors = []
                            for ref in refs:
                                try:
                                    feature_store.abort(
                                        ref.sample_id,
                                        reason="producer-resident-byte-hard-cap",
                                    )
                                except Exception as exc:
                                    cleanup_errors.append(
                                        f"{ref.sample_id}: {type(exc).__name__}: {exc}"
                                    )
                            message = (
                                "producer feature-store hard cap exceeded: "
                                f"projected={projected_bytes} > "
                                f"limit={feature_store_max_resident_bytes} bytes"
                            )
                            if cleanup_errors:
                                message += f"; cleanup errors={cleanup_errors}"
                            raise MemoryError(message)
                        _publish_refs_with_cleanup(
                            channel=channel,
                            feature_store=feature_store,
                            refs=refs,
                        )
                        published_sizes.extend(ref_sizes)
                        state["resident_bytes"] = projected_bytes
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
            epoch_prompts = _epoch_online_prompts(
                prompts,
                epoch,
                prompt_epochs,
                seed=prompt_seed,
            )
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

    drive_producer.flow_control = flow_control
    return workers, drive_producer


def build_disagg_online_consumer(
    *,
    algorithm: AlgorithmRegistration,
    modality: str = "text",
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
    eval_interval: int = 0,
    eval_data_factory=None,
    collate_fn=None,
    idle_timeout_s: Optional[float] = None,
    metadata_store: Optional[MetadataStore] = None,
    metadata_db_path: Optional[str] = None,
    logger=None,
    log_interval: int = 50,
    strategy_kwargs: Optional[Mapping[str, Any]] = None,
    max_checkpoints: int = 0,
    tp_size: int = 1,
    sp_ulysses_size: int = 1,
    sp_ring_size: int = 1,
    dp_rank: Optional[int] = None,
    dp_size: Optional[int] = None,
    inbox_dir: Optional[str] = None,
    resume_from: Optional[str] = None,
    dataloader_num_workers: int = 0,
    profiling_options=None,
):
    """Consumer (trainer) side of an ONLINE disaggregated run.

    Rank 0 always runs the :class:`RefDistributor`, including for ``dp_size=1``.
    It is the only reader of ``channel`` and the only writer to the
    ``metadata_store``/``metadata_db_path`` ledger, then dispatches refs into one
    inbox per rank. Every rank consumes the same inbox-based path and durable
    acknowledgements gather to rank 0 through :class:`DPAckController`.

    ``resume_from`` is consumer-only. Rank 0 reconciles the retained SQLite
    ledger, skips optimizer-durable refs, requeues the unacked tail, and requires
    the durable marker to match the checkpoint step. The producer and original
    data plane must still be available; this does not restart a producer.
    """
    import torch.distributed as dist

    from specforge.runtime.control_plane.dp_ack import DPAckController
    from specforge.runtime.data_plane.ref_distributor import (
        InboxChannel,
        RefDistributor,
    )
    from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefQueue

    algorithm.providers.server_streaming_for(modality)
    distributed = dist.is_available() and dist.is_initialized()
    world = dist.get_world_size() if distributed else 1
    actual_rank = dist.get_rank() if distributed else 0
    preflight_exc = None
    try:
        dp_rank, dp_size = _dp_consumer_layout(
            dp_rank,
            dp_size,
            tp_size,
            sp_ulysses_size,
            sp_ring_size,
        )
        if metadata_store is None and metadata_db_path is None:
            raise ValueError(
                "online consumer needs a metadata_store/metadata_db_path "
                "for its single rank-0 ledger"
            )
        if getattr(feature_store, "retain_on_release", False) is not True:
            raise ValueError(
                "online consumer feature_store must set retain_on_release=True; "
                "features are deleted only after an optimizer-boundary durable ack"
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

    distributor = None
    store = None
    setup_exc = None
    if dp_rank == 0:
        try:
            store = _resolve_metadata_store(metadata_store, metadata_db_path)
            if store is None or isinstance(store, NoOpMetadataStore):
                raise ValueError("online consumer requires a retaining metadata ledger")
            controller = DPAckController(
                run_id,
                is_authority=True,
                metadata_store=store,
                feature_store=feature_store,
            )
            skip_ids = None
            requeued_ids = None
            committed = store.committed_count()
            if resume_from is None:
                if committed > 0:
                    raise ValueError(
                        f"metadata store already holds {committed} committed "
                        "samples; a fresh online attempt cannot reuse a ledger. "
                        "Set training.resume_from to reconcile this retained "
                        "consumer attempt, or use a new metadata_db_path"
                    )
            else:
                if isinstance(store, InMemoryMetadataStore):
                    raise ValueError(
                        "online consumer resume requires a durable metadata store; "
                        "use metadata_db_path/SQLite rather than an in-memory ledger"
                    )
                reconciled = controller.reconcile_on_restart(feature_store)
                # The fresh authority adopted every optimizer-durable committed
                # ref before aborting it. Resolve any lease-deferred remote
                # deletes now, while setup failures can still be broadcast to
                # every rank, rather than carrying a hidden leak into training.
                drain_feature_store_removals(feature_store)
                marker_step = reconciled["global_step"]
                checkpoint_step = _checkpoint_global_step(resume_from)
                if marker_step is not None and not reconciled["optimizer_durable"]:
                    raise RuntimeError(
                        f"durable marker at global_step={marker_step} is not marked "
                        "optimizer_durable; refusing to skip or replay ambiguously"
                    )
                if marker_step != checkpoint_step and not (
                    marker_step is None and checkpoint_step == 0
                ):
                    direction = (
                        "ahead of"
                        if marker_step is not None and marker_step > checkpoint_step
                        else "behind"
                    )
                    raise RuntimeError(
                        f"durable marker global_step={marker_step} is {direction} "
                        f"checkpoint {resume_from!r} global_step={checkpoint_step}; "
                        "the retained ack set and restored weights do not describe "
                        "the same optimizer boundary"
                    )
                skip_ids = set(reconciled["released"])
                requeued_ids = set(reconciled["requeued"])
            distributor = RefDistributor(
                channel,
                controller,
                inbox_dir,
                dp_size,
                feature_store=feature_store,
                refs_per_rank_step=batch_size * accumulation_steps,
                refs_per_rank_batch=batch_size,
                skip_ids=skip_ids,
                requeued_ids=requeued_ids,
                idle_timeout_s=idle_timeout_s,
            )
            channel.publish_consumer_quantum(
                dp_size * batch_size * accumulation_steps,
                allow_existing=resume_from is not None,
            )
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
            run_id,
            is_authority=False,
            metadata_store=InMemoryMetadataStore(),
            feature_store=feature_store,
        )

    # The successful rank-0 setup broadcast guarantees inbox recreation and the
    # optimizer-window sidecar are visible before any rank opens its reader.
    inbox = InboxChannel(RefDistributor.inbox_path(inbox_dir, dp_rank))
    queue = StreamingRefQueue(inbox, idle_timeout_s=idle_timeout_s)

    drain_state = {"attempted": False}
    fit_failure = {"error": None}

    def drain_consumer_store() -> None:
        drain_state["attempted"] = True
        drain_feature_store_removals(feature_store)

    def mark_consumer_done(_step: int) -> None:
        # Do not publish success while another DP rank still has a deferred
        # Mooncake remove. Every rank drains its own client, then collectively
        # reports cleanup success/failure before rank 0 releases the producer.
        cleanup_error = None
        try:
            drain_consumer_store()
        except BaseException as exc:
            cleanup_error = f"{type(exc).__name__}: {exc}"
        if distributed and world > 1:
            gathered_errors = [None] * world
            dist.all_gather_object(gathered_errors, cleanup_error)
            failures = [
                f"rank {rank}: {error}"
                for rank, error in enumerate(gathered_errors)
                if error is not None
            ]
            cleanup_error = "; ".join(failures) or None
        if cleanup_error is not None:
            raise RuntimeError(
                "online consumer could not drain rank-local feature removals: "
                f"{cleanup_error}"
            )
        if dp_rank == 0:
            channel.mark_consumer_done()

    def mark_consumer_failed(exc: BaseException) -> None:
        # Every rank may report failure: a non-authority rank can fail while
        # rank 0 is blocked in a distributed collective. Publishing the same
        # terminal sidecar is idempotent and lets the producer stop promptly.
        fit_failure["error"] = exc
        try:
            channel.mark_consumer_failed(f"{type(exc).__name__}: {exc}")
        except Exception as signal_exc:
            print(f"failed to publish consumer failure: {signal_exc}", flush=True)

    def stop_distributor_and_drain() -> None:
        if distributor is not None:
            distributor.stop()
        # The success hook already drained before publishing consumer_done. On
        # an exception, finalization still makes one bounded local attempt and
        # reports a cleanup failure loudly without replacing the primary fit
        # exception that is already propagating through Trainer.fit().
        if not drain_state["attempted"]:
            try:
                drain_consumer_store()
            except BaseException as cleanup_exc:
                primary = fit_failure["error"]
                if primary is None:
                    raise
                combined = RuntimeError(
                    f"primary consumer failure ({type(primary).__name__}: "
                    f"{primary}); pending-remove drain also failed "
                    f"({type(cleanup_exc).__name__}: {cleanup_exc})"
                )
                try:
                    channel.mark_consumer_failed(
                        f"{type(combined).__name__}: {combined}"
                    )
                except Exception as signal_exc:
                    print(
                        "failed to publish consumer cleanup failure: " f"{signal_exc}",
                        flush=True,
                    )
                logging.getLogger(__name__).error("%s", combined)

    try:
        trainer = _assemble_trainer(
            algorithm=algorithm,
            controller=controller,
            store=feature_store,
            ref_source={
                "queue": queue,
                "prepositioned": resume_from is not None,
                "defer_ack_until_durable": True,
            },
            model=draft_model,
            target_head=(
                target_head
                if algorithm.providers.step.uses_external_target_head
                else None
            ),
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
            eval_data_factory=eval_data_factory,
            logger=logger,
            log_interval=log_interval,
            collate_fn=_streaming_collate(algorithm, modality, collate_fn),
            strategy_kwargs=strategy_kwargs,
            per_sample_transform=None,
            max_checkpoints=max_checkpoints,
            tp_size=tp_size,
            sp_ulysses_size=sp_ulysses_size,
            sp_ring_size=sp_ring_size,
            resume_from=resume_from,
            dataloader_num_workers=dataloader_num_workers,
            profiling_options=profiling_options,
            on_fit_success=mark_consumer_done,
            on_fit_failure=mark_consumer_failed,
            on_fit_finally=stop_distributor_and_drain,
        )
        #: Rank 0's lifecycle-owned RefDistributor handle (None elsewhere), exposed
        #: for runtime observability. ``Trainer.fit()`` always stops it.
        trainer.ref_distributor = distributor
        if distributor is not None:
            distributor.start()
        return trainer
    except BaseException as exc:
        # The canonical builder also owns failures before ``Trainer.fit`` can
        # take over, so a direct Python caller cannot strand its producer.
        mark_consumer_failed(exc)
        stop_distributor_and_drain()
        raise


def _windowed_prompt_tasks(run_id: str, prompts) -> List[Any]:
    """Normalize a restart-stable canonical prompt stream."""
    from specforge.runtime.contracts import PromptTask, assert_no_tensors

    tasks = []
    for prompt in prompts:
        if isinstance(prompt, PromptTask):
            task = prompt
            if task.run_id != run_id:
                raise ValueError(
                    f"prompt {task.task_id!r} run_id={task.run_id!r} does not "
                    f"match capture run {run_id!r}"
                )
        else:
            assert_no_tensors(prompt)
            task_id = prompt.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(
                    "windowed capture requires an explicit stable task_id on "
                    "every prompt"
                )
            task = PromptTask(
                task_id=task_id,
                run_id=run_id,
                source_id=str(prompt.get("source_id", "prompt_source")),
                payload=dict(prompt.get("payload", prompt)),
                max_length=int(prompt.get("max_length", 2048)),
                chat_template=prompt.get("chat_template"),
                loss_mask_policy=dict(prompt.get("loss_mask_policy", {})),
                target_model_version=str(prompt.get("target_model_version", "unknown")),
                draft_weight_version=prompt.get("draft_weight_version"),
                metadata=dict(prompt.get("metadata", {})),
            )
        assert_no_tensors(task)
        tasks.append(task)
    if not tasks:
        raise ValueError("windowed capture prompts must not be empty")
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("windowed capture task_ids must be unique")
    return tasks


def _resolve_window_algorithm(
    strategy: str | AlgorithmRegistration,
) -> AlgorithmRegistration:
    if isinstance(strategy, AlgorithmRegistration):
        return strategy
    from specforge.algorithms.builtin import builtin_algorithm_registry

    try:
        return builtin_algorithm_registry().resolve(strategy)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc


def build_disagg_windowed_capture_contract(
    *,
    strategy: str | AlgorithmRegistration,
    modality: str = "text",
    target_hidden_size: int,
    target_model_version: str,
    tokenizer_version: str,
    target_vocab_size: Optional[int] = None,
    draft_vocab_size: Optional[int] = None,
    target_repr: str = "logits",
    aux_hidden_state_layer_ids=None,
    vocab_map_version: Optional[str] = None,
):
    """Build the typed capture request and its cross-process identity digest."""
    from specforge.inference.capture import CaptureConfig
    from specforge.runtime.data_plane.windowed_capture import capture_contract_digest

    algorithm = _resolve_window_algorithm(strategy)
    feature_contract = algorithm.spec.feature_contract("streaming", modality)
    capture = CaptureConfig.from_strategy(
        required_features=feature_contract.required_tensors,
        aux_hidden_state_layer_ids=tuple(aux_hidden_state_layer_ids or ()),
        target_repr=target_repr,
        target_hidden_size=target_hidden_size,
        target_vocab_size=target_vocab_size,
        draft_vocab_size=draft_vocab_size,
        vocab_map_version=vocab_map_version,
    )
    digest = capture_contract_digest(
        {
            "strategy": algorithm.name,
            "capture": capture,
            "target_model_version": target_model_version,
            "tokenizer_version": tokenizer_version,
        }
    )
    return capture, digest


@dataclass
class DisaggWindowedProducerRuntime:
    registry: Any
    service: Any
    contract_digest: str

    def drive(self, max_rounds: int = 10_000_000, *, should_stop=None) -> int:
        return self.service.drive(should_stop=should_stop, max_rounds=max_rounds)

    def accounting_snapshot(self) -> Dict[str, Any]:
        snapshot = self.service.snapshot()
        snapshot["contract_digest"] = self.contract_digest
        return snapshot

    def close(self) -> None:
        self.registry.close()


def build_disagg_online_windowed_producer(
    *,
    prompts,
    feature_store: FeatureStore,
    feature_source: Any,
    run_id: str,
    consumer_ids,
    registry_db_path: str,
    max_live_refs: int,
    target_hidden_size: int,
    target_model_version: str,
    tokenizer_version: str,
    strategy: str = "eagle3",
    target_vocab_size: Optional[int] = None,
    draft_vocab_size: Optional[int] = None,
    target_repr: str = "logits",
    aux_hidden_state_layer_ids=None,
    vocab_map_version: Optional[str] = None,
    max_live_bytes: Optional[int] = None,
    capture_reservation_bytes: Optional[int] = None,
    capture_batch_size: int = 8,
    capture_batch_wait_s: float = 0.002,
    max_capture_retries: int = 2,
    retry_backoff_s: float = 0.05,
    consumer_registration_timeout_s: float = 600.0,
    consumer_heartbeat_timeout_s: float = 120.0,
    registry_poll_s: float = 0.01,
    recover: bool = False,
) -> DisaggWindowedProducerRuntime:
    """Build one demand-driven producer for fixed independent consumers."""
    from specforge.runtime.data_plane.windowed_capture import (
        SQLiteWindowedCaptureRegistry,
    )
    from specforge.runtime.data_plane.windowed_capture_runtime import (
        WindowedCaptureService,
    )

    tasks = _windowed_prompt_tasks(run_id, prompts)
    incompatible_prompts = [
        task.task_id
        for task in tasks
        if task.target_model_version not in ("unknown", target_model_version)
    ]
    if incompatible_prompts:
        raise ValueError(
            "windowed prompts target a different model version: "
            f"{incompatible_prompts[:8]}"
        )
    capture, digest = build_disagg_windowed_capture_contract(
        strategy=strategy,
        target_hidden_size=target_hidden_size,
        target_model_version=target_model_version,
        tokenizer_version=tokenizer_version,
        target_vocab_size=target_vocab_size,
        draft_vocab_size=draft_vocab_size,
        target_repr=target_repr,
        aux_hidden_state_layer_ids=aux_hidden_state_layer_ids,
        vocab_map_version=vocab_map_version,
    )
    registry = SQLiteWindowedCaptureRegistry(
        registry_db_path,
        max_live_refs=max_live_refs,
        max_live_bytes=max_live_bytes,
        capture_reservation_bytes=capture_reservation_bytes,
        poll_s=registry_poll_s,
    )
    try:
        registry.initialize_run(
            run_id=run_id,
            contract_digest=digest,
            source_sample_ids=[task.task_id for task in tasks],
            expected_consumers=tuple(consumer_ids),
            recover_inflight=recover,
            recovery_store=feature_store if recover else None,
        )
        service = WindowedCaptureService(
            registry,
            prompts=tasks,
            feature_source=feature_source,
            capture=capture,
            owner_store=feature_store,
            capture_batch_size=capture_batch_size,
            batch_wait_s=capture_batch_wait_s,
            max_capture_retries=max_capture_retries,
            retry_backoff_s=retry_backoff_s,
            consumer_registration_timeout_s=consumer_registration_timeout_s,
            consumer_heartbeat_timeout_s=consumer_heartbeat_timeout_s,
            poll_s=registry_poll_s,
        )
    except BaseException:
        registry.close()
        raise
    return DisaggWindowedProducerRuntime(registry, service, digest)


@dataclass
class DisaggWindowedConsumerRuntime:
    trainer: Any
    loader: Any
    queue: Any
    control: Any
    controller: DataFlowController
    registry: Any
    max_steps: Optional[int]

    def run(self) -> int:
        """Train to EOF or to this consumer's explicit independent step cap."""
        try:
            self.control.mark_ready()
            step = self.trainer.fit()
            self.control.ensure_healthy()
            if self.queue.drained():
                self.control.complete()
            elif self.max_steps is not None and step >= self.max_steps:
                self.queue.close()
                self.control.complete(allow_partial=True)
            else:
                raise RuntimeError(
                    "windowed consumer stopped before EOF without reaching its "
                    "configured max_steps"
                )
            return step
        except BaseException as exc:
            try:
                self.queue.close()
            except BaseException as cleanup_error:
                exc.add_note(f"failed to close windowed queue: {cleanup_error!r}")
            try:
                state = self.registry.snapshot()["consumers"][self.control.consumer_id][
                    "state"
                ]
                if state != "completed":
                    self.control.fail(exc)
            except BaseException as cleanup_error:
                exc.add_note(f"failed to report consumer failure: {cleanup_error!r}")
            raise

    def accounting_snapshot(self) -> Dict[str, Any]:
        marker = self.controller.store.durable_marker()
        return {
            "consumer_id": self.control.consumer_id,
            "window": self.registry.snapshot()["consumers"][self.control.consumer_id],
            "committed": self.controller.store.committed_count(),
            "acked": len(marker["acked"]),
            "global_step": marker["global_step"],
            "queue": self.queue.metrics(),
            "input_pipeline": self.loader.metrics(),
        }

    def close(self) -> None:
        self.control.close()
        close_store = getattr(self.controller.store, "close", None)
        if callable(close_store):
            close_store()
        self.registry.close()


def _durable_window_cursor(metadata_store: MetadataStore) -> int:
    """Return the contiguous optimizer-durable prefix in canonical fetch order."""
    marker = metadata_store.durable_marker()
    acked = marker["acked"]
    cursor = 0
    for sample_id in metadata_store.all_committed_ids():
        if sample_id not in acked:
            break
        cursor += 1
    return cursor


def build_disagg_online_windowed_consumer(
    *,
    consumer_id: str,
    registry_db_path: str,
    max_live_refs: int,
    contract_digest: str,
    total_samples: int,
    feature_store: FeatureStore,
    eagle3_model,
    optimizer_factory,
    run_id: str,
    output_dir: str,
    metadata_db_path: str,
    lookbehind: int = 0,
    lookahead: int = 0,
    prefetch_depth: int = 0,
    max_outstanding: int = 1,
    strategy: str | AlgorithmRegistration = "eagle3",
    modality: str = "text",
    batch_size: int = 1,
    accumulation_steps: int = 1,
    num_epochs: int = 1,
    max_steps: Optional[int] = None,
    total_steps: Optional[int] = None,
    save_interval: int = 0,
    eval_interval: int = 0,
    collate_fn=None,
    idle_timeout_s: Optional[float] = 1800.0,
    logger=None,
    log_interval: int = 50,
    strategy_kwargs: Optional[dict] = None,
    resume: bool = False,
    resume_from: Optional[str] = None,
    max_checkpoints: int = 0,
    max_live_bytes: Optional[int] = None,
    capture_reservation_bytes: Optional[int] = None,
    heartbeat_interval_s: float = 5.0,
    initialization_timeout_s: float = 600.0,
    registry_poll_s: float = 0.01,
    loader_prefetch_batches: int = 0,
    consumer_control=None,
) -> DisaggWindowedConsumerRuntime:
    """Build one single-GPU trainer over an independent windowed cursor."""
    from specforge.runtime.control_plane.metadata_store import SQLiteMetadataStore
    from specforge.runtime.data_plane.windowed_capture import (
        SQLiteWindowedCaptureRegistry,
        WindowedCaptureQueue,
    )
    from specforge.runtime.data_plane.windowed_capture_runtime import (
        start_windowed_consumer_control,
    )

    if num_epochs != 1:
        raise ValueError("windowed canonical streams support exactly one epoch")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be >= 1 or None")
    if resume_from is not None and not resume:
        raise ValueError("resume_from requires resume=True")
    if max_outstanding < batch_size * accumulation_steps:
        raise ValueError(
            "max_outstanding must cover batch_size * accumulation_steps so "
            "optimizer-boundary ACKs cannot deadlock the queue"
        )
    if hasattr(feature_store, "lifetime_owner") and feature_store.lifetime_owner:
        raise ValueError("windowed consumers must not own shared payload lifetime")

    owns_registry = consumer_control is None
    registry = (
        consumer_control.registry
        if not owns_registry
        else SQLiteWindowedCaptureRegistry(
            registry_db_path,
            max_live_refs=max_live_refs,
            max_live_bytes=max_live_bytes,
            capture_reservation_bytes=capture_reservation_bytes,
            poll_s=registry_poll_s,
        )
    )
    metadata_store = None
    try:
        initialized = registry.wait_initialized(initialization_timeout_s)
        observed = (
            initialized["run_id"],
            initialized["contract_digest"],
            initialized["total_samples"],
            initialized["max_live_refs"],
            initialized["max_live_bytes"],
            initialized["capture_reservation_bytes"],
        )
        expected = (
            run_id,
            contract_digest,
            total_samples,
            max_live_refs,
            max_live_bytes,
            capture_reservation_bytes or 0,
        )
        if observed != expected:
            raise RuntimeError(
                f"windowed registry identity mismatch: expected={expected!r}, "
                f"observed={observed!r}"
            )

        os.makedirs(os.path.dirname(os.path.abspath(metadata_db_path)), exist_ok=True)
        metadata_store = SQLiteMetadataStore(metadata_db_path)
        if not resume and metadata_store.committed_count():
            raise ValueError(
                "windowed consumer metadata store is not fresh; pass resume=True "
                "with a matching checkpoint or use a new run directory"
            )
        durable_cursor = _durable_window_cursor(metadata_store) if resume else 0
        if durable_cursor and resume_from is None:
            raise ValueError(
                "windowed consumer resume found an acknowledged prefix but no "
                "resume_from checkpoint; skipping trained samples without restoring "
                "their weight updates would lose data"
            )
        if resume_from is not None:
            marker_step = metadata_store.durable_marker()["global_step"]
            checkpoint_step = _checkpoint_global_step(resume_from)
            if marker_step is not None and marker_step > checkpoint_step:
                raise RuntimeError(
                    f"durable marker global_step={marker_step} is ahead of "
                    f"checkpoint global_step={checkpoint_step}"
                )
        if consumer_control is None:
            existing = registry.snapshot()["consumers"].get(consumer_id)
            if existing is not None and not resume:
                raise RuntimeError(
                    f"consumer {consumer_id!r} already exists; pass resume=True"
                )
            consumer_control = start_windowed_consumer_control(
                registry,
                consumer_id,
                lookbehind=lookbehind,
                lookahead=lookahead,
                prefetch_depth=prefetch_depth,
                max_outstanding=max_outstanding,
                heartbeat_interval_s=heartbeat_interval_s,
                durable_cursor=durable_cursor,
            )
        else:
            if consumer_control.consumer_id != consumer_id:
                raise ValueError("consumer_control identity mismatch")
            existing = registry.snapshot()["consumers"][consumer_id]
            expected_window = (
                lookbehind,
                lookahead,
                prefetch_depth,
                max_outstanding,
            )
            observed_window = tuple(
                int(existing[name])
                for name in (
                    "lookbehind",
                    "lookahead",
                    "prefetch_depth",
                    "max_outstanding",
                )
            )
            if observed_window != expected_window:
                raise ValueError(
                    f"consumer_control window mismatch: expected={expected_window}, "
                    f"observed={observed_window}"
                )
            if resume:
                registry.resume_consumer(consumer_id, durable_cursor=durable_cursor)
    except BaseException as exc:
        if owns_registry and consumer_control is not None:
            try:
                consumer_control.fail(exc)
            except BaseException as cleanup_error:
                exc.add_note(
                    f"failed to report windowed consumer setup failure: "
                    f"{cleanup_error!r}"
                )
        if metadata_store is not None:
            metadata_store.close()
        if owns_registry:
            registry.close()
        raise

    queue = None
    try:
        controller = DataFlowController(run_id, metadata_store=metadata_store)
        queue = WindowedCaptureQueue(
            registry,
            consumer_id,
            idle_timeout_s=idle_timeout_s,
            record_refs=lambda refs: controller.record_external_refs(list(refs)),
        )
        controller.sample_queue = queue
        algorithm = _resolve_window_algorithm(strategy)
        trainer = _assemble_trainer(
            algorithm=algorithm,
            modality=modality,
            controller=controller,
            store=feature_store,
            ref_source={"queue": queue, "defer_ack_until_durable": True},
            model=eagle3_model,
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
            eval_interval=eval_interval,
            tp_size=1,
            sp_ulysses_size=1,
            sp_ring_size=1,
            logger=logger,
            log_interval=log_interval,
            collate_fn=_streaming_collate(algorithm, modality, collate_fn),
            strategy_kwargs=strategy_kwargs,
            per_sample_transform=None,
            durable_ack=True,
            resume_from=resume_from,
            max_checkpoints=max_checkpoints,
            dataloader_num_workers=loader_prefetch_batches,
        )
    except BaseException as exc:
        if queue is not None:
            try:
                queue.close()
            except BaseException as cleanup_error:
                exc.add_note(f"failed to close windowed queue: {cleanup_error!r}")
        try:
            consumer_control.fail(exc)
        except BaseException as cleanup_error:
            exc.add_note(
                f"failed to report windowed consumer setup failure: "
                f"{cleanup_error!r}"
            )
        metadata_store.close()
        if owns_registry:
            registry.close()
        raise
    return DisaggWindowedConsumerRuntime(
        trainer,
        trainer._loader,
        queue,
        consumer_control,
        controller,
        registry,
        max_steps,
    )


__all__ = [
    "build_offline_runtime",
    "build_disagg_offline_runtime",
    "build_disagg_online_producer",
    "build_disagg_online_consumer",
]
