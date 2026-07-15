# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Disaggregated run assembly used by ``specforge train``.

The role is configuration, not a different Python entry point. Producer and
consumer processes load the same run file with ``training.role`` set to their
respective role. Cross-process transport values intentionally remain environment
variables because they are deployment secrets/addresses rather than model
hyperparameters.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Optional, Sequence

from specforge.config import Config

_PRODUCER_CLAIM_SUFFIX = ".producer_claim"
_ONLINE_CONTROL_SUFFIXES = (
    ".closed",
    ".consumed_count",
    ".failed",
    ".consumer_done",
    ".consumer_failed",
    ".consumer_quantum",
)
_OFFLINE_CONTROL_SUFFIXES = (".done", ".consumed", ".failed", ".consumer_failed")


def _write_control(path: str, value: str = "") -> None:
    """Atomically publish one small filesystem control record."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _read_control(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8") as stream:
            return stream.read().strip() or "unknown remote failure"
    except FileNotFoundError:
        return None


def _publish_control_failure(path: str, exc: BaseException) -> None:
    """Best-effort peer notification that never hides the root exception."""
    try:
        _write_control(path, f"{type(exc).__name__}: {exc}")
    except Exception as signal_exc:
        print(
            f"failed to publish role failure to {path}: {signal_exc}",
            flush=True,
        )


def _claim_fresh_control_path(path: str, suffixes: Sequence[str]) -> None:
    """Reject artifact reuse and atomically claim a new producer attempt."""
    claim = path + _PRODUCER_CLAIM_SUFFIX
    artifacts = [path, claim, *(path + suffix for suffix in suffixes)]
    existing = [item for item in artifacts if os.path.exists(item)]
    if existing:
        raise ValueError(
            f"control path {path!r} has artifacts from an existing attempt: "
            f"{existing}; choose a new attempt-specific path"
        )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise ValueError(
            f"control path {path!r} is already claimed by another producer; "
            "choose a new attempt-specific path"
        ) from exc
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(f"pid={os.getpid()}\n")
        stream.flush()
        os.fsync(stream.fileno())


def _publish_role_assembly_failure(cfg: Config, exc: BaseException) -> None:
    """Notify the peer when a role fails before its run closure is returned.

    Producer notification is allowed only after this process successfully
    claimed the attempt path.  That prevents a typo/retry using an occupied
    path from poisoning another live producer's channel.
    """
    env_name = "DISAGG_MANIFEST" if cfg.mode == "offline" else "DISAGG_REF_CHANNEL"
    path = os.environ.get(env_name)
    if not path:
        return
    if cfg.training.role == "producer":
        claim = _read_control(path + _PRODUCER_CLAIM_SUFFIX)
        if claim != f"pid={os.getpid()}":
            return
        failure_path = path + ".failed"
    else:
        failure_path = path + ".consumer_failed"
    _publish_control_failure(failure_path, exc)


def _primary_rank() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"disaggregated training requires environment variable {name}")
    return value


def _mooncake_store(cfg: Config, *, retain_on_release: bool = False):
    from specforge.runtime.data_plane.disaggregated import AuthPolicy
    from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore

    token = os.environ.get("DISAGG_AUTH_TOKEN") or None
    setup_kwargs = {
        "local_hostname": os.environ.get("MOONCAKE_LOCAL_HOSTNAME", "127.0.0.1"),
        "metadata_server": _env("MOONCAKE_METADATA_SERVER"),
        "master_server_addr": _env("MOONCAKE_MASTER_SERVER_ADDR"),
        "protocol": os.environ.get("MOONCAKE_PROTOCOL", "tcp"),
        "rdma_devices": os.environ.get("MOONCAKE_RDMA_DEVICES", ""),
    }
    for env_name, key in (
        ("MOONCAKE_GLOBAL_SEGMENT_SIZE", "global_segment_size"),
        ("DISAGG_CLIENT_SEGMENT_SIZE", "global_segment_size"),
        ("MOONCAKE_LOCAL_BUFFER_SIZE", "local_buffer_size"),
        ("DISAGG_CLIENT_BUFFER_SIZE", "local_buffer_size"),
    ):
        if os.environ.get(env_name):
            setup_kwargs[key] = int(os.environ[env_name])
    return MooncakeFeatureStore(
        store_id=os.environ.get("DISAGG_STORE_ID", cfg.run_id),
        setup_kwargs=setup_kwargs,
        auth=AuthPolicy(token),
        credential=token,
        retain_on_release=retain_on_release,
    )


def _offline_store(cfg: Config, *, retain_on_release: bool = False):
    backend = os.environ.get("DISAGG_BACKEND", "shared_dir")
    if backend == "mooncake":
        return _mooncake_store(cfg, retain_on_release=retain_on_release)
    if backend != "shared_dir":
        raise ValueError(
            f"unknown DISAGG_BACKEND={backend!r}; expected shared_dir or mooncake"
        )
    from specforge.runtime.data_plane.disaggregated import (
        AuthPolicy,
        SharedDirFeatureStore,
    )

    token = os.environ.get("DISAGG_AUTH_TOKEN") or None
    return SharedDirFeatureStore(
        _env("DISAGG_STORE_ROOT"),
        store_id=os.environ.get("DISAGG_STORE_ID", cfg.run_id),
        auth=AuthPolicy(token),
        credential=token,
        retain_on_release=retain_on_release,
    )


def _server_urls(cfg: Config) -> list[str]:
    if cfg.training.server_urls:
        return list(cfg.training.server_urls)
    raw = os.environ.get("DISAGG_SERVER_URLS") or os.environ.get("DISAGG_SERVER_URL")
    if not raw:
        raise ValueError(
            "online producer requires training.server_urls, DISAGG_SERVER_URLS, "
            "or DISAGG_SERVER_URL"
        )
    return [item.strip() for item in raw.split(",") if item.strip()]


def _wait_for(
    path: str,
    *,
    timeout_s: float = 1800.0,
    failure_path: Optional[str] = None,
) -> None:
    deadline = time.monotonic() + timeout_s
    while not os.path.exists(path):
        failure = _read_control(failure_path) if failure_path else None
        if failure is not None:
            raise RuntimeError(
                f"remote role failed while waiting for {path}: {failure}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.25)


def _hold_mooncake_producer(manifest: str) -> None:
    if os.environ.get("DISAGG_BACKEND", "shared_dir") != "mooncake":
        return
    consumed = manifest + ".consumed"
    _wait_for(
        consumed,
        timeout_s=float(os.environ.get("DISAGG_PRODUCER_HOLD_S", "3600")),
        failure_path=manifest + ".consumer_failed",
    )


def _cleanup_offline_mooncake_refs(store, refs: Sequence, *, reason: str) -> None:
    """Remove every object owned by one terminal offline producer attempt."""
    from specforge.runtime.data_plane.feature_store import drain_feature_store_removals

    cleanup_errors = []
    for ref in refs:
        try:
            store.abort(ref.sample_id, reason=reason)
        except Exception as exc:
            cleanup_errors.append(f"{ref.sample_id}: {type(exc).__name__}: {exc}")
    try:
        drain_feature_store_removals(store)
    except Exception as exc:
        cleanup_errors.append(f"pending-remove drain: {type(exc).__name__}: {exc}")
    if cleanup_errors:
        raise RuntimeError(
            "offline Mooncake cleanup did not remove every ingested sample: "
            f"{cleanup_errors}"
        )


def _build_offline(
    cfg: Config,
    *,
    build_model_bundle: Callable,
    optimizer_factory: Callable,
    logger: Callable,
):
    from specforge.training.assembly import (
        TrainingRun,
        _dataloader_num_workers,
        _profiling_options,
    )

    manifest = _env("DISAGG_MANIFEST")
    done = manifest + ".done"

    if cfg.training.role == "producer":
        _claim_fresh_control_path(manifest, _OFFLINE_CONTROL_SUFFIXES)

        def produce() -> int:
            from specforge.runtime.data_plane.disagg_ingest import (
                ingest_offline_features,
                write_ref_manifest,
            )

            store = None
            tracked_refs = []
            produced = 0
            primary_exc = None
            try:
                store = _offline_store(cfg)
                refs = ingest_offline_features(
                    store,
                    cfg.data.hidden_states_path,
                    strategy=cfg.training.strategy,
                    run_id=cfg.run_id,
                    ttt_length=cfg.training.ttt_length,
                    max_len=cfg.data.max_length,
                    on_ref=tracked_refs.append,
                )
                produced = len(refs)
                write_ref_manifest(refs, manifest)
                _write_control(done)
                _hold_mooncake_producer(manifest)
            except BaseException as exc:
                primary_exc = exc

            cleanup_exc = None
            if (
                store is not None
                and os.environ.get("DISAGG_BACKEND", "shared_dir") == "mooncake"
            ):
                try:
                    _cleanup_offline_mooncake_refs(
                        store,
                        tracked_refs,
                        reason=(
                            "offline-attempt-failed"
                            if primary_exc is not None
                            else "offline-attempt-finished"
                        ),
                    )
                except Exception as exc:
                    cleanup_exc = exc

            if primary_exc is not None and cleanup_exc is not None:
                combined = RuntimeError(
                    f"offline producer failed ({type(primary_exc).__name__}: "
                    f"{primary_exc}) and Mooncake cleanup also failed "
                    f"({type(cleanup_exc).__name__}: {cleanup_exc})"
                )
                _publish_control_failure(manifest + ".failed", combined)
                raise combined from primary_exc
            if primary_exc is not None:
                _publish_control_failure(manifest + ".failed", primary_exc)
                raise primary_exc
            if cleanup_exc is not None:
                _publish_control_failure(manifest + ".failed", cleanup_exc)
                raise cleanup_exc
            return produced

        return TrainingRun(execute=produce)

    _wait_for(done, failure_path=manifest + ".failed")
    from specforge.launch import build_disagg_offline_runtime
    from specforge.runtime.data_plane.disagg_ingest import read_ref_manifest

    bundle = build_model_bundle(cfg, load_target_engine=False)
    accumulation_steps = cfg.training.accumulation_steps
    if cfg.training.attention_backend == "usp":
        accumulation_steps *= cfg.training.sp_ulysses_size * cfg.training.sp_ring_size
    trainer = build_disagg_offline_runtime(
        strategy=cfg.training.strategy,
        feature_store=_offline_store(cfg, retain_on_release=True),
        refs=read_ref_manifest(manifest),
        draft_model=bundle.model,
        target_head=bundle.target_head,
        optimizer_factory=optimizer_factory(cfg),
        run_id=cfg.run_id,
        output_dir=cfg.output_dir,
        ttt_length=cfg.training.ttt_length,
        max_len=cfg.data.max_length,
        batch_size=cfg.training.batch_size,
        accumulation_steps=accumulation_steps,
        num_epochs=cfg.training.num_epochs,
        max_steps=cfg.training.max_steps,
        total_steps=cfg.training.total_steps,
        save_interval=cfg.training.save_interval,
        eval_interval=cfg.training.eval_interval,
        eval_hidden_states_path=cfg.data.eval_hidden_states_path or None,
        logger=logger,
        log_interval=cfg.training.log_interval,
        strategy_kwargs=bundle.strategy_kwargs,
        resume_from=cfg.training.resume_from,
        max_checkpoints=cfg.training.max_checkpoints,
        tp_size=cfg.training.tp_size,
        sp_ulysses_size=cfg.training.sp_ulysses_size,
        sp_ring_size=cfg.training.sp_ring_size,
        use_usp_preprocess=(cfg.training.attention_backend == "usp"),
        seed=cfg.training.seed,
        dataloader_num_workers=_dataloader_num_workers(cfg),
        profiling_options=_profiling_options(cfg),
    )

    def mark_consumed(_step: int) -> None:
        if _primary_rank():
            _write_control(manifest + ".consumed")

    def mark_consumer_failed(exc: BaseException) -> None:
        _publish_control_failure(manifest + ".consumer_failed", exc)

    return TrainingRun(
        trainer=trainer,
        on_success=mark_consumed,
        on_failure=mark_consumer_failed,
    )


def _producer_capture_metadata(cfg: Config):
    from specforge.training.capture_contract import resolve_server_capture_contract

    contract = resolve_server_capture_contract(cfg)
    return (
        list(contract.aux_layer_ids),
        contract.target_hidden_size,
        contract.target_vocab_size,
        contract.draft_vocab_size,
    )


def _build_online(
    cfg: Config,
    *,
    build_model_bundle: Callable,
    prepare_prompts: Callable,
    optimizer_factory: Callable,
    logger: Callable,
):
    from specforge.training.assembly import (
        TrainingRun,
        _dataloader_num_workers,
        _profiling_options,
    )

    strategy = cfg.training.strategy
    from specforge.training.strategies.registry import resolve_strategy

    spec = resolve_strategy(strategy)
    channel_path = _env("DISAGG_REF_CHANNEL")
    if cfg.training.role == "producer":
        _claim_fresh_control_path(channel_path, _ONLINE_CONTROL_SUFFIXES)
    # The producer owns capture and explicit attempt cleanup. The consumer must
    # retain materialized features until DPAckController commits the optimizer
    # boundary and explicitly aborts the acknowledged ids.
    store = _mooncake_store(cfg, retain_on_release=cfg.training.role == "consumer")
    from specforge.runtime.data_plane.feature_store import drain_feature_store_removals
    from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel

    channel = StreamingRefChannel(channel_path)

    if cfg.training.role == "producer":
        from transformers import AutoTokenizer

        from specforge.inference.adapters.server_capture import (
            SGLangServerCaptureAdapter,
        )
        from specforge.launch import build_disagg_online_producer

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.target_model_path,
            cache_dir=cfg.model.cache_dir,
            trust_remote_code=cfg.model.trust_remote_code,
        )
        prompts = prepare_prompts(cfg, tokenizer)
        layers, hidden_size, target_vocab, draft_vocab = _producer_capture_metadata(cfg)
        adapters = [
            SGLangServerCaptureAdapter(
                url,
                store,
                run_id=cfg.run_id,
                strategy=strategy,
                target_model_version=cfg.model.target_model_path,
            )
            for url in _server_urls(cfg)
        ]
        target_repr = spec.assembly.server_target_repr
        peer_wait_timeout_s = float(os.environ.get("DISAGG_PEER_WAIT_TIMEOUT", "1800"))
        high_watermark_override = os.environ.get("DISAGG_IN_FLIGHT_HIGH_WATERMARK")
        in_flight_high_watermark = int(
            high_watermark_override or cfg.runtime.in_flight_high_watermark
        )
        low_watermark_override = os.environ.get("DISAGG_IN_FLIGHT_LOW_WATERMARK")
        # Preserve the legacy one-watermark environment override: when only the
        # old high value is supplied, resume at that same threshold.
        in_flight_low_watermark = (
            int(low_watermark_override)
            if low_watermark_override is not None
            else (
                None
                if high_watermark_override is not None
                else cfg.runtime.in_flight_low_watermark
            )
        )
        _workers, drive = build_disagg_online_producer(
            strategy=strategy,
            prompts=prompts,
            feature_store=store,
            channel=channel,
            feature_source=adapters if len(adapters) > 1 else adapters[0],
            num_rollout_workers=len(adapters),
            run_id=cfg.run_id,
            target_hidden_size=hidden_size,
            target_vocab_size=target_vocab,
            draft_vocab_size=draft_vocab,
            target_repr=target_repr,
            aux_hidden_state_layer_ids=layers,
            prompt_epochs=cfg.training.num_epochs,
            lease=cfg.runtime.producer_lease,
            in_flight_high_watermark=in_flight_high_watermark,
            in_flight_low_watermark=in_flight_low_watermark,
            resident_high_watermark_bytes=(
                int(os.environ["DISAGG_RESIDENT_HIGH_WATERMARK_BYTES"])
                if os.environ.get("DISAGG_RESIDENT_HIGH_WATERMARK_BYTES")
                else cfg.runtime.resident_high_watermark_bytes
            ),
            resident_low_watermark_bytes=(
                int(os.environ["DISAGG_RESIDENT_LOW_WATERMARK_BYTES"])
                if os.environ.get("DISAGG_RESIDENT_LOW_WATERMARK_BYTES")
                else cfg.runtime.resident_low_watermark_bytes
            ),
            feature_store_max_resident_bytes=(
                cfg.runtime.feature_store_max_resident_bytes
            ),
            peer_wait_timeout_s=peer_wait_timeout_s,
        )

        def produce() -> int:
            produced = 0
            primary_exc = None
            try:
                produced = drive(should_stop=channel.consumer_stopped)
                deadline = time.monotonic() + peer_wait_timeout_s
                while not channel.consumer_stopped():
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "producer timed out waiting for the consumer result "
                            f"after {peer_wait_timeout_s:.0f}s"
                        )
                    time.sleep(0.25)
                consumer_failure = channel.consumer_failure()
                if consumer_failure is not None:
                    raise RuntimeError(f"consumer failed: {consumer_failure}")
            except BaseException as exc:
                primary_exc = exc
                try:
                    channel.fail(f"{type(exc).__name__}: {exc}")
                except Exception as signal_exc:
                    print(
                        f"failed to publish producer failure: {signal_exc}",
                        flush=True,
                    )
            cleanup_errors = []
            try:
                reader = StreamingRefChannel(channel_path)
                while True:
                    refs = reader.poll(max_n=1024)
                    if not refs:
                        break
                    for ref in refs:
                        try:
                            store.abort(ref.sample_id, reason="online-attempt-finished")
                        except Exception as exc:
                            cleanup_errors.append(
                                f"{ref.sample_id}: {type(exc).__name__}: {exc}"
                            )
            except Exception as exc:
                cleanup_errors.append(
                    f"published-ref scan: {type(exc).__name__}: {exc}"
                )
            try:
                drain_feature_store_removals(store)
            except Exception as exc:
                cleanup_errors.append(
                    f"pending-remove drain: {type(exc).__name__}: {exc}"
                )
            if primary_exc is not None and cleanup_errors:
                raise RuntimeError(
                    f"producer failed ({type(primary_exc).__name__}: "
                    f"{primary_exc}) and Mooncake cleanup also failed: "
                    f"{cleanup_errors}"
                ) from primary_exc
            if primary_exc is not None:
                raise primary_exc
            if cleanup_errors:
                raise RuntimeError(
                    "producer could not clean all published Mooncake features: "
                    f"{cleanup_errors}"
                )
            return produced

        return TrainingRun(execute=produce)

    from specforge.launch import build_disagg_online_consumer

    bundle = build_model_bundle(cfg, load_target_engine=False)
    trainer = build_disagg_online_consumer(
        strategy=strategy,
        feature_store=store,
        channel=channel,
        draft_model=bundle.model,
        target_head=bundle.target_head,
        optimizer_factory=optimizer_factory(cfg),
        run_id=cfg.run_id,
        output_dir=cfg.output_dir,
        batch_size=cfg.training.batch_size,
        accumulation_steps=cfg.training.accumulation_steps,
        max_steps=cfg.training.max_steps,
        total_steps=cfg.training.total_steps,
        save_interval=cfg.training.save_interval,
        idle_timeout_s=float(os.environ.get("DISAGG_IDLE_TIMEOUT", "0")) or None,
        metadata_db_path=(
            cfg.training.metadata_db_path or os.environ.get("DISAGG_DB") or None
        ),
        logger=logger,
        log_interval=cfg.training.log_interval,
        strategy_kwargs=bundle.strategy_kwargs,
        max_checkpoints=cfg.training.max_checkpoints,
        tp_size=cfg.training.tp_size,
        sp_ulysses_size=cfg.training.sp_ulysses_size,
        sp_ring_size=cfg.training.sp_ring_size,
        inbox_dir=os.environ.get("DISAGG_INBOX_DIR") or None,
        resume_from=cfg.training.resume_from,
        dataloader_num_workers=_dataloader_num_workers(cfg),
        profiling_options=_profiling_options(cfg),
    )

    return TrainingRun(trainer=trainer)


def build_disaggregated_run(
    cfg: Config,
    *,
    build_model_bundle: Callable,
    prepare_prompts: Callable,
    optimizer_factory: Callable,
    logger: Callable,
):
    """Assemble the configured producer or consumer role."""
    if cfg.training.role not in ("producer", "consumer"):
        raise ValueError(
            "disaggregated runs require training.role=producer or consumer"
        )
    try:
        if cfg.mode == "offline":
            return _build_offline(
                cfg,
                build_model_bundle=build_model_bundle,
                optimizer_factory=optimizer_factory,
                logger=logger,
            )
        return _build_online(
            cfg,
            build_model_bundle=build_model_bundle,
            prepare_prompts=prepare_prompts,
            optimizer_factory=optimizer_factory,
            logger=logger,
        )
    except BaseException as exc:
        _publish_role_assembly_failure(cfg, exc)
        raise


__all__ = ["build_disaggregated_run"]
