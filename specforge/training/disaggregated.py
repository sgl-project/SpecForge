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

import json
import os
import time
from typing import Callable, Optional, Sequence

from specforge.algorithms.registry import AlgorithmRegistration
from specforge.config import Config

_PRODUCER_CLAIM_SUFFIX = ".producer_claim"
_ONLINE_SCHEDULE_SUFFIX = ".schedule.json"
_ONLINE_CONTROL_SUFFIXES = (
    ".closed",
    ".consumed_count",
    ".failed",
    ".consumer_done",
    ".consumer_failed",
    ".consumer_quantum",
    _ONLINE_SCHEDULE_SUFFIX,
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
    deployment = cfg.deployment.disaggregated
    if deployment is not None and deployment.server_urls:
        return list(deployment.server_urls)
    raw = os.environ.get("DISAGG_SERVER_URLS") or os.environ.get("DISAGG_SERVER_URL")
    if not raw:
        raise ValueError(
            "online producer requires deployment.disaggregated.server_urls, "
            "DISAGG_SERVER_URLS, or DISAGG_SERVER_URL"
        )
    return [item.strip() for item in raw.split(",") if item.strip()]


def _consumer_database_path(cfg: Config) -> Optional[str]:
    configured = os.environ.get("DISAGG_DB")
    if configured:
        return configured
    deployment = cfg.deployment.disaggregated
    if deployment is None:
        return None
    state_dir = deployment.consumer_state_dir or deployment.control_dir
    return os.path.join(state_dir, "consumer.sqlite")


def _online_schedule_payload(cfg: Config, *, num_prompts: int) -> dict:
    """Describe the exact finite online schedule prepared by the producer."""
    from specforge.training.schedule import resolve_online_total_steps

    trainer = cfg.deployment.trainer
    dp_size = trainer.nnodes * trainer.nproc_per_node
    total_steps = resolve_online_total_steps(
        num_prompts=num_prompts,
        prompt_epochs=cfg.training.num_epochs,
        dp_size=dp_size,
        batch_size=cfg.training.batch_size,
        accumulation_steps=cfg.training.accumulation_steps,
    )
    return {
        "version": 1,
        "total_steps": total_steps,
        "num_prompts": num_prompts,
        "prompt_epochs": cfg.training.num_epochs,
        "prompt_seed": cfg.training.seed,
        "dp_size": dp_size,
        "batch_size": cfg.training.batch_size,
        "accumulation_steps": cfg.training.accumulation_steps,
    }


def _read_online_total_steps(cfg: Config, channel_path: str) -> int:
    """Read and validate the producer-owned online schedule contract."""
    schedule_path = channel_path + _ONLINE_SCHEDULE_SUFFIX
    _wait_for(
        schedule_path,
        timeout_s=_optional_timeout_s("DISAGG_PEER_WAIT_TIMEOUT"),
        failure_path=channel_path + ".failed",
    )
    try:
        with open(schedule_path, encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"invalid online schedule record {schedule_path}: {exc}"
        ) from exc

    trainer = cfg.deployment.trainer
    expected = {
        "version": 1,
        "prompt_epochs": cfg.training.num_epochs,
        "prompt_seed": cfg.training.seed,
        "dp_size": trainer.nnodes * trainer.nproc_per_node,
        "batch_size": cfg.training.batch_size,
        "accumulation_steps": cfg.training.accumulation_steps,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"online schedule record {schedule_path} does not match this "
            f"consumer configuration: {mismatches}"
        )
    total_steps = payload.get("total_steps")
    if not isinstance(total_steps, int) or total_steps < 1:
        raise ValueError(
            f"online schedule record {schedule_path} has invalid "
            f"total_steps={total_steps!r}"
        )
    return total_steps


def _optional_timeout_s(name: str) -> Optional[float]:
    """Read an explicitly configured positive timeout.

    Long-running producer/consumer coordination is unbounded by default.  A
    deployment may opt into a terminal timeout through the corresponding
    environment variable, but zero and negative values are almost certainly a
    configuration mistake and match neither the schema nor useful semantics.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        timeout_s = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number, got {raw!r}") from exc
    if timeout_s <= 0:
        raise ValueError(f"{name} must be positive, got {raw!r}")
    return timeout_s


def _wait_for(
    path: str,
    *,
    timeout_s: Optional[float] = None,
    failure_path: Optional[str] = None,
) -> None:
    deadline = time.monotonic() + timeout_s if timeout_s is not None else None
    while not os.path.exists(path):
        failure = _read_control(failure_path) if failure_path else None
        if failure is not None:
            raise RuntimeError(
                f"remote role failed while waiting for {path}: {failure}"
            )
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.25)


def _hold_mooncake_producer(manifest: str) -> None:
    if os.environ.get("DISAGG_BACKEND", "shared_dir") != "mooncake":
        return
    consumed = manifest + ".consumed"
    _wait_for(
        consumed,
        timeout_s=_optional_timeout_s("DISAGG_PRODUCER_HOLD_S"),
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
    algorithm: AlgorithmRegistration,
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
                    algorithm_name=algorithm.name,
                    build_reader=algorithm.providers.offline_for(
                        cfg.model.input_modality
                    ).build_reader,
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
                # Publish the failure sentinel BEFORE the Mooncake abort sweep:
                # the sweep can take minutes over thousands of refs, and a
                # supervisor SIGKILL at grace expiry mid-sweep must not leave a
                # remote consumer waiting forever on a sentinel that never
                # arrives. A cleanup failure below overwrites this record with
                # the combined error.
                _publish_control_failure(manifest + ".failed", primary_exc)

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
                raise primary_exc
            if cleanup_exc is not None:
                _publish_control_failure(manifest + ".failed", cleanup_exc)
                raise cleanup_exc
            return produced

        return TrainingRun(execute=produce)

    _wait_for(done, failure_path=manifest + ".failed")
    from specforge.launch import build_disagg_offline_runtime
    from specforge.runtime.data_plane.disagg_ingest import read_ref_manifest

    bundle = build_model_bundle(cfg)
    accumulation_steps = cfg.training.accumulation_steps
    if cfg.training.attention_backend == "usp":
        accumulation_steps *= cfg.training.sp_ulysses_size * cfg.training.sp_ring_size
    trainer = build_disagg_offline_runtime(
        algorithm=algorithm,
        modality=cfg.model.input_modality,
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
        dataloader_num_workers=_dataloader_num_workers(cfg, algorithm),
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


def _producer_capture_metadata(cfg: Config, algorithm: AlgorithmRegistration):
    from specforge.training.capture_contract import resolve_server_capture_contract

    contract = resolve_server_capture_contract(cfg, algorithm=algorithm)
    return (
        list(contract.aux_layer_ids),
        contract.target_hidden_size,
        contract.target_vocab_size,
        contract.draft_vocab_size,
    )


def _build_online(
    cfg: Config,
    *,
    algorithm: AlgorithmRegistration,
    build_model_bundle: Callable,
    prepare_prompts: Callable,
    optimizer_factory: Callable,
    logger: Callable,
):
    from specforge.training.assembly import (
        TrainingRun,
        _dataloader_num_workers,
        _load_input_tools,
        _profiling_options,
    )

    modality = cfg.model.input_modality
    streaming = algorithm.providers.server_streaming_for(modality)
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
        from specforge.inference.adapters.server_capture import (
            ServerCaptureSchema,
            SGLangServerCaptureAdapter,
        )
        from specforge.launch import build_disagg_online_producer
        from specforge.training.model_loading import resolve_draft_config

        input_adapter = streaming.create_input_adapter(cfg)
        input_tools = _load_input_tools(
            cfg,
            algorithm,
            input_adapter=input_adapter,
        )
        draft_config = resolve_draft_config(
            cfg,
            provider=algorithm.providers.model.draft_config,
        )
        if input_adapter is None:
            prompts = prepare_prompts(
                cfg,
                input_tools,
                draft_config=draft_config,
            )
        else:
            prompts = input_adapter.prepare_prompts(
                cfg,
                input_tools,
                draft_config=draft_config,
            )
        if cfg.training.total_steps is None and cfg.training.max_steps is None:
            schedule = _online_schedule_payload(cfg, num_prompts=len(prompts))
            _write_control(
                channel_path + _ONLINE_SCHEDULE_SUFFIX,
                json.dumps(schedule, sort_keys=True),
            )
        layers, hidden_size, target_vocab, draft_vocab = _producer_capture_metadata(
            cfg, algorithm
        )
        layout = streaming.layout
        capture_schema = ServerCaptureSchema(
            aux_feature=layout.aux_feature,
            last_hidden_feature=layout.last_hidden_feature,
            passthrough=layout.passthrough,
            attention_mask_feature=layout.attention_mask_feature,
        )
        adapters = [
            SGLangServerCaptureAdapter(
                url,
                store,
                run_id=cfg.run_id,
                algorithm=algorithm.name,
                schema=capture_schema,
                request_input_adapter=input_adapter,
                target_model_version=cfg.model.target_model_path,
            )
            for url in _server_urls(cfg)
        ]
        target_repr = streaming.target_representation
        peer_wait_timeout_s = _optional_timeout_s("DISAGG_PEER_WAIT_TIMEOUT")
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
            algorithm=algorithm,
            modality=modality,
            prompts=prompts,
            feature_store=store,
            channel=channel,
            feature_source=adapters if len(adapters) > 1 else adapters[0],
            num_rollout_workers=len(adapters),
            producer_concurrency=cfg.runtime.producer_concurrency,
            run_id=cfg.run_id,
            target_hidden_size=hidden_size,
            target_vocab_size=target_vocab,
            draft_vocab_size=draft_vocab,
            target_repr=target_repr,
            aux_hidden_state_layer_ids=layers,
            prompt_epochs=cfg.training.num_epochs,
            prompt_seed=cfg.training.seed,
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
                deadline = (
                    time.monotonic() + peer_wait_timeout_s
                    if peer_wait_timeout_s is not None
                    else None
                )
                while not channel.consumer_stopped():
                    if deadline is not None and time.monotonic() >= deadline:
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
                store.discard_external_attempts(
                    reason="online-attempt-unadopted-capture"
                )
            except Exception as exc:
                cleanup_errors.append(
                    f"unadopted-capture cleanup: {type(exc).__name__}: {exc}"
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

    total_steps = cfg.training.total_steps
    if total_steps is None and cfg.training.max_steps is None:
        total_steps = _read_online_total_steps(cfg, channel_path)
    bundle = build_model_bundle(cfg)
    trainer = build_disagg_online_consumer(
        algorithm=algorithm,
        modality=modality,
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
        total_steps=total_steps,
        save_interval=cfg.training.save_interval,
        idle_timeout_s=float(os.environ.get("DISAGG_IDLE_TIMEOUT", "0")) or None,
        metadata_db_path=_consumer_database_path(cfg),
        logger=logger,
        log_interval=cfg.training.log_interval,
        strategy_kwargs=bundle.strategy_kwargs,
        max_checkpoints=cfg.training.max_checkpoints,
        tp_size=cfg.training.tp_size,
        sp_ulysses_size=cfg.training.sp_ulysses_size,
        sp_ring_size=cfg.training.sp_ring_size,
        inbox_dir=os.environ.get("DISAGG_INBOX_DIR") or None,
        resume_from=cfg.training.resume_from,
        dataloader_num_workers=_dataloader_num_workers(cfg, algorithm),
        profiling_options=_profiling_options(cfg),
    )

    return TrainingRun(trainer=trainer)


def build_disaggregated_run(
    cfg: Config,
    *,
    algorithm: AlgorithmRegistration,
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
                algorithm=algorithm,
                build_model_bundle=build_model_bundle,
                optimizer_factory=optimizer_factory,
                logger=logger,
            )
        return _build_online(
            cfg,
            algorithm=algorithm,
            build_model_bundle=build_model_bundle,
            prepare_prompts=prepare_prompts,
            optimizer_factory=optimizer_factory,
            logger=logger,
        )
    except BaseException as exc:
        _publish_role_assembly_failure(cfg, exc)
        raise


__all__ = ["build_disaggregated_run"]
