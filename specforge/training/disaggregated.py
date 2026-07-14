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


def _build_offline(
    cfg: Config,
    *,
    build_model_bundle: Callable,
    optimizer_factory: Callable,
    logger: Callable,
):
    from specforge.training.assembly import TrainingRun

    if cfg.training.strategy != "eagle3":
        raise NotImplementedError(
            "disaggregated offline ingestion currently supports EAGLE3 features only"
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

            try:
                store = _offline_store(cfg)
                refs = ingest_offline_features(
                    store,
                    cfg.data.hidden_states_path,
                    run_id=cfg.run_id,
                    ttt_length=cfg.training.ttt_length,
                    max_len=cfg.data.max_length,
                )
                write_ref_manifest(refs, manifest)
                _write_control(done)
                _hold_mooncake_producer(manifest)
                return len(refs)
            except BaseException as exc:
                _publish_control_failure(manifest + ".failed", exc)
                raise

        return TrainingRun(execute=produce)

    _wait_for(done, failure_path=manifest + ".failed")
    from specforge.launch import build_disagg_offline_runtime
    from specforge.runtime.data_plane.disagg_ingest import read_ref_manifest

    bundle = build_model_bundle(cfg, load_target_engine=False)
    trainer, loader = build_disagg_offline_runtime(
        strategy="eagle3",
        feature_store=_offline_store(cfg, retain_on_release=True),
        refs=read_ref_manifest(manifest),
        eagle3_model=bundle.model,
        target_head=bundle.target_head,
        optimizer_factory=optimizer_factory(cfg),
        run_id=cfg.run_id,
        output_dir=cfg.output_dir,
        max_len=cfg.data.max_length,
        batch_size=cfg.training.batch_size,
        accumulation_steps=cfg.training.accumulation_steps,
        num_epochs=cfg.training.num_epochs,
        max_steps=cfg.training.max_steps,
        total_steps=cfg.training.total_steps,
        save_interval=cfg.training.save_interval,
        tp_size=1,
        sp_ulysses_size=1,
        sp_ring_size=1,
        logger=logger,
        log_interval=cfg.training.log_interval,
        strategy_kwargs=bundle.strategy_kwargs,
        resume_from=cfg.training.resume_from,
        max_checkpoints=cfg.training.max_checkpoints,
    )

    def consume() -> int:
        try:
            step = trainer.fit(loader)
            if step > 0 and trainer.last_checkpoint_step != step:
                trainer.save_checkpoint(step)
            if _primary_rank():
                _write_control(manifest + ".consumed")
            return step
        except BaseException as exc:
            _publish_control_failure(manifest + ".consumer_failed", exc)
            raise

    return TrainingRun(trainer=trainer, loader=loader, execute=consume)


def _producer_capture_metadata(cfg: Config):
    import json

    from transformers import AutoConfig

    target_cfg = AutoConfig.from_pretrained(
        cfg.model.target_model_path,
        cache_dir=cfg.model.cache_dir,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    target_cfg = getattr(target_cfg, "text_config", target_cfg)
    draft_path = cfg.model.draft_model_config
    if os.path.isdir(draft_path):
        draft_path = os.path.join(draft_path, "config.json")
    with open(draft_path, encoding="utf-8") as stream:
        draft_cfg = json.load(stream)

    if cfg.training.strategy in ("eagle3", "peagle"):
        from specforge.training.assembly import resolve_eagle_capture_layers

        layers = resolve_eagle_capture_layers(cfg, draft_cfg, target_cfg)
    else:
        layers = list(
            (draft_cfg.get("dflash_config") or {}).get("target_layer_ids", [])
        )
    if not layers:
        raise ValueError("draft config does not define target capture layer ids")
    return (
        list(layers),
        int(target_cfg.hidden_size),
        int(target_cfg.vocab_size),
        int(draft_cfg.get("draft_vocab_size") or draft_cfg["vocab_size"]),
    )


def _build_online(
    cfg: Config,
    *,
    build_model_bundle: Callable,
    prepare_prompts: Callable,
    optimizer_factory: Callable,
    logger: Callable,
):
    from specforge.training.assembly import TrainingRun

    strategy = cfg.training.strategy
    if strategy == "peagle":
        raise NotImplementedError("P-EAGLE disaggregated capture is not wired")
    channel_path = _env("DISAGG_REF_CHANNEL")
    if cfg.training.role == "producer":
        _claim_fresh_control_path(channel_path, _ONLINE_CONTROL_SUFFIXES)
    store = _mooncake_store(cfg)
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
        target_repr = "hidden_state" if strategy in ("eagle3", "dspark") else None
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
            peer_wait_timeout_s=float(
                os.environ.get("DISAGG_PEER_WAIT_TIMEOUT", "1800")
            ),
        )

        def produce() -> int:
            produced = drive(should_stop=channel.consumer_stopped)
            consumer_failure = channel.consumer_failure()
            if consumer_failure is not None:
                reason = f"consumer failed: {consumer_failure}"
                try:
                    channel.fail(reason)
                except Exception as signal_exc:
                    print(
                        f"failed to publish producer failure: {signal_exc}",
                        flush=True,
                    )
                raise RuntimeError(reason)
            return produced

        return TrainingRun(execute=produce)

    from specforge.launch import build_disagg_online_consumer

    bundle = build_model_bundle(cfg, load_target_engine=False)
    target_head = None
    if strategy == "eagle3":
        from specforge.modeling.target import TargetHead

        target_head = TargetHead.from_pretrained(
            cfg.model.target_model_path,
            lm_head_key=cfg.model.lm_head_key,
            cache_dir=cfg.model.cache_dir,
            trust_remote_code=cfg.model.trust_remote_code,
        )
    trainer, loader = build_disagg_online_consumer(
        strategy=strategy,
        feature_store=store,
        channel=channel,
        eagle3_model=bundle.model,
        target_head=target_head,
        optimizer_factory=optimizer_factory(cfg),
        run_id=cfg.run_id,
        output_dir=cfg.output_dir,
        batch_size=cfg.training.batch_size,
        accumulation_steps=cfg.training.accumulation_steps,
        num_epochs=1,
        max_steps=cfg.training.max_steps,
        total_steps=cfg.training.total_steps,
        save_interval=cfg.training.save_interval,
        tp_size=1,
        sp_ulysses_size=1,
        sp_ring_size=1,
        idle_timeout_s=float(os.environ.get("DISAGG_IDLE_TIMEOUT", "0")) or None,
        metadata_db_path=(
            cfg.training.metadata_db_path or os.environ.get("DISAGG_DB") or None
        ),
        resume=bool(cfg.training.resume_from),
        logger=logger,
        log_interval=cfg.training.log_interval,
        strategy_kwargs=bundle.strategy_kwargs,
        resume_from=cfg.training.resume_from,
        max_checkpoints=cfg.training.max_checkpoints,
        inbox_dir=os.environ.get("DISAGG_INBOX_DIR") or None,
    )

    def consume() -> int:
        try:
            step = trainer.fit(loader)
            if step > 0 and trainer.last_checkpoint_step != step:
                trainer.save_checkpoint(step)
            if _primary_rank():
                channel.mark_consumer_done()
            return step
        except BaseException as exc:
            try:
                channel.mark_consumer_failed(f"{type(exc).__name__}: {exc}")
            except Exception as signal_exc:
                print(
                    f"failed to publish consumer failure: {signal_exc}",
                    flush=True,
                )
            raise
        finally:
            distributor = getattr(trainer, "ref_distributor", None)
            if distributor is not None:
                distributor.stop()

    return TrainingRun(trainer=trainer, loader=loader, execute=consume)


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
