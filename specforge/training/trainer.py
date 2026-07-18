# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Domain ``Trainer``: the caller-facing training object.

Composes (ref source + FeatureStore) -> FeatureDataLoader (the data path) and
model + an injected step factory -> FSDPTrainingBackend -> TrainerCore ->
TrainerController (the trainer seam) behind one object with a ``.fit()``.
Online / offline / disaggregated is invisible here: it is fully absorbed by the
ref source + store the Trainer is handed — the loader IS the stream.
"""

from __future__ import annotations

import logging
import sys
from contextlib import nullcontext
from typing import Callable, Mapping, Optional

from specforge.algorithms.common.providers import (
    MODEL_PROVENANCE_CONTRACT_KEY,
    OMITTED_STATE_FINGERPRINT_CONTRACT_KEY,
    StepRuntimeConfig,
    checkpoint_key_fingerprint,
)
from specforge.runtime.data_plane import FeatureDataLoader, FeatureStore
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.checkpoint import CheckpointManager
from specforge.training.controller import TrainerController, TrainerCore

logger = logging.getLogger(__name__)


def _legacy_scheduler_total_steps(state: Mapping[str, object]) -> Optional[int]:
    """Recover the BF16 optimizer horizon from a pre-contract checkpoint.

    Older checkpoints did not copy the effective horizon into the shared
    trainer contract, but their rank-local cosine scheduler persisted both the
    warmup length and post-warmup ``T_max``.  Reading those two immutable
    scheduler fields preserves resume compatibility without guessing from a
    possibly changed ``max_steps`` value.
    """
    backend = state.get("backend")
    if not isinstance(backend, Mapping):
        return None
    optimizer = backend.get("optimizer")
    if not isinstance(optimizer, Mapping):
        return None
    scheduler = optimizer.get("scheduler_state_dict")
    if not isinstance(scheduler, Mapping):
        return None
    after_scheduler = scheduler.get("after_scheduler_dict")
    if not isinstance(after_scheduler, Mapping):
        return None
    warmup_steps = scheduler.get("warmup_epochs")
    cosine_steps = after_scheduler.get("T_max")
    if (
        isinstance(warmup_steps, bool)
        or not isinstance(warmup_steps, int)
        or isinstance(cosine_steps, bool)
        or not isinstance(cosine_steps, int)
        or warmup_steps < 0
        or cosine_steps < 1
    ):
        return None
    return warmup_steps + cosine_steps


class Trainer:
    """Domain training lifecycle wrapping the runtime controller/core seam."""

    def __init__(
        self,
        *,
        algorithm_name: str,
        make_step_strategy: Callable,
        controller,  # runtime.control_plane.DataFlowController (metadata only)
        store: FeatureStore,
        ref_source: dict,  # {"refs": [...]} (offline) | {"queue": q} (online)
        model,
        target_head,
        optimizer_factory,
        run_id: str,
        output_dir: str,
        batch_size: int,
        accumulation_steps: int,
        num_epochs: int,
        max_steps: Optional[int],
        save_interval: int,
        eval_interval: int = 0,
        eval_data_factory: Optional[Callable[[], object]] = None,
        logger,
        log_interval: int,
        collate_fn,
        strategy_kwargs: Optional[Mapping[str, object]] = None,
        total_steps: Optional[int] = None,
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
        ref_source = dict(ref_source)
        data_prepositioned = bool(ref_source.pop("prepositioned", False))
        defer_queue_ack = bool(ref_source.pop("defer_ack_until_durable", False))
        if data_prepositioned and "queue" not in ref_source:
            raise ValueError("prepositioned data is valid only for a queue source")
        if defer_queue_ack and "queue" not in ref_source:
            raise ValueError("deferred queue ack is valid only for a queue source")
        if defer_queue_ack and not durable_ack:
            raise ValueError("deferred queue ack requires durable_ack=True")

        # Fixed offline refs never enter an online staging queue. The loader
        # releases each feature as it consumes it and can re-iterate the same
        # refs on later epochs.
        if "refs" in ref_source:
            from specforge.runtime.contracts import assert_no_tensors

            for ref in ref_source["refs"]:
                assert_no_tensors(ref)
        refs_for_epoch = ref_source.pop("refs_for_epoch", None)
        if refs_for_epoch is not None and (
            "refs" not in ref_source or not callable(refs_for_epoch)
        ):
            raise ValueError(
                "refs_for_epoch must be callable and accompany a refs source"
            )
        trainer_id = controller.register_trainer({"role": "trainer", "run_id": run_id})
        loader = FeatureDataLoader(
            store,
            **ref_source,
            batch_size=batch_size,
            collate_fn=collate_fn,
            per_sample_transform=per_sample_transform,
            drop_last=True,
            strategy=algorithm_name,
            ack=not defer_queue_ack,
            num_workers=dataloader_num_workers,
        )
        if refs_for_epoch is not None:
            expected_refs = len(ref_source["refs"])

            def set_epoch(epoch):
                planned = list(refs_for_epoch(epoch))
                if len(planned) != expected_refs:
                    raise ValueError(
                        f"epoch {epoch} ref plan has {len(planned)} refs; "
                        f"expected {expected_refs}"
                    )
                loader._refs = planned
                loader._seek_batches = 0

            loader.set_epoch = set_epoch

        inferred_dataset_size = (
            len(ref_source["refs"]) if "refs" in ref_source else None
        )
        if dataset_size is None:
            dataset_size = inferred_dataset_size
        elif (
            inferred_dataset_size is not None and dataset_size != inferred_dataset_size
        ):
            raise ValueError(
                f"dataset_size={dataset_size} does not match the "
                f"{inferred_dataset_size} supplied refs"
            )
        if dataset_size is not None:
            from specforge.training.schedule import validate_fixed_accumulation_plan

            validate_fixed_accumulation_plan(
                num_samples=dataset_size,
                batch_size=batch_size,
                accumulation_steps=accumulation_steps,
                num_epochs=num_epochs,
                max_steps=max_steps,
            )
        if isinstance(strategy_kwargs, StepRuntimeConfig):
            algorithm_checkpoint_extra = dict(strategy_kwargs.resume_contract)
            allowed_missing_checkpoint_keys = set(
                strategy_kwargs.allowed_missing_checkpoint_keys
            )
            resolved_strategy_kwargs = dict(strategy_kwargs)
        else:
            algorithm_checkpoint_extra = {}
            allowed_missing_checkpoint_keys = set()
            resolved_strategy_kwargs = dict(strategy_kwargs or {})

        if allowed_missing_checkpoint_keys:
            draft_state_keys = set(model.draft_model.state_dict())
            unknown_allowed_missing = allowed_missing_checkpoint_keys - draft_state_keys
            if unknown_allowed_missing:
                raise ValueError(
                    "algorithm checkpoint policy allows unknown draft keys: "
                    f"{sorted(unknown_allowed_missing)}"
                )
            recorded_fingerprint = algorithm_checkpoint_extra[
                OMITTED_STATE_FINGERPRINT_CONTRACT_KEY
            ]
            live_fingerprint = checkpoint_key_fingerprint(
                model.draft_model,
                allowed_missing_checkpoint_keys,
            )
            if recorded_fingerprint != live_fingerprint:
                raise ValueError(
                    f"{OMITTED_STATE_FINGERPRINT_CONTRACT_KEY!r} does not match "
                    "the live draft model and complete allowed-missing-key policy"
                )

        effective_total_steps = None
        if total_steps is not None or max_steps is not None or dataset_size is not None:
            from specforge.training.schedule import resolve_total_steps

            effective_total_steps = resolve_total_steps(
                total_steps=total_steps,
                max_steps=max_steps,
                num_samples=dataset_size,
                batch_size=batch_size,
                accumulation_steps=accumulation_steps,
                num_epochs=num_epochs,
            )

        configure_schedule = getattr(optimizer_factory, "configure_total_steps", None)
        if effective_total_steps is None and configure_schedule is not None:
            raise ValueError(
                "a configured optimizer on a streaming source requires total_steps "
                "or max_steps"
            )
        if configure_schedule is not None:
            configure_schedule(effective_total_steps)

        standard_checkpoint_extra = {
            "dataset_size": dataset_size,
            "batch_size": batch_size,
            "accumulation_steps": accumulation_steps,
            "num_epochs": num_epochs,
            "effective_total_steps": effective_total_steps,
            "tp_size": tp_size,
            "sp_ulysses_size": sp_ulysses_size,
            "sp_ring_size": sp_ring_size,
        }
        explicit_checkpoint_extra = dict(checkpoint_extra or {})
        duplicate_algorithm_keys = (
            algorithm_checkpoint_extra.keys() & explicit_checkpoint_extra.keys()
        )
        if duplicate_algorithm_keys:
            raise ValueError(
                "checkpoint_extra cannot override algorithm-owned fields: "
                f"{sorted(duplicate_algorithm_keys)}"
            )
        custom_checkpoint_extra = {
            **algorithm_checkpoint_extra,
            **explicit_checkpoint_extra,
        }
        reserved = standard_checkpoint_extra.keys() & custom_checkpoint_extra.keys()
        if reserved:
            raise ValueError(
                "checkpoint_extra cannot override trainer-owned fields: "
                f"{sorted(reserved)}"
            )
        persisted_contract = {**standard_checkpoint_extra, **custom_checkpoint_extra}
        # Resume: draft weights load BEFORE the FSDP wrap so the optimizer's fp32
        # masters (cloned at build) start from them; this rank's own optimizer/RNG
        # shard (``state['backend']``) loads after the wrap builds the optimizer.
        resume = None
        if resume_from:
            state = (
                resume_state
                if resume_state is not None
                else CheckpointManager.read_resume_state(resume_from)
            )
            saved_strategy = state.get("strategy")
            if saved_strategy != algorithm_name:
                raise ValueError(
                    f"checkpoint {resume_from} was written by strategy "
                    f"{saved_strategy!r}; this run trains {algorithm_name!r}"
                )
            import torch.distributed as dist

            world_size = dist.get_world_size() if dist.is_initialized() else 1
            resume_contract = {
                "world_size": world_size,
                **persisted_contract,
            }
            for key, current in resume_contract.items():
                persisted = state.get(key)
                persisted_available = key in state
                comparison_current = current
                comparison_persisted = persisted
                if key == MODEL_PROVENANCE_CONTRACT_KEY and key in state:
                    from specforge.training.provenance import (
                        model_provenance_for_resume_comparison,
                    )

                    (
                        comparison_current,
                        comparison_persisted,
                    ) = model_provenance_for_resume_comparison(
                        current,
                        persisted,
                        cache_root=output_dir,
                    )
                if key == "effective_total_steps" and key not in state:
                    persisted = _legacy_scheduler_total_steps(state)
                    comparison_persisted = persisted
                    persisted_available = persisted is not None
                    if not persisted_available:
                        raise ValueError(
                            f"checkpoint {resume_from} does not record "
                            "effective_total_steps and its saved scheduler does "
                            "not expose a recoverable horizon; the restored "
                            "optimizer schedule cannot be proven to match this run"
                        )
                if key in custom_checkpoint_extra and key not in state:
                    raise ValueError(
                        f"checkpoint {resume_from} does not record required "
                        f"algorithm resume semantic {key}; start a fresh run "
                        "rather than guessing the prior objective"
                    )
                if persisted_available and comparison_persisted != comparison_current:
                    raise ValueError(
                        f"checkpoint {resume_from} was written with "
                        f"{key}={comparison_persisted} but this run has "
                        f"{key}={comparison_current}; "
                        f"resume with the original configuration"
                    )
            saved_weights = state["draft_state_dict"]
            load_result = model.draft_model.load_state_dict(saved_weights, strict=False)
            loaded = len(saved_weights) - len(load_result.unexpected_keys)
            missing_keys = set(load_result.missing_keys)
            disallowed_missing = missing_keys - allowed_missing_checkpoint_keys
            # strict=False exists solely for provider-declared omissions such as
            # EAGLE3's frozen target-copied embedding. Every other mismatch is
            # checkpoint corruption or a different model contract.
            if load_result.unexpected_keys or disallowed_missing or loaded == 0:
                raise ValueError(
                    f"checkpoint {resume_from} draft weights do not match this "
                    f"model: {loaded}/{len(saved_weights)} keys loaded, "
                    f"unexpected={sorted(load_result.unexpected_keys)}, "
                    f"missing={sorted(disallowed_missing)}, "
                    "provider_allowed_missing="
                    f"{sorted(allowed_missing_checkpoint_keys)}"
                )
            # Mid-epoch position persists in SAMPLES (batch-size independent);
            # Offline refs are rebuilt for the saved epoch before seek; a local
            # online queue arrives prepositioned from its deterministic prompt
            # plan. A batch-size drift that does not divide the count fails fast.
            start_batch = start_samples = 0
            samples = int(state.get("epoch_samples", 0))
            if samples < 0:
                raise ValueError(
                    f"checkpoint {resume_from} has negative epoch_samples={samples}"
                )
            if dataset_size is not None and samples > dataset_size:
                raise ValueError(
                    f"checkpoint {resume_from} stopped after {samples} samples, "
                    f"past this run's dataset_size={dataset_size}"
                )
            saved_epoch = int(state.get("epoch", 0))
            if not 0 <= saved_epoch <= num_epochs:
                raise ValueError(
                    f"checkpoint {resume_from} has epoch={saved_epoch}, outside "
                    f"this run's [0, {num_epochs}] epoch range"
                )
            if saved_epoch == num_epochs and samples:
                raise ValueError(
                    f"checkpoint {resume_from} is complete at epoch={saved_epoch} "
                    f"but still records epoch_samples={samples}"
                )
            if "refs" in ref_source or data_prepositioned:
                if samples % batch_size:
                    raise ValueError(
                        f"checkpoint {resume_from} stopped mid-epoch after "
                        f"{samples} samples, which is not a whole number of "
                        f"batches at batch_size={batch_size}; resume with the "
                        f"batch size the checkpoint was written with"
                    )
                start_batch, start_samples = samples // batch_size, samples
                persisted_batch = state.get("epoch_batch")
                if persisted_batch is not None and int(persisted_batch) != start_batch:
                    raise ValueError(
                        f"checkpoint {resume_from} has epoch_batch="
                        f"{persisted_batch} but epoch_samples={samples} implies "
                        f"{start_batch} batches at batch_size={batch_size}"
                    )
            elif samples:
                raise ValueError(
                    f"checkpoint {resume_from} has a streamed mid-epoch position, "
                    "but the queue was not rebuilt as prepositioned"
                )
            resume = {
                "backend": state["backend"],
                "global_step": state["global_step"],
                "epoch": saved_epoch,
                "epoch_batch": start_batch,
                "epoch_samples": start_samples,
            }
            # drop the full draft state before the wrap
            del state, saved_weights

        parallel = ParallelConfig.from_distributed(
            tp_size=tp_size,
            sp_ulysses_size=sp_ulysses_size,
            sp_ring_size=sp_ring_size,
        )
        backend = FSDPTrainingBackend(parallel, optimizer_factory=optimizer_factory)
        # FSDP-wrap the composite model and build the optimizer over the inner draft
        # AFTER wrapping; the strategy MUST run forward through the wrapped module so
        # FSDP is actually in the forward/backward path (not bypassed at >1 rank).
        wrapped = backend.prepare_model(model, optimizer_target=model.draft_model)
        if resume is not None:
            backend.load_state_dict(resume["backend"])
        strategy = make_step_strategy(
            wrapped,
            target_head=target_head,
            **resolved_strategy_kwargs,
        )
        core = TrainerCore(strategy, backend, accumulation_steps=accumulation_steps)
        ack_fn = None
        if durable_ack:

            def ack_fn(ids, step):
                controller.ack_train_refs(
                    trainer_id, ids, global_step=step, optimizer_durable=True
                )
                if defer_queue_ack:
                    ack_ids = getattr(ref_source["queue"], "ack_ids", None)
                    if not callable(ack_ids):
                        raise TypeError(
                            "deferred queue ack requires queue.ack_ids(sample_ids)"
                        )
                    ack_ids(ids)

        controller_obj = TrainerController(
            core,
            run_id=run_id,
            output_dir=output_dir,
            num_epochs=num_epochs,
            max_steps=max_steps,
            total_steps=effective_total_steps,
            save_interval=save_interval,
            eval_interval=eval_interval,
            eval_data_factory=eval_data_factory,
            log_interval=log_interval,
            logger=logger,
            ack_fn=ack_fn,
            checkpoint_manager=CheckpointManager(
                output_dir, run_id, max_checkpoints=max_checkpoints
            ),
            checkpoint_extra=persisted_contract,
            start_step=resume["global_step"] if resume else 0,
            start_epoch=resume["epoch"] if resume else 0,
            start_batch=resume["epoch_batch"] if resume else 0,
            start_samples=resume["epoch_samples"] if resume else 0,
            data_prepositioned=data_prepositioned,
            profiling_options=profiling_options,
        )
        if resume is not None:
            # The loaded checkpoint already represents this durable step. If a
            # completed run or max_steps cap makes fit() a no-op, do not rewrite
            # it as a new checkpoint with altered epoch counters.
            controller_obj.last_checkpoint_step = resume["global_step"]

        # Runtime pieces remain inspectable behind the single Trainer lifecycle.
        self.algorithm_name = algorithm_name
        self.make_step_strategy = make_step_strategy
        self.run_id = run_id
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.max_checkpoints = max_checkpoints
        self.dataflow_controller = controller
        self.trainer_id = trainer_id
        self.backend = backend
        self.core = core
        self._controller = controller_obj
        self._loader = loader
        self._fit_context = fit_context
        self._logger = logger
        self._on_fit_success = on_fit_success
        self._on_fit_failure = on_fit_failure
        self._on_fit_finally = on_fit_finally

    @property
    def global_step(self) -> int:
        return self._controller.global_step

    @property
    def micro_step(self) -> int:
        return self._controller.micro_step

    @property
    def last_checkpoint_step(self) -> Optional[int]:
        return self._controller.last_checkpoint_step

    @property
    def loader(self):
        return self._loader

    def fit(self) -> int:
        """Run training and configured evaluation through one lifecycle."""
        loader_close_attempted = False

        def close_loader() -> None:
            nonlocal loader_close_attempted
            loader_close_attempted = True
            close = getattr(self._loader, "close", None)
            if callable(close):
                close()

        try:
            context = (
                self._fit_context if self._fit_context is not None else nullcontext()
            )
            with context:
                step = self._controller.fit(self._loader)
            if step > 0 and self.last_checkpoint_step != step:
                self.save_checkpoint()
            # A distributed consumer is not successful until prefetch has
            # stopped and every never-yielded lease has an explicit outcome.
            # Run this before on_fit_success publishes consumer_done.
            close_loader()
            if self._on_fit_success is not None:
                self._on_fit_success(step)
            return step
        except BaseException as exc:
            if self._on_fit_failure is not None:
                self._on_fit_failure(exc)
            raise
        finally:
            primary_exception = sys.exception()
            cleanup_errors: list[tuple[str, BaseException]] = []

            def capture_cleanup(label: str, action) -> None:
                try:
                    action()
                except BaseException as cleanup_error:
                    cleanup_errors.append((label, cleanup_error))

            if not loader_close_attempted:
                capture_cleanup("FeatureDataLoader.close", close_loader)
            capture_cleanup(
                "TrainerController.close_profiler",
                self._controller.close_profiler,
            )
            if self._on_fit_finally is not None:
                capture_cleanup("on_fit_finally", self._on_fit_finally)
            close_logger = getattr(self._logger, "close", None)
            if callable(close_logger):
                capture_cleanup("logger.close", close_logger)

            if cleanup_errors and primary_exception is not None:
                for label, cleanup_error in cleanup_errors:
                    note = (
                        f"{label} also failed during cleanup: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                    primary_exception.add_note(note)
                    logger.error(
                        "%s",
                        note,
                        exc_info=(
                            type(cleanup_error),
                            cleanup_error,
                            cleanup_error.__traceback__,
                        ),
                    )
            elif cleanup_errors:
                label, cleanup_error = cleanup_errors[0]
                for other_label, other_error in cleanup_errors[1:]:
                    cleanup_error.add_note(
                        f"{other_label} also failed during cleanup: "
                        f"{type(other_error).__name__}: {other_error}"
                    )
                cleanup_error.add_note(f"cleanup owner: {label}")
                raise cleanup_error

    def save_checkpoint(self):
        """Persist the current optimizer step through the one trainer surface."""
        return self._controller.save_checkpoint(self.global_step)

    def evaluate(self, data=None):
        """Run one eval pass, defaulting to the configured eval source.

        The training loader remains the compatibility fallback when a caller
        assembled ``Trainer`` directly without a separate eval source.
        """
        if data is not None:
            return self._controller.evaluate(data)
        if self._controller.eval_data_factory is not None:
            return self._controller.evaluate_configured()
        return self._controller.evaluate(self._loader)


__all__ = ["Trainer"]
