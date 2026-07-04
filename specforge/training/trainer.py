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
model + spec.make_strategy -> FSDPTrainingBackend -> TrainerCore ->
TrainerController (the trainer seam) behind one object with a ``.fit()``.
Online / offline / disaggregated is invisible here: it is fully absorbed by the
ref source + store the Trainer is handed — the loader IS the stream.
"""

from __future__ import annotations

from typing import Optional

from specforge.runtime.data_plane import FeatureDataLoader, FeatureStore
from specforge.runtime.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.runtime.training.registry import StrategySpec, resolve_strategy
from specforge.runtime.training.trainer import TrainerController, TrainerCore
from specforge.training.checkpoint import CheckpointManager


class Trainer:
    """Domain training lifecycle wrapping the runtime controller/core seam."""

    def __init__(
        self,
        *,
        spec: StrategySpec,
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
        eval_interval: int,
        tp_size: int,
        sp_ulysses_size: int,
        sp_ring_size: int,
        logger,
        log_interval: int,
        collate_fn,
        total_steps: Optional[int] = None,
        per_sample_transform=None,
        durable_ack: bool = True,
        resume_from: Optional[str] = None,
        max_checkpoints: int = 0,
    ):
        # durable_ack off (local_colocated): no durable ack transaction — the
        # loader releases each feature as it consumes it — so the offline enqueue
        # is skipped too (nothing acks or reads that queue).
        if "refs" in ref_source:
            if durable_ack:
                controller.enqueue_offline_refs(ref_source["refs"])
            else:
                from specforge.runtime.contracts import assert_no_tensors

                # metadata-only contract still holds for refs entering the trainer
                for ref in ref_source["refs"]:
                    assert_no_tensors(ref)
        trainer_id = controller.register_trainer({"role": "trainer", "run_id": run_id})
        loader = FeatureDataLoader(
            store,
            **ref_source,
            batch_size=batch_size,
            collate_fn=collate_fn,
            per_sample_transform=per_sample_transform,
            drop_last=True,
            strategy=spec.name,
        )

        dataset_size = len(ref_source["refs"]) if "refs" in ref_source else None
        # Resume: draft weights load BEFORE the FSDP wrap so the optimizer's fp32
        # masters (cloned at build) start from them; this rank's own optimizer/RNG
        # shard (``state['backend']``) loads after the wrap builds the optimizer.
        resume = None
        if resume_from:
            state = CheckpointManager.read_resume_state(resume_from)
            saved_strategy = state.get("strategy")
            if saved_strategy != spec.name:
                raise ValueError(
                    f"checkpoint {resume_from} was written by strategy "
                    f"{saved_strategy!r}; this run trains {spec.name!r}"
                )
            for key, current in (
                ("dataset_size", dataset_size),
                ("accumulation_steps", accumulation_steps),
            ):
                persisted = state.get(key)
                if persisted is not None and persisted != current:
                    raise ValueError(
                        f"checkpoint {resume_from} was written with "
                        f"{key}={persisted} but this run has {key}={current}; "
                        f"resume with the original configuration"
                    )
            saved_weights = state["draft_state_dict"]
            load_result = model.draft_model.load_state_dict(saved_weights, strict=False)
            loaded = len(saved_weights) - len(load_result.unexpected_keys)
            # strict=False must not degrade into loading nothing silently.
            if load_result.unexpected_keys or loaded == 0:
                raise ValueError(
                    f"checkpoint {resume_from} draft weights do not match this "
                    f"model: {loaded}/{len(saved_weights)} keys loaded, "
                    f"unexpected={sorted(load_result.unexpected_keys)}"
                )
            # Mid-epoch position persists in SAMPLES (batch-size independent);
            # only an offline (refs) stream can be repositioned, and a batch-size
            # drift that does not divide the count fails fast (a silent mis-seek
            # would skip or re-train data).
            start_batch = start_samples = 0
            if "refs" in ref_source:
                samples = state.get("epoch_samples", 0)
                if samples % batch_size:
                    raise ValueError(
                        f"checkpoint {resume_from} stopped mid-epoch after "
                        f"{samples} samples, which is not a whole number of "
                        f"batches at batch_size={batch_size}; resume with the "
                        f"batch size the checkpoint was written with"
                    )
                start_batch, start_samples = samples // batch_size, samples
            resume = {
                "backend": state["backend"],
                "global_step": state["global_step"],
                "epoch": state.get("epoch", 0),
                "epoch_batch": start_batch,
                "epoch_samples": start_samples,
            }
            # drop the full draft state before the wrap
            del state, saved_weights

        parallel = ParallelConfig.from_distributed(
            tp_size=tp_size, sp_ulysses_size=sp_ulysses_size, sp_ring_size=sp_ring_size
        )
        backend = FSDPTrainingBackend(parallel, optimizer_factory=optimizer_factory)
        # FSDP-wrap the composite model and build the optimizer over the inner draft
        # AFTER wrapping; the strategy MUST run forward through the wrapped module so
        # FSDP is actually in the forward/backward path (not bypassed at >1 rank).
        wrapped = backend.prepare_model(model, optimizer_target=model.draft_model)
        if resume is not None:
            backend.load_state_dict(resume["backend"])
        strategy = spec.make_strategy(wrapped, target_head=target_head)
        core = TrainerCore(strategy, backend, accumulation_steps=accumulation_steps)
        ack_fn = None
        if durable_ack:

            def ack_fn(ids, step):
                controller.ack_train_refs(
                    trainer_id, ids, global_step=step, optimizer_durable=True
                )

        controller_obj = TrainerController(
            core,
            run_id=run_id,
            output_dir=output_dir,
            num_epochs=num_epochs,
            max_steps=max_steps,
            total_steps=total_steps,
            save_interval=save_interval,
            eval_interval=eval_interval,
            log_interval=log_interval,
            logger=logger,
            ack_fn=ack_fn,
            checkpoint_manager=CheckpointManager(
                output_dir, run_id, max_checkpoints=max_checkpoints
            ),
            checkpoint_extra={
                "dataset_size": dataset_size,
                "accumulation_steps": accumulation_steps,
            },
            start_step=resume["global_step"] if resume else 0,
            start_epoch=resume["epoch"] if resume else 0,
            start_batch=resume["epoch_batch"] if resume else 0,
            start_samples=resume["epoch_samples"] if resume else 0,
        )

        # Runtime pieces exposed for direct callers (and for
        # launch._assemble_trainer's (controller, loader) tuple).
        self.spec = spec
        self.dataflow_controller = controller
        self.trainer_id = trainer_id
        self.backend = backend
        self.core = core
        #: the runtime TrainerController (has fit / evaluate / save_checkpoint)
        self.controller = controller_obj
        #: the FeatureDataLoader -> TrainBatch iterator (the canonical "stream")
        self.loader = loader

    @classmethod
    def from_strategy_name(cls, strategy: str, **kwargs) -> "Trainer":
        """Resolve the :class:`StrategySpec` by name, then assemble."""
        return cls(spec=resolve_strategy(strategy), **kwargs)

    def fit(self, eval_data=None) -> int:
        """Run the training loop over the loader; returns the final global step."""
        return self.controller.fit(self.loader, eval_data=eval_data)

    def evaluate(self, data=None):
        return self.controller.evaluate(self.loader if data is None else data)

    def save_checkpoint(self, step: Optional[int] = None):
        step = self.controller.global_step if step is None else step
        return self.controller.save_checkpoint(step)


__all__ = ["Trainer"]
