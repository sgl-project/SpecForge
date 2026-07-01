# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Domain ``Trainer``: the caller-facing training object (Phase B).

``Trainer`` composes — behind one object with a ``.fit()`` — exactly what
``launch._assemble_trainer`` wired inline before:

    ref source + FeatureStore
        -> FeatureDataLoader(transform, collate)              (the data path)
    model + spec.make_strategy
        -> FSDPTrainingBackend.prepare_model (FSDP wrap)
        -> TrainerCore -> TrainerController                    (the trainer seam)

The runtime seam (``TrainerController`` / ``TrainerCore`` /
``DraftTrainStrategy`` / ``FSDPTrainingBackend``) is byte-for-byte unchanged —
this is the domain facade over it, so ``launch._assemble_trainer`` now delegates
here (one wiring path, no fork). The online / offline / disaggregated distinction
is invisible to ``Trainer``: it is fully absorbed by the (ref source +
``FeatureStore``) it is handed, behind ``FeatureDataLoader -> TrainBatch``. There
is NO ``HiddenStateStream`` — the loader is the stream.
"""

from __future__ import annotations

from typing import Optional

from specforge.runtime.data_plane import FeatureDataLoader, FeatureStore
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.strategies.registry import StrategySpec, resolve_strategy
from specforge.training.controller import TrainerController, TrainerCore


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
        # ``durable_ack`` gates the control-plane bookkeeping a colocated run does
        # not need (Phase C, local_colocated): with it off there is no durable ack
        # transaction — the loader releases each feature as it consumes it — so we
        # skip the offline enqueue too (nothing acks or reads that queue).
        #
        # Offline = a fixed, re-iterable ref set: record committed state so the ack
        # lookup works (num_epochs > 1 then re-iterates). Online streams refs through
        # a queue and commits them elsewhere (rollout / channel).
        if durable_ack and "refs" in ref_source:
            controller.enqueue_offline_refs(ref_source["refs"])
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

        # Resume restores the trainable draft weights BEFORE the FSDP wrap so the
        # optimizer's fp32 master (cloned at build) starts from them; the optimizer
        # + scheduler + RNG are restored just after the wrap builds the optimizer.
        resume_state = self._load_resume_state(resume_from) if resume_from else None
        if resume_state is not None:
            model.draft_model.load_state_dict(
                resume_state["draft_state_dict"], strict=False
            )

        parallel = ParallelConfig.from_distributed(
            tp_size=tp_size, sp_ulysses_size=sp_ulysses_size, sp_ring_size=sp_ring_size
        )
        backend = FSDPTrainingBackend(parallel, optimizer_factory=optimizer_factory)
        # FSDP-wrap the composite model and build the optimizer over the inner draft
        # AFTER wrapping; the strategy MUST run forward through the wrapped module so
        # FSDP is actually in the forward/backward path (not bypassed at >1 rank).
        wrapped = backend.prepare_model(model, optimizer_target=model.draft_model)
        if resume_state is not None:
            backend.load_state_dict(
                {
                    "optimizer": resume_state.get("optimizer_state_dict"),
                    "rng": resume_state.get("rng_state"),
                }
            )
        strategy = spec.make_strategy(wrapped, target_head=target_head)
        core = TrainerCore(strategy, backend, accumulation_steps=accumulation_steps)
        # Durable ack transaction at each optimizer-step boundary. Off for
        # local_colocated: the loader releases features as it consumes them.
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
            max_checkpoints=max_checkpoints,
            start_step=resume_state["global_step"] if resume_state else 0,
            start_epoch=resume_state["epoch"] if resume_state else 0,
        )

        # The runtime pieces, exposed for callers that still want them directly
        # (and for launch._assemble_trainer's (controller, loader) tuple).
        self.spec = spec
        self.dataflow_controller = controller
        self.trainer_id = trainer_id
        self.backend = backend
        self.core = core
        #: the runtime TrainerController (has fit / evaluate / save_checkpoint)
        self.controller = controller_obj
        #: the FeatureDataLoader -> TrainBatch iterator (the canonical "stream")
        self.loader = loader

    @staticmethod
    def _load_resume_state(resume_from: str) -> dict:
        """Read a checkpoint's ``training_state.pt`` (accepts a dir or ``file://`` uri)."""
        import os

        import torch

        path = (
            resume_from[len("file://") :]
            if resume_from.startswith("file://")
            else resume_from
        )
        if os.path.isdir(path):
            path = os.path.join(path, "training_state.pt")
        return torch.load(path, map_location="cpu", weights_only=False)

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
