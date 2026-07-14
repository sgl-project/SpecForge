# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Domain ``Trainer`` wiring: which runtime seam objects get built, with which
args, and that ``.fit()`` delegates to the controller — checked with fakes in
place of the FSDP/model-heavy pieces (no GPU, no real model)."""

import unittest
from types import SimpleNamespace
from unittest import mock


class DomainTrainerWiringTest(unittest.TestCase):
    def _build(self, ref_source, *, durable_ack=False):
        import specforge.training.trainer as tr

        cap = {}

        class FakeLoader:
            def __init__(self, store, **kw):
                cap["loader_store"] = store
                cap["loader_kw"] = kw

        class FakeParallel:
            @classmethod
            def from_distributed(cls, **kw):
                cap["parallel_kw"] = kw
                return "PARALLEL"

        class FakeBackend:
            def __init__(self, parallel, *, optimizer_factory):
                cap["backend_parallel"] = parallel
                cap["optimizer_factory"] = optimizer_factory

            def prepare_model(self, model, *, optimizer_target):
                cap["prepare_model"] = (model, optimizer_target)
                return "WRAPPED"

        class FakeCore:
            def __init__(self, strategy, backend, *, accumulation_steps):
                cap["core"] = (strategy, backend, accumulation_steps)

        class FakeController:
            def __init__(self, core, **kw):
                cap["ctrl_core"] = core
                cap["ctrl_kw"] = kw
                self.global_step = 0

            def fit(self, loader):
                cap["fit"] = loader
                return 42

        dfc_calls = []

        class FakeDataflowController:
            def register_trainer(self, meta):
                dfc_calls.append(("register", meta))
                return "trainer-0"

            def ack_train_refs(self, tid, ids, *, global_step, optimizer_durable):
                dfc_calls.append(("ack", tid, ids, global_step, optimizer_durable))

        spec = SimpleNamespace(
            name="eagle3",
            make_strategy=lambda wrapped, *, target_head: (
                "STRAT",
                wrapped,
                target_head,
            ),
        )
        model = SimpleNamespace(draft_model="DRAFT")
        dfc = FakeDataflowController()

        with mock.patch.multiple(
            tr,
            FeatureDataLoader=FakeLoader,
            ParallelConfig=FakeParallel,
            FSDPTrainingBackend=FakeBackend,
            TrainerCore=FakeCore,
            TrainerController=FakeController,
        ):
            trainer = tr.Trainer(
                spec=spec,
                controller=dfc,
                store="STORE",
                ref_source=ref_source,
                model=model,
                target_head="HEAD",
                optimizer_factory="OPT",
                run_id="run",
                output_dir="/out",
                batch_size=2,
                accumulation_steps=3,
                num_epochs=1,
                max_steps=None,
                total_steps=None,
                save_interval=0,
                logger=None,
                log_interval=50,
                collate_fn="COLLATE",
                durable_ack=durable_ack,
            )
        return trainer, cap, dfc_calls, model

    def test_offline_wiring_matches_assemble_trainer(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch unavailable")
        trainer, cap, dfc_calls, model = self._build({"refs": [1, 2]})

        # Fixed offline refs bypass online queue/ledger state; trainer registered.
        self.assertIn(("register", {"role": "trainer", "run_id": "run"}), dfc_calls)

        # loader built over the store with the spec's name + drop_last + ref_source
        self.assertEqual(cap["loader_store"], "STORE")
        self.assertEqual(cap["loader_kw"]["batch_size"], 2)
        self.assertEqual(cap["loader_kw"]["strategy"], "eagle3")
        self.assertIs(cap["loader_kw"]["drop_last"], True)
        self.assertEqual(cap["loader_kw"]["refs"], [1, 2])

        # optimizer built over the inner draft AFTER FSDP wrap
        self.assertEqual(cap["prepare_model"], (model, "DRAFT"))
        self.assertEqual(cap["core"][0], ("STRAT", "WRAPPED", "HEAD"))
        self.assertEqual(cap["core"][2], 3)  # accumulation_steps threaded

        self.assertIsNone(cap["ctrl_kw"]["ack_fn"])

        # run identity rides the shared checkpoint payload, validated on resume
        self.assertEqual(
            cap["ctrl_kw"]["checkpoint_extra"],
            {"dataset_size": 2, "accumulation_steps": 3},
        )

    def test_fit_delegates_to_controller_over_loader(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch unavailable")
        trainer, cap, _, _ = self._build({"refs": []})
        out = trainer.fit()
        self.assertEqual(out, 42)
        self.assertIs(cap["fit"], trainer.loader)

    def test_streaming_durable_ack_routes_through_controller(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch unavailable")
        _, cap, dfc_calls, _ = self._build(
            {"queue": object()}, durable_ack=True
        )
        ack = cap["ctrl_kw"]["ack_fn"]
        ack(["s1"], 7)
        self.assertIn(("ack", "trainer-0", ["s1"], 7, True), dfc_calls)


if __name__ == "__main__":
    unittest.main()
