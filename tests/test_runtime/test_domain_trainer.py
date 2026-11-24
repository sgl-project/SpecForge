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

from specforge.distributed import Topology


class DomainTrainerWiringTest(unittest.TestCase):
    def _build(
        self,
        ref_source,
        *,
        durable_ack=False,
        fit_step=42,
        checkpointed_step=None,
        fit_context=None,
        fit_error=None,
        loader_close_error=None,
        save_error=None,
        events=None,
        on_fit_success=None,
        on_fit_failure=None,
        on_fit_finally=None,
        topology=Topology(rank=3, world_size=8),
        store="STORE",
    ):
        import specforge.training.trainer as tr

        cap = {}
        events = [] if events is None else events

        class FakeLoader:
            def __init__(self, store, **kw):
                cap["loader_store"] = store
                cap["loader_kw"] = kw
                cap["loader"] = self

            def close(self):
                cap["loader_closes"] = cap.get("loader_closes", 0) + 1
                if loader_close_error is not None:
                    raise loader_close_error

        class FakeParallel:
            @classmethod
            def from_topology(cls, topology, **kw):
                cap["parallel_topology"] = topology
                cap["parallel_kw"] = kw
                return SimpleNamespace()

        class FakeBackend:
            def __init__(self, parallel, *, optimizer_factory):
                cap["backend_parallel"] = parallel
                cap["optimizer_factory"] = optimizer_factory

            def prepare_model(self, model, *, optimizer_target, frozen, blocks):
                cap["prepare_model"] = (model, optimizer_target, frozen, blocks)
                return "WRAPPED"

        class FakeRunner:
            def __init__(self, program, backend, spec):
                cap["runner"] = (program, backend, spec)

        class FakeController:
            def __init__(self, runner, **kw):
                cap["ctrl_runner"] = runner
                cap["ctrl_kw"] = kw
                self.global_step = 0
                self.micro_step = 0
                self.last_checkpoint_step = checkpointed_step

            def fit(self, loader):
                cap["fit"] = loader
                events.append("fit")
                if fit_error is not None:
                    raise fit_error
                self.global_step = fit_step
                return fit_step

            def save_checkpoint(self, step):
                events.append("save")
                if save_error is not None:
                    raise save_error
                cap.setdefault("saves", []).append(step)
                self.last_checkpoint_step = step
                return SimpleNamespace(global_step=step)

            def close_profiler(self):
                cap["profiler_closes"] = cap.get("profiler_closes", 0) + 1

        dfc_calls = []

        class FakeDataflowController:
            def register_trainer(self, meta):
                dfc_calls.append(("register", meta))
                return "trainer-0"

            def ack_train_refs(self, tid, ids, *, global_step, optimizer_durable):
                dfc_calls.append(("ack", tid, ids, global_step, optimizer_durable))

        model = SimpleNamespace(draft_model="DRAFT", frozen=("TABLE",))
        declared = SimpleNamespace(partition=SimpleNamespace(blocks=("Block",)))
        cap["program_spec"] = declared
        dfc = FakeDataflowController()

        with mock.patch.multiple(
            tr,
            FeatureDataLoader=FakeLoader,
            ParallelConfig=FakeParallel,
            FSDPTrainingBackend=FakeBackend,
            DataParallelRunner=FakeRunner,
            TrainerController=FakeController,
        ):
            trainer = tr.Trainer(
                algorithm_name="eagle3",
                controller=dfc,
                store=store,
                ref_source=ref_source,
                model=model,
                program_spec=declared,
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
                fit_context=fit_context,
                on_fit_success=on_fit_success,
                on_fit_failure=on_fit_failure,
                on_fit_finally=on_fit_finally,
                topology=topology,
            )
        return trainer, cap, dfc_calls, model

    def test_pinned_receive_requests_device_tensors_from_the_loader(self):
        import torch

        from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
        from tests.test_runtime.test_mooncake_store import _FakeMooncakeStore

        store = MooncakeFeatureStore(
            store=_FakeMooncakeStore(), receive_buffers="pinned"
        )
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            with mock.patch.dict("os.environ", {"LOCAL_RANK": "2"}):
                _, captured, _, _ = self._build({"refs": list(range(6))}, store=store)
        self.assertEqual(captured["loader_kw"]["device"], torch.device("cuda", 2))

    def test_pinned_receive_keeps_loader_on_explicit_cpu_device(self):
        import torch

        from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
        from tests.test_runtime.test_mooncake_store import _FakeMooncakeStore

        store = MooncakeFeatureStore(
            store=_FakeMooncakeStore(), receive_buffers="pinned"
        )
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.dict(
                "os.environ", {"SPECFORGE_DEVICE": "cpu", "LOCAL_RANK": "0"}
            ),
        ):
            _, captured, _, _ = self._build({"refs": list(range(6))}, store=store)
        self.assertEqual(captured["loader_kw"]["device"], "cpu")

    def test_offline_wiring_matches_assemble_trainer(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch unavailable")
        refs = list(range(6))
        trainer, cap, dfc_calls, model = self._build({"refs": refs})

        # Fixed offline refs bypass online queue/ledger state; trainer registered.
        self.assertIn(("register", {"role": "trainer", "run_id": "run"}), dfc_calls)

        # loader built over the store with the spec's name + drop_last + ref_source
        self.assertEqual(cap["loader_store"], "STORE")
        self.assertEqual(cap["loader_kw"]["batch_size"], 2)
        self.assertEqual(cap["loader_kw"]["strategy"], "eagle3")
        self.assertIs(cap["loader_kw"]["drop_last"], True)
        self.assertEqual(cap["loader_kw"]["refs"], refs)

        # optimizer built over the inner draft AFTER FSDP wrap; the backend is
        # told what stays replicated and what stays whole
        self.assertEqual(cap["prepare_model"], (model, "DRAFT", ("TABLE",), ("Block",)))
        # the runner executes the wrapped model under the declarations it was given
        program, backend, spec = cap["runner"]
        self.assertEqual(program, "WRAPPED")
        self.assertIs(backend, trainer.backend)
        self.assertIs(spec, cap["program_spec"])

        # the run loop drives that runner and holds neither the program nor the backend
        self.assertIs(cap["ctrl_runner"], trainer.runner)
        self.assertEqual(cap["ctrl_kw"]["strategy_name"], "eagle3")
        self.assertEqual(cap["ctrl_kw"]["accumulation_steps"], 3)
        self.assertEqual(cap["ctrl_kw"]["data_parallel_size"], 8)
        # the run loop's rank and world size are the topology's
        self.assertEqual((cap["ctrl_kw"]["rank"], cap["ctrl_kw"]["world_size"]), (3, 8))

        self.assertIsNone(cap["ctrl_kw"]["ack_fn"])
        self.assertEqual(cap["parallel_topology"], Topology(rank=3, world_size=8))
        self.assertEqual(cap["parallel_kw"], {})

        # run identity rides the shared checkpoint payload, validated on resume
        self.assertEqual(
            cap["ctrl_kw"]["checkpoint_extra"],
            {
                "dataset_size": 6,
                "batch_size": 2,
                "accumulation_steps": 3,
                "num_epochs": 1,
                "effective_total_steps": 1,
                "tp_size": 1,
                "sp_ulysses_size": 1,
                "sp_ring_size": 1,
            },
        )

    def test_throughput_scale_counts_only_data_parallel_ranks(self):
        _, cap, _, _ = self._build(
            {"refs": list(range(6))},
            topology=Topology(world_size=8, tp_size=2, sp_ulysses_size=2),
        )
        # 8 ranks in groups of tp 2 x sp 2 receive 2 different sample streams
        self.assertEqual(cap["ctrl_kw"]["data_parallel_size"], 2)
        self.assertEqual(
            {
                key: cap["ctrl_kw"]["checkpoint_extra"][key]
                for key in ("tp_size", "sp_ulysses_size", "sp_ring_size")
            },
            {"tp_size": 2, "sp_ulysses_size": 2, "sp_ring_size": 1},
        )

    def test_fixed_refs_reject_partial_accumulation_during_assembly(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch unavailable")

        with self.assertRaisesRegex(
            ValueError, "ends with incomplete gradient accumulation"
        ):
            self._build({"refs": list(range(8))})

    def test_fit_delegates_to_controller_over_loader(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch unavailable")
        trainer, cap, _, _ = self._build({"refs": list(range(6))})
        out = trainer.fit()
        self.assertEqual(out, 42)
        self.assertIs(cap["fit"], cap["loader"])
        self.assertEqual(cap["saves"], [42])
        self.assertEqual(cap["loader_closes"], 1)

        # Re-entering at an already durable step does not write it twice.
        self.assertEqual(trainer.fit(), 42)
        self.assertEqual(cap["saves"], [42])
        self.assertEqual(cap["loader_closes"], 2)

    def test_offline_loader_rebuilds_the_ref_plan_on_set_epoch(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch unavailable")
        calls = []

        def refs_for_epoch(epoch):
            calls.append(epoch)
            return [f"e{epoch}-{index}" for index in range(6)]

        _, cap, _, _ = self._build(
            {
                "refs": refs_for_epoch(0),
                "refs_for_epoch": refs_for_epoch,
            }
        )
        cap["loader"].set_epoch(3)
        self.assertEqual(calls, [0, 3])
        self.assertEqual(
            cap["loader"]._refs,
            ["e3-0", "e3-1", "e3-2", "e3-3", "e3-4", "e3-5"],
        )
        self.assertEqual(cap["loader"]._seek_batches, 0)

    def test_fit_exits_topology_context_before_final_checkpoint(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch unavailable")
        events = []

        class FitContext:
            def __enter__(self):
                events.append("enter")

            def __exit__(self, *_exc):
                events.append("exit")

        trainer, _, _, _ = self._build(
            {"refs": list(range(6))},
            fit_context=FitContext(),
            events=events,
        )
        self.assertEqual(trainer.fit(), 42)
        self.assertEqual(events, ["enter", "fit", "exit", "save"])

    def test_fit_owns_terminal_success_and_cleanup_after_checkpoint(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch unavailable")
        events = []
        trainer, _, _, _ = self._build(
            {"refs": list(range(6))},
            events=events,
            on_fit_success=lambda step: events.append(f"success:{step}"),
            on_fit_failure=lambda exc: events.append(f"failure:{exc}"),
            on_fit_finally=lambda: events.append("finally"),
        )
        self.assertEqual(trainer.fit(), 42)
        self.assertEqual(events, ["fit", "save", "success:42", "finally"])

    def test_fit_owns_terminal_failure_and_cleanup(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch unavailable")
        events = []
        error = RuntimeError("fit failed")
        trainer, cap, _, _ = self._build(
            {"refs": list(range(6))},
            fit_error=error,
            events=events,
            on_fit_success=lambda step: events.append(f"success:{step}"),
            on_fit_failure=lambda exc: events.append(f"failure:{exc}"),
            on_fit_finally=lambda: events.append("finally"),
        )
        with self.assertRaises(RuntimeError) as raised:
            trainer.fit()
        self.assertIs(raised.exception, error)
        self.assertEqual(events, ["fit", "failure:fit failed", "finally"])
        self.assertEqual(cap["loader_closes"], 1)

    def test_loader_cleanup_failure_does_not_mask_primary_fit_error(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch unavailable")
        events = []
        primary = RuntimeError("optimizer failed first")
        cleanup = RuntimeError("prefetch worker stayed blocked")
        trainer, cap, _, _ = self._build(
            {"refs": list(range(6))},
            fit_error=primary,
            loader_close_error=cleanup,
            events=events,
            on_fit_failure=lambda exc: events.append(f"failure:{exc}"),
            on_fit_finally=lambda: events.append("finally"),
        )

        with (
            self.assertLogs("specforge.training.trainer", level="ERROR") as logs,
            self.assertRaises(RuntimeError) as raised,
        ):
            trainer.fit()

        self.assertIs(raised.exception, primary)
        self.assertEqual(cap["loader_closes"], 1)
        self.assertEqual(events, ["fit", "failure:optimizer failed first", "finally"])
        self.assertTrue(
            any("prefetch worker stayed blocked" in note for note in primary.__notes__)
        )
        self.assertIn("prefetch worker stayed blocked", "\n".join(logs.output))

    def test_loader_close_failure_signals_failure_before_final_cleanup(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch unavailable")
        events = []
        cleanup = RuntimeError("prefetch close failed")
        trainer, cap, _, _ = self._build(
            {"refs": list(range(6))},
            loader_close_error=cleanup,
            events=events,
            on_fit_success=lambda step: events.append(f"success:{step}"),
            on_fit_failure=lambda exc: events.append(f"failure:{exc}"),
            on_fit_finally=lambda: events.append("finally"),
        )

        with self.assertRaises(RuntimeError) as raised:
            trainer.fit()

        self.assertIs(raised.exception, cleanup)
        self.assertEqual(cap["loader_closes"], 1)
        self.assertEqual(
            events,
            ["fit", "save", "failure:prefetch close failed", "finally"],
        )

    def test_zero_step_skips_final_checkpoint(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch unavailable")
        trainer, cap, _, _ = self._build({"refs": list(range(6))}, fit_step=0)
        self.assertEqual(trainer.fit(), 0)
        self.assertNotIn("saves", cap)

    def test_final_checkpoint_failure_propagates(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch unavailable")
        error = RuntimeError("checkpoint failed")
        trainer, _, _, _ = self._build({"refs": list(range(6))}, save_error=error)
        with self.assertRaises(RuntimeError) as raised:
            trainer.fit()
        self.assertIs(raised.exception, error)

    def test_streaming_durable_ack_routes_through_controller(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch unavailable")
        _, cap, dfc_calls, _ = self._build({"queue": object()}, durable_ack=True)
        ack = cap["ctrl_kw"]["ack_fn"]
        ack(["s1"], 7)
        self.assertIn(("ack", "trainer-0", ["s1"], 7, True), dfc_calls)


if __name__ == "__main__":
    unittest.main()
