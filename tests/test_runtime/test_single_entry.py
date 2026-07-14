# coding=utf-8
"""CPU-only lifecycle gates for the single public SpecForge entry point."""

import os
import sys
import types
import unittest
from unittest import mock

from specforge.cli import (
    _bootstrap_single_process_env,
    _train,
    _validate_world_size,
    main,
)
from specforge.config import Config
from specforge.training.assembly import TrainingRun
from specforge.training.disaggregated import (
    _ONLINE_CONTROL_SUFFIXES,
    _claim_fresh_control_path,
    build_disaggregated_run,
)


class _FakeTrainer:
    def __init__(self, *, fit_step=3, error=None, events=None):
        self.fit_step = fit_step
        self.error = error
        self.events = events
        self.fit_calls = 0

    def fit(self):
        self.fit_calls += 1
        if self.events is not None:
            self.events.append("fit")
        if self.error is not None:
            raise self.error
        return self.fit_step


class TestTrainingRunLifecycle(unittest.TestCase):
    def test_training_run_delegates_to_the_one_trainer_entry(self):
        trainer = _FakeTrainer(fit_step=3)
        self.assertEqual(TrainingRun(trainer=trainer).run(), 3)
        self.assertEqual(trainer.fit_calls, 1)

    def test_trainer_cannot_be_bypassed_by_an_executor(self):
        trainer = _FakeTrainer(fit_step=4)
        with self.assertRaisesRegex(ValueError, "exactly one trainer or executor"):
            TrainingRun(trainer=trainer, execute=lambda: 4)

    def test_consumer_hooks_wrap_the_one_trainer_entry_in_order(self):
        events = []
        trainer = _FakeTrainer(fit_step=4, events=events)
        run = TrainingRun(
            trainer=trainer,
            on_success=lambda step: events.append(f"success:{step}"),
            on_failure=lambda exc: events.append(f"failure:{exc}"),
            on_finally=lambda: events.append("finally"),
        )
        self.assertEqual(run.run(), 4)
        self.assertEqual(events, ["fit", "success:4", "finally"])

    def test_consumer_failure_hook_precedes_cleanup(self):
        events = []
        error = RuntimeError("fit failed")
        trainer = _FakeTrainer(error=error, events=events)
        run = TrainingRun(
            trainer=trainer,
            on_success=lambda step: events.append(f"success:{step}"),
            on_failure=lambda exc: events.append(f"failure:{exc}"),
            on_finally=lambda: events.append("finally"),
        )
        with self.assertRaises(RuntimeError) as raised:
            run.run()
        self.assertIs(raised.exception, error)
        self.assertEqual(events, ["fit", "failure:fit failed", "finally"])

    def test_producer_result_does_not_require_a_trainer(self):
        self.assertEqual(TrainingRun(execute=lambda: 7).run(), 7)

    def test_missing_lifecycle_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "exactly one trainer or executor"):
            TrainingRun()

    def test_producer_executor_cannot_take_trainer_hooks(self):
        with self.assertRaisesRegex(ValueError, "hooks belong to trainer-bearing"):
            TrainingRun(execute=lambda: 1, on_finally=lambda: None)

    def test_disaggregated_producer_requires_fresh_attempt_path(self):
        import tempfile

        path = os.path.join(tempfile.mkdtemp(prefix="attempt_"), "refs.jsonl")
        _claim_fresh_control_path(path, (".closed", ".failed"))
        with self.assertRaisesRegex(ValueError, "new attempt-specific path"):
            _claim_fresh_control_path(path, (".closed", ".failed"))

    def test_online_fresh_attempt_rejects_stale_consumer_quantum(self):
        import tempfile

        path = os.path.join(tempfile.mkdtemp(prefix="attempt_quantum_"), "refs.jsonl")
        with open(path + ".consumer_quantum", "w", encoding="utf-8") as stream:
            stream.write("8")
        with self.assertRaisesRegex(ValueError, "consumer_quantum"):
            _claim_fresh_control_path(path, _ONLINE_CONTROL_SUFFIXES)

    def test_disaggregated_assembly_failure_notifies_the_peer(self):
        import tempfile

        for role, suffix in (
            ("producer", ".failed"),
            ("consumer", ".consumer_failed"),
        ):
            with self.subTest(role=role):
                root = tempfile.mkdtemp(prefix=f"assembly_{role}_")
                channel = os.path.join(root, "refs.jsonl")
                cfg = Config.model_validate(
                    {
                        "model": {
                            "target_model_path": "t",
                            "draft_model_config": "d",
                        },
                        "data": {"prompts_path": "/prompts.jsonl"},
                        "training": {
                            "strategy": "dflash",
                            "deployment_mode": "disaggregated",
                            "role": role,
                            "max_steps": 1,
                        },
                    }
                )

                def fail_during_assembly(*_args, **_kwargs):
                    if role == "producer":
                        _claim_fresh_control_path(channel, (".failed",))
                    raise RuntimeError("assembly exploded")

                with (
                    mock.patch.dict(
                        os.environ, {"DISAGG_REF_CHANNEL": channel}, clear=False
                    ),
                    mock.patch(
                        "specforge.training.disaggregated._build_online",
                        side_effect=fail_during_assembly,
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, "assembly exploded"):
                        build_disaggregated_run(
                            cfg,
                            build_model_bundle=mock.Mock(),
                            prepare_prompts=mock.Mock(),
                            optimizer_factory=mock.Mock(),
                            logger=mock.Mock(),
                        )

                with open(channel + suffix, encoding="utf-8") as stream:
                    self.assertIn("RuntimeError: assembly exploded", stream.read())

    def test_failed_claim_does_not_poison_an_existing_attempt(self):
        import tempfile

        root = tempfile.mkdtemp(prefix="assembly_foreign_")
        channel = os.path.join(root, "refs.jsonl")
        with open(channel + ".producer_claim", "w", encoding="utf-8") as stream:
            stream.write("pid=999999\n")
        cfg = Config.model_validate(
            {
                "model": {"target_model_path": "t", "draft_model_config": "d"},
                "data": {"prompts_path": "/prompts.jsonl"},
                "training": {
                    "strategy": "dflash",
                    "deployment_mode": "disaggregated",
                    "role": "producer",
                    "max_steps": 1,
                },
            }
        )
        with (
            mock.patch.dict(os.environ, {"DISAGG_REF_CHANNEL": channel}, clear=False),
            mock.patch(
                "specforge.training.disaggregated._build_online",
                side_effect=RuntimeError("claim rejected"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "claim rejected"):
                build_disaggregated_run(
                    cfg,
                    build_model_bundle=mock.Mock(),
                    prepare_prompts=mock.Mock(),
                    optimizer_factory=mock.Mock(),
                    logger=mock.Mock(),
                )
        self.assertFalse(os.path.exists(channel + ".failed"))


class TestCliLifecycle(unittest.TestCase):
    def test_capture_only_producer_does_not_import_distributed_cuda_runtime(self):
        cfg = Config.model_validate(
            {
                "model": {
                    "target_model_path": "target",
                    "draft_model_config": "draft",
                    "target_backend": "sglang",
                },
                "data": {"train_data_path": "train.jsonl"},
                "training": {
                    "strategy": "dflash",
                    "deployment_mode": "disaggregated",
                    "role": "producer",
                    "server_urls": ["http://capture"],
                    "total_steps": 1,
                },
            }
        )
        run = mock.Mock()
        run.run.return_value = 3
        assembly = types.ModuleType("specforge.training.assembly")
        assembly.build_training_run = mock.Mock(return_value=run)

        with mock.patch.dict(
            sys.modules,
            {
                "specforge.distributed": None,
                "specforge.training.assembly": assembly,
            },
        ):
            self.assertEqual(_train(cfg), 3)

        assembly.build_training_run.assert_called_once_with(cfg)
        run.run.assert_called_once_with()

    def test_world_size_matches_the_configured_parallel_topology(self):
        local = Config.model_validate(
            {
                "model": {"target_model_path": "t", "draft_model_config": "d"},
                "data": {"prompts_path": "/prompts.jsonl"},
                "training": {"tp_size": 2},
            }
        )
        _validate_world_size(local, 2)
        with self.assertRaisesRegex(ValueError, "divisible"):
            _validate_world_size(local, 3)

        consumer = Config.model_validate(
            {
                "model": {"target_model_path": "t", "draft_model_config": "d"},
                "data": {"prompts_path": "/prompts.jsonl"},
                "training": {
                    "strategy": "dflash",
                    "deployment_mode": "disaggregated",
                    "role": "consumer",
                    "total_steps": 10,
                },
            }
        )
        _validate_world_size(consumer, 8)

    def test_direct_invocation_bootstraps_one_process_rendezvous(self):
        rendezvous = mock.MagicMock()
        rendezvous.__enter__.return_value.getsockname.return_value = (
            "127.0.0.1",
            32123,
        )
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("specforge.cli.socket.socket", return_value=rendezvous),
        ):
            _bootstrap_single_process_env()
            self.assertEqual(os.environ["RANK"], "0")
            self.assertEqual(os.environ["WORLD_SIZE"], "1")
            self.assertEqual(os.environ["LOCAL_RANK"], "0")
            self.assertEqual(os.environ["MASTER_ADDR"], "127.0.0.1")
            self.assertEqual(os.environ["MASTER_PORT"], "32123")

    def test_partial_distributed_environment_fails_loudly(self):
        with mock.patch.dict(os.environ, {"RANK": "0"}, clear=True):
            with self.assertRaisesRegex(ValueError, "environment is incomplete"):
                _bootstrap_single_process_env()

    def test_hf_export_dispatches_through_shared_cli(self):
        calls = []
        module = types.ModuleType("specforge.export.to_hf")
        module.export_to_hf = lambda *args, **kwargs: calls.append((args, kwargs))
        with mock.patch.dict(sys.modules, {"specforge.export.to_hf": module}):
            self.assertEqual(
                main(
                    [
                        "export",
                        "--to",
                        "hf",
                        "--checkpoint",
                        "checkpoint",
                        "--draft-config",
                        "draft.json",
                        "--output-dir",
                        "exported",
                        "--embedding-source",
                        "target",
                    ]
                ),
                0,
            )
        self.assertEqual(calls[0][0], ("checkpoint", "draft.json", "exported"))
        self.assertEqual(calls[0][1]["embedding_source"], "target")


if __name__ == "__main__":
    unittest.main(verbosity=2)
