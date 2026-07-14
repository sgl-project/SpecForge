# coding=utf-8
"""Dependency-light contracts for the YAML-driven unified launcher."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from pydantic import ValidationError

from specforge.cli import main
from specforge.config import Config, apply_overrides
from specforge.launch_plan import (
    CommandSpec,
    LaunchPlan,
    build_launch_plan,
    run_commands,
)


def _config(*, mode="local_colocated", nproc=1, nnodes=1):
    raw = {
        "model": {"target_model_path": "target", "draft_model_config": "draft"},
        "data": {"prompts_path": "prompts.jsonl"},
        "training": {"strategy": "dflash", "max_steps": 1},
        "deployment": {
            "mode": mode,
            "trainer": {
                "nnodes": nnodes,
                "nproc_per_node": nproc,
                **({"master_addr": "trainer-0"} if nnodes > 1 else {}),
            },
        },
    }
    if mode == "disaggregated":
        raw["deployment"]["disaggregated"] = {
            "control_dir": "/shared/attempt-1",
            "backend": "mooncake",
            "server_urls": ["http://capture:30000"],
        }
    return Config.model_validate(raw)


def _offline_disaggregated_config(*, backend="mooncake", producer_segment_size=None):
    raw = _config(mode="disaggregated").model_dump()
    raw["data"] = {"hidden_states_path": "features"}
    raw["deployment"]["disaggregated"]["backend"] = backend
    raw["deployment"]["disaggregated"]["server_urls"] = []
    raw["training"]["server_urls"] = []
    if backend == "shared_dir":
        raw["deployment"]["disaggregated"]["store_root"] = "/shared/features"
    if producer_segment_size is not None:
        raw["deployment"]["disaggregated"][
            "producer_segment_size"
        ] = producer_segment_size
    return Config.model_validate(raw)


MOONCAKE_ENV = {
    "MOONCAKE_METADATA_SERVER": "http://metadata:8080/metadata",
    "MOONCAKE_MASTER_SERVER_ADDR": "master:50051",
}


class _FakeProcess:
    _next_pid = 41000

    def __init__(self, poll_results):
        self.poll_results = list(poll_results)
        self.status = None
        self.terminated = False
        self.killed = False
        self.pid = self._next_pid
        type(self)._next_pid += 1

    def poll(self):
        if self.status is not None:
            return self.status
        if not self.poll_results:
            return None
        result = self.poll_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        if result is not None:
            self.status = result
        return result

    def terminate(self):
        self.terminated = True
        self.status = -15

    def kill(self):
        self.killed = True
        self.status = -9

    def wait(self, timeout=None):
        return self.status


class LaunchPlanTest(unittest.TestCase):
    def test_local_multi_rank_self_launches_torchrun(self):
        plan = build_launch_plan(
            _config(nproc=4),
            config_path="run.yaml",
            worker_prefix=("specforge",),
            torchrun_prefix=("torchrun",),
            env={},
        )
        self.assertEqual(plan.kind, "command")
        self.assertEqual(
            plan.commands[0].argv,
            (
                "torchrun",
                "--standalone",
                "--nproc_per_node",
                "4",
                "specforge",
                "train",
                "--config",
                "run.yaml",
                "--role",
                "all",
            ),
        )

    def test_production_command_uses_python_module_entrypoints(self):
        plan = build_launch_plan(
            _config(nproc=2),
            config_path="run.yaml",
            overrides=("training.batch_size=2",),
            env={},
        )
        self.assertEqual(
            plan.commands[0].argv,
            (
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node",
                "2",
                "--module",
                "specforge.cli",
                "train",
                "--config",
                "run.yaml",
                "--role",
                "all",
                "training.batch_size=2",
            ),
        )
        self.assertEqual(plan.commands[0].argv.count("training.batch_size=2"), 1)

    def test_existing_torchrun_becomes_a_worker_without_nested_launch(self):
        distributed = {
            "RANK": "1",
            "WORLD_SIZE": "4",
            "LOCAL_RANK": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
        }
        plan = build_launch_plan(
            _config(nproc=4), config_path="run.yaml", env=distributed
        )
        self.assertEqual(plan.kind, "worker")
        self.assertEqual(plan.role, "all")
        self.assertEqual(plan.commands, ())

    def test_existing_torchrun_must_match_yaml_world_size(self):
        distributed = {
            "RANK": "0",
            "WORLD_SIZE": "2",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
        }
        with self.assertRaisesRegex(ValueError, "does not match YAML"):
            build_launch_plan(_config(nproc=4), config_path="run.yaml", env=distributed)

    def test_disaggregated_auto_supervises_both_roles_on_one_node(self):
        plan = build_launch_plan(
            _config(mode="disaggregated", nproc=2),
            config_path="run.yaml",
            worker_prefix=("specforge",),
            torchrun_prefix=("torchrun",),
            env=MOONCAKE_ENV,
        )
        self.assertEqual((plan.kind, plan.role), ("supervisor", "both"))
        self.assertEqual(
            [item.label for item in plan.commands], ["producer", "consumer"]
        )
        self.assertEqual(plan.commands[0].argv[-2:], ("--role", "producer"))
        self.assertEqual(
            plan.commands[1].argv[:4],
            ("torchrun", "--standalone", "--nproc_per_node", "2"),
        )
        self.assertEqual(
            plan.commands[0].env["DISAGG_REF_CHANNEL"],
            "/shared/attempt-1/refs.jsonl",
        )
        self.assertEqual(
            plan.commands[1].env["DISAGG_DB"],
            "/shared/attempt-1/consumer.sqlite",
        )
        self.assertEqual(plan.commands[0].env["DISAGG_CLIENT_SEGMENT_SIZE"], "0")
        self.assertEqual(
            plan.commands[1].env["DISAGG_CLIENT_BUFFER_SIZE"], str(256 << 20)
        )

    def test_disaggregated_roles_are_independently_selectable(self):
        producer = build_launch_plan(
            _config(mode="disaggregated", nproc=4),
            config_path="run.yaml",
            requested_role="producer",
            env=MOONCAKE_ENV,
        )
        self.assertEqual((producer.kind, producer.role), ("worker", "producer"))

        consumer = build_launch_plan(
            _config(mode="disaggregated", nproc=4),
            config_path="run.yaml",
            requested_role="consumer",
            worker_prefix=("specforge",),
            torchrun_prefix=("torchrun",),
            env=MOONCAKE_ENV,
        )
        self.assertEqual((consumer.kind, consumer.role), ("command", "consumer"))
        self.assertIn("--nproc_per_node", consumer.commands[0].argv)

    def test_online_disaggregation_forces_server_owned_client_segments(self):
        raw = _config(mode="disaggregated").model_dump()
        raw["deployment"]["disaggregated"]["producer_segment_size"] = 4096
        config = Config.model_validate(raw)
        plan = build_launch_plan(
            config,
            config_path="run.yaml",
            env={
                **MOONCAKE_ENV,
                "DISAGG_CLIENT_SEGMENT_SIZE": "8192",
            },
        )
        self.assertEqual(plan.commands[0].env["DISAGG_CLIENT_SEGMENT_SIZE"], "0")
        self.assertEqual(plan.commands[1].env["DISAGG_CLIENT_SEGMENT_SIZE"], "0")

    def test_offline_mooncake_uses_role_specific_segment_ownership(self):
        config = _offline_disaggregated_config(producer_segment_size=4096)
        plan = build_launch_plan(
            config,
            config_path="run.yaml",
            env=MOONCAKE_ENV,
        )
        self.assertEqual(plan.commands[0].env["DISAGG_CLIENT_SEGMENT_SIZE"], "4096")
        self.assertEqual(plan.commands[1].env["DISAGG_CLIENT_SEGMENT_SIZE"], "0")

    def test_offline_mooncake_producer_requires_a_positive_segment(self):
        config = _offline_disaggregated_config()
        with self.assertRaisesRegex(ValueError, "positive.*producer_segment_size"):
            build_launch_plan(
                config,
                config_path="run.yaml",
                requested_role="producer",
                env=MOONCAKE_ENV,
            )

        fallback = build_launch_plan(
            config,
            config_path="run.yaml",
            requested_role="producer",
            env={**MOONCAKE_ENV, "MOONCAKE_GLOBAL_SEGMENT_SIZE": "8192"},
        )
        self.assertEqual(fallback.worker_env["DISAGG_CLIENT_SEGMENT_SIZE"], "8192")

    def test_offline_shared_directory_does_not_set_mooncake_segments(self):
        plan = build_launch_plan(
            _offline_disaggregated_config(backend="shared_dir"),
            config_path="run.yaml",
            env={},
        )
        for command in plan.commands:
            self.assertNotIn("DISAGG_CLIENT_SEGMENT_SIZE", command.env)
            self.assertNotIn("DISAGG_CLIENT_BUFFER_SIZE", command.env)

    def test_capture_server_urls_are_required_only_for_producers(self):
        raw = _config(mode="disaggregated").model_dump()
        raw["deployment"]["disaggregated"]["server_urls"] = []
        raw["training"]["server_urls"] = []
        cfg = Config.model_validate(raw)
        with self.assertRaisesRegex(ValueError, "requires server URLs"):
            build_launch_plan(
                cfg,
                config_path="run.yaml",
                requested_role="producer",
                env=MOONCAKE_ENV,
            )
        consumer = build_launch_plan(
            cfg,
            config_path="run.yaml",
            requested_role="consumer",
            env=MOONCAKE_ENV,
        )
        self.assertEqual(consumer.role, "consumer")

    def test_multi_node_auto_fails_instead_of_remote_spawning(self):
        cfg = _config(mode="disaggregated", nproc=2, nnodes=2)
        with self.assertRaisesRegex(ValueError, "one trainer node only"):
            build_launch_plan(
                cfg,
                config_path="run.yaml",
                node_rank=0,
                env=MOONCAKE_ENV,
            )

    def test_multi_node_producer_ignores_scheduler_node_rank(self):
        plan = build_launch_plan(
            _config(mode="disaggregated", nproc=2, nnodes=2),
            config_path="run.yaml",
            requested_role="producer",
            env={**MOONCAKE_ENV, "NODE_RANK": "99"},
        )
        self.assertEqual((plan.kind, plan.role), ("worker", "producer"))

    def test_multi_node_consumer_requires_rank_before_database_checks(self):
        with self.assertRaisesRegex(ValueError, "requires --node-rank"):
            build_launch_plan(
                _config(mode="disaggregated", nproc=2, nnodes=2),
                config_path="run.yaml",
                requested_role="consumer",
                env=MOONCAKE_ENV,
            )

    def test_partial_distributed_environment_fails_before_launch(self):
        with self.assertRaisesRegex(ValueError, "environment is incomplete"):
            build_launch_plan(_config(), config_path="run.yaml", env={"RANK": "0"})

    def test_multi_rank_torchrun_rejects_duplicate_producers(self):
        distributed = {
            "RANK": "0",
            "WORLD_SIZE": "2",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
        }
        with self.assertRaisesRegex(ValueError, "direct single process"):
            build_launch_plan(
                _config(mode="disaggregated", nproc=2),
                config_path="run.yaml",
                requested_role="producer",
                env={**distributed, **MOONCAKE_ENV},
            )

    def test_online_database_freshness_and_resume_are_checked(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = _config(mode="disaggregated")
            raw = cfg.model_dump()
            raw["deployment"]["disaggregated"]["control_dir"] = root
            cfg = Config.model_validate(raw)
            database = os.path.join(root, "consumer.sqlite")
            open(database, "a", encoding="utf-8").close()
            with self.assertRaisesRegex(ValueError, "fresh attempt path"):
                build_launch_plan(
                    cfg,
                    config_path="run.yaml",
                    requested_role="consumer",
                    env=MOONCAKE_ENV,
                )

            raw = cfg.model_dump()
            raw["training"]["resume_from"] = os.path.join(root, "latest")
            resumed = Config.model_validate(raw)
            plan = build_launch_plan(
                resumed,
                config_path="run.yaml",
                env=MOONCAKE_ENV,
            )
            self.assertEqual(plan.role, "consumer")

            os.remove(database)
            with self.assertRaisesRegex(ValueError, "retained metadata database"):
                build_launch_plan(
                    resumed,
                    config_path="run.yaml",
                    requested_role="consumer",
                    env=MOONCAKE_ENV,
                )

    def test_new_deployment_surface_rejects_legacy_conflicts(self):
        raw = _config().model_dump()
        raw["training"]["deployment_mode"] = "disaggregated"
        with self.assertRaisesRegex(ValidationError, "conflicts"):
            Config.model_validate(raw)

    def test_online_disaggregated_schema_rejects_shared_directory_store(self):
        raw = _config(mode="disaggregated").model_dump()
        raw["deployment"]["disaggregated"]["backend"] = "shared_dir"
        raw["deployment"]["disaggregated"]["store_root"] = "/shared/features"
        with self.assertRaisesRegex(ValidationError, "backend=mooncake"):
            Config.model_validate(raw)

    def test_deployment_server_url_override_updates_assembly_view(self):
        cfg = _config(mode="disaggregated")
        updated = apply_overrides(
            cfg,
            ["deployment.disaggregated.server_urls=[http://new-capture:31000]"],
        )
        self.assertEqual(
            updated.deployment.disaggregated.server_urls,
            ["http://new-capture:31000"],
        )
        self.assertEqual(
            updated.training.server_urls,
            ["http://new-capture:31000"],
        )

        legacy_updated = apply_overrides(
            updated,
            ["training.server_urls=[http://legacy-name:32000]"],
        )
        self.assertEqual(
            legacy_updated.deployment.disaggregated.server_urls,
            ["http://legacy-name:32000"],
        )

    def test_deployment_aliases_round_trip_and_survive_unrelated_overrides(self):
        cfg = _config(mode="disaggregated")
        self.assertEqual(Config.model_validate(cfg.model_dump()), cfg)
        updated = apply_overrides(cfg, ["training.batch_size=2"])
        self.assertEqual(updated.training.server_urls, cfg.training.server_urls)
        self.assertEqual(
            updated.deployment.disaggregated.server_urls,
            cfg.deployment.disaggregated.server_urls,
        )

    def test_conflicting_server_url_aliases_fail_loudly(self):
        raw = _config(mode="disaggregated").model_dump()
        raw["training"]["server_urls"] = ["http://legacy:30000"]
        with self.assertRaisesRegex(ValidationError, "server_urls conflicts"):
            Config.model_validate(raw)

    def test_cli_plan_is_json_and_does_not_execute(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "run.json")
            raw = _config(nproc=2).model_dump(mode="json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(raw, stream)
            with (
                mock.patch("specforge.launch_plan.run_commands") as run,
                mock.patch("builtins.print") as output,
            ):
                self.assertEqual(main(["train", "-c", path, "--plan"]), 0)
            run.assert_not_called()
            rendered = output.call_args.args[0]
            payload = json.loads(rendered)
            self.assertEqual(payload["kind"], "command")
            self.assertNotIn("--plan", payload["commands"][0]["argv"])

    def test_cli_producer_projection_ignores_consumer_only_settings(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "run.json")
            raw = _config(mode="disaggregated").model_dump(mode="json")
            raw["training"]["metadata_db_path"] = os.path.join(root, "consumer.sqlite")
            raw["profiling"]["enabled"] = True
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(raw, stream)

            with (
                mock.patch.dict(os.environ, MOONCAKE_ENV, clear=True),
                mock.patch("specforge.cli._train", return_value=0) as train,
            ):
                self.assertEqual(
                    main(["train", "-c", path, "--role", "producer"]),
                    0,
                )

            projected = train.call_args.args[0]
            self.assertEqual(projected.training.role, "producer")
            self.assertIsNone(projected.training.metadata_db_path)
            self.assertFalse(projected.profiling.enabled)

    def test_plan_redacts_secret_overrides(self):
        plan = build_launch_plan(
            _config(nproc=2),
            config_path="run.yaml",
            overrides=("tracking.wandb_key=top-secret",),
            worker_prefix=("specforge",),
            torchrun_prefix=("torchrun",),
            env={},
        )
        rendered = plan.render()
        self.assertNotIn("top-secret", rendered)
        self.assertIn("tracking.wandb_key=<redacted>", rendered)

        url_plan = build_launch_plan(
            _config(nproc=2),
            config_path="run.yaml",
            overrides=("tracking.mlflow_tracking_uri=https://user:pass@mlflow.test/a",),
            worker_prefix=("specforge",),
            torchrun_prefix=("torchrun",),
            env={},
        )
        rendered = url_plan.render()
        self.assertNotIn("user:pass", rendered)
        self.assertIn("<redacted>@mlflow.test", rendered)

        malformed_port = build_launch_plan(
            _config(nproc=2),
            config_path="run.yaml",
            overrides=(
                "tracking.mlflow_tracking_uri=https://user:pass@mlflow.test:bad/a",
            ),
            worker_prefix=("specforge",),
            torchrun_prefix=("torchrun",),
            env={},
        ).render()
        self.assertNotIn("user:pass", malformed_port)

    def test_explicit_both_rejects_consumer_only_resume(self):
        cfg = _config(mode="disaggregated")
        raw = cfg.model_dump()
        raw["training"]["resume_from"] = "/checkpoint/latest"
        cfg = Config.model_validate(raw)
        with self.assertRaisesRegex(ValueError, "--role consumer"):
            build_launch_plan(
                cfg,
                config_path="run.yaml",
                requested_role="both",
                env=MOONCAKE_ENV,
            )

    def test_supervisor_terminates_a_sibling_after_child_failure(self):
        producer = _FakeProcess([None])
        consumer = _FakeProcess([7])
        processes = iter((producer, consumer))
        plan = LaunchPlan(
            "supervisor",
            "both",
            commands=(
                CommandSpec("producer", ("producer",)),
                CommandSpec("consumer", ("consumer",)),
            ),
        )
        with mock.patch(
            "specforge.launch_plan.os.killpg", side_effect=ProcessLookupError
        ):
            status = run_commands(plan, popen=lambda *_args, **_kwargs: next(processes))
        self.assertEqual(status, 7)
        self.assertTrue(producer.terminated)

    def test_clean_producer_exit_does_not_cancel_a_running_consumer(self):
        producer = _FakeProcess([0])
        consumer = _FakeProcess([None, 0])
        processes = iter((producer, consumer))
        plan = LaunchPlan(
            "supervisor",
            "both",
            commands=(
                CommandSpec("producer", ("producer",)),
                CommandSpec("consumer", ("consumer",)),
            ),
        )
        with mock.patch("specforge.launch_plan.time.sleep"):
            status = run_commands(plan, popen=lambda *_args, **_kwargs: next(processes))
        self.assertEqual(status, 0)
        self.assertFalse(consumer.terminated)

    def test_keyboard_interrupt_cleans_up_every_child(self):
        producer = _FakeProcess([KeyboardInterrupt()])
        consumer = _FakeProcess([None])
        processes = iter((producer, consumer))
        plan = LaunchPlan(
            "supervisor",
            "both",
            commands=(
                CommandSpec("producer", ("producer",)),
                CommandSpec("consumer", ("consumer",)),
            ),
        )
        with (
            mock.patch(
                "specforge.launch_plan.os.killpg", side_effect=ProcessLookupError
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            run_commands(plan, popen=lambda *_args, **_kwargs: next(processes))
        self.assertTrue(producer.terminated)
        self.assertTrue(consumer.terminated)

    @unittest.skipUnless(
        hasattr(os, "killpg") and hasattr(signal, "SIGTERM"),
        "requires POSIX process groups",
    )
    def test_sigterm_cleans_the_real_child_process_group(self):
        child_code = """
import os
import subprocess
import sys
import time

grandchild = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(60)"]
)
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    stream.write(f"{os.getpid()} {grandchild.pid}")
time.sleep(60)
"""
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        with tempfile.TemporaryDirectory() as root:
            marker = os.path.join(root, "children.txt")
            runner_code = f"""
import sys
from specforge.launch_plan import CommandSpec, LaunchPlan, run_commands

plan = LaunchPlan(
    "command",
    "all",
    commands=(
        CommandSpec(
            "child",
            (sys.executable, "-c", {child_code!r}, {marker!r}),
        ),
    ),
)
raise SystemExit(run_commands(plan))
"""
            runner = subprocess.Popen(
                [sys.executable, "-c", runner_code],
                cwd=repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            child_pid = None
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    if os.path.exists(marker):
                        with open(marker, encoding="utf-8") as stream:
                            raw = stream.read().strip()
                        if len(raw.split()) == 2:
                            child_pid = int(raw.split()[0])
                            break
                    if runner.poll() is not None:
                        break
                    time.sleep(0.02)
                self.assertIsNotNone(child_pid, "managed child never became ready")
                self.assertEqual(os.getpgid(child_pid), child_pid)

                os.kill(runner.pid, signal.SIGTERM)
                stdout, stderr = runner.communicate(timeout=10)
                self.assertEqual(
                    runner.returncode,
                    128 + signal.SIGTERM,
                    f"stdout={stdout!r} stderr={stderr!r}",
                )

                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        os.killpg(child_pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.02)
                else:
                    self.fail(f"managed process group {child_pid} survived SIGTERM")
            finally:
                if runner.poll() is None:
                    runner.kill()
                    runner.wait(timeout=5)
                if child_pid is not None:
                    try:
                        os.killpg(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_signal_exit_status_is_normalized(self):
        process = _FakeProcess([-15])
        plan = LaunchPlan(
            "command",
            "consumer",
            commands=(CommandSpec("consumer", ("consumer",)),),
        )
        self.assertEqual(
            run_commands(plan, popen=lambda *_args, **_kwargs: process),
            143,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
