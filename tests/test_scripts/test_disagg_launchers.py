"""Dependency-light contract tests for the disaggregated shell launchers."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ONLINE = ROOT / "examples" / "disagg" / "run_online.sh"
OFFLINE = ROOT / "examples" / "disagg" / "run_offline.sh"

_DISAGG_ENV_PREFIXES = ("DISAGG_", "MOONCAKE_")


class TestDisaggregatedLaunchers(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="disagg_launchers_")
        self.root = Path(self._tmp.name)
        self.bin_dir = self.root / "bin"
        self.capture_dir = self.root / "capture"
        self.bin_dir.mkdir()
        self.capture_dir.mkdir()
        self.config = self.root / "run config.yaml"
        self.config.write_text("run_id: launcher-contract\n", encoding="utf-8")
        for program in ("specforge", "torchrun"):
            executable = self.bin_dir / program
            executable.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'printf "%s\\n" "$@" > "${CAPTURE_DIR}/'
                f'{program}.args"\n',
                encoding="utf-8",
            )
            executable.chmod(0o755)

    def tearDown(self):
        self._tmp.cleanup()

    def _base_env(self):
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(_DISAGG_ENV_PREFIXES)
            and key not in {"CONFIG", "NPROC_PER_NODE", "CAPTURE_DIR"}
        }
        env.update(
            {
                "PATH": f"{self.bin_dir}{os.pathsep}{env.get('PATH', '')}",
                "CAPTURE_DIR": str(self.capture_dir),
                "CONFIG": str(self.config),
            }
        )
        return env

    def _online_env(self):
        env = self._base_env()
        env.update(
            {
                "DISAGG_REF_CHANNEL": str(self.root / "attempt.refs.jsonl"),
                "MOONCAKE_METADATA_SERVER": "http://metadata:8080/metadata",
                "MOONCAKE_MASTER_SERVER_ADDR": "master:50051",
                "MOONCAKE_LOCAL_HOSTNAME": "worker.example",
            }
        )
        return env

    def _offline_env(self):
        env = self._base_env()
        env.update(
            {
                "DISAGG_MANIFEST": str(self.root / "attempt.manifest.json"),
                "DISAGG_STORE_ROOT": str(self.root / "features"),
            }
        )
        return env

    def _run(self, script: Path, role: str | None, env, *overrides: str):
        command = [str(script)]
        if role is not None:
            command.append(role)
        command.extend(overrides)
        return subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def _captured(self, program: str):
        path = self.capture_dir / f"{program}.args"
        return path.read_text(encoding="utf-8").splitlines()

    def test_launchers_are_executable(self):
        self.assertTrue(os.access(ONLINE, os.X_OK))
        self.assertTrue(os.access(OFFLINE, os.X_OK))
        for launcher in (ONLINE, OFFLINE):
            result = subprocess.run(
                ["bash", "-n", str(launcher)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((ONLINE.parent / "run_role.sh").exists())

    def test_launchers_reject_missing_config_and_invalid_role(self):
        for launcher in (ONLINE, OFFLINE):
            with self.subTest(launcher=launcher.name, case="missing role"):
                result = self._run(launcher, None, self._base_env())
                self.assertEqual(result.returncode, 2)
            with self.subTest(launcher=launcher.name, case="invalid role"):
                result = self._run(launcher, "all", self._base_env())
                self.assertEqual(result.returncode, 2)
                self.assertIn("producer or consumer", result.stderr)
            with self.subTest(launcher=launcher.name, case="missing config"):
                env = self._base_env()
                del env["CONFIG"]
                result = self._run(launcher, "producer", env)
                self.assertEqual(result.returncode, 2)
                self.assertIn("CONFIG", result.stderr)
        self.assertEqual(list(self.capture_dir.iterdir()), [])

    def test_online_producer_dispatches_to_the_single_training_cli(self):
        env = self._online_env()
        result = self._run(
            ONLINE,
            "producer",
            env,
            "training.batch_size=2",
            "training.role=consumer",
            "run_id=attempt with spaces",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self._captured("specforge"),
            [
                "train",
                "--config",
                str(self.config),
                "training.batch_size=2",
                "training.role=consumer",
                "run_id=attempt with spaces",
                "training.role=producer",
            ],
        )

    def test_online_consumer_dispatches_dp_through_torchrun(self):
        env = self._online_env()
        env["DISAGG_DB"] = str(self.root / "consumer.sqlite")
        env["NPROC_PER_NODE"] = "4"
        result = self._run(ONLINE, "consumer", env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self._captured("torchrun"),
            [
                "--standalone",
                "--nproc_per_node",
                "4",
                str(self.bin_dir / "specforge"),
                "train",
                "--config",
                str(self.config),
                f"training.metadata_db_path={env['DISAGG_DB']}",
                "training.role=consumer",
            ],
        )

    def test_online_rejects_missing_transport_and_stale_database(self):
        missing = self._online_env()
        del missing["DISAGG_REF_CHANNEL"]
        result = self._run(ONLINE, "producer", missing)
        self.assertEqual(result.returncode, 2)
        self.assertIn("DISAGG_REF_CHANNEL", result.stderr)

        stale = self._online_env()
        database = self.root / "stale.sqlite"
        database.touch()
        stale["DISAGG_DB"] = str(database)
        result = self._run(ONLINE, "consumer", stale)
        self.assertEqual(result.returncode, 2)
        self.assertIn("fresh attempt path", result.stderr)

    def test_online_consumer_requires_an_explicit_database_for_dp(self):
        env = self._online_env()
        env["NPROC_PER_NODE"] = "4"
        result = self._run(ONLINE, "consumer", env)
        self.assertEqual(result.returncode, 2)
        self.assertIn("DISAGG_DB", result.stderr)

    def test_online_consumer_inbox_requires_an_explicit_database(self):
        env = self._online_env()
        env["DISAGG_INBOX_DIR"] = str(self.root / "inbox")
        result = self._run(ONLINE, "consumer", env)
        self.assertEqual(result.returncode, 2)
        self.assertIn("DISAGG_DB", result.stderr)

    def test_online_single_rank_consumer_requires_database(self):
        env = self._online_env()
        result = self._run(ONLINE, "consumer", env)
        self.assertEqual(result.returncode, 2)
        self.assertIn("DISAGG_DB", result.stderr)

    def test_online_database_env_overrides_the_config_path(self):
        env = self._online_env()
        env["DISAGG_DB"] = str(self.root / "fresh.sqlite")
        result = self._run(
            ONLINE,
            "consumer",
            env,
            "training.metadata_db_path=/shared/stale.sqlite",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self._captured("torchrun")[-3:],
            [
                "training.metadata_db_path=/shared/stale.sqlite",
                f"training.metadata_db_path={env['DISAGG_DB']}",
                "training.role=consumer",
            ],
        )

    def test_offline_roles_dispatch_directly_and_stay_single_rank(self):
        for role in ("producer", "consumer"):
            with self.subTest(role=role):
                result = self._run(OFFLINE, role, self._offline_env())
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    self._captured("specforge"),
                    ["train", "--config", str(self.config), f"training.role={role}"],
                )

        multi_rank = self._offline_env()
        multi_rank["NPROC_PER_NODE"] = "2"
        result = self._run(OFFLINE, "consumer", multi_rank)
        self.assertEqual(result.returncode, 2)
        self.assertIn("single-rank", result.stderr)

    def test_offline_mooncake_validates_its_transport_contract(self):
        env = self._offline_env()
        env["DISAGG_BACKEND"] = "mooncake"
        result = self._run(OFFLINE, "producer", env)
        self.assertEqual(result.returncode, 2)
        self.assertIn("MOONCAKE_METADATA_SERVER", result.stderr)

        env.update(
            {
                "MOONCAKE_METADATA_SERVER": "http://metadata:8080/metadata",
                "MOONCAKE_MASTER_SERVER_ADDR": "master:50051",
                "MOONCAKE_LOCAL_HOSTNAME": "producer.example",
            }
        )
        result = self._run(OFFLINE, "producer", env)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_offline_rejects_missing_or_unknown_backend_contract(self):
        missing_store = self._offline_env()
        del missing_store["DISAGG_STORE_ROOT"]
        result = self._run(OFFLINE, "producer", missing_store)
        self.assertEqual(result.returncode, 2)
        self.assertIn("DISAGG_STORE_ROOT", result.stderr)

        unknown = self._offline_env()
        unknown["DISAGG_BACKEND"] = "object_store"
        result = self._run(OFFLINE, "producer", unknown)
        self.assertEqual(result.returncode, 2)
        self.assertIn("shared_dir or mooncake", result.stderr)

    def test_docs_reference_only_the_mode_specific_launchers(self):
        markdown_paths = [
            ROOT / "README.md",
            *(ROOT / "docs").rglob("*.md"),
            *(ROOT / "examples").rglob("*.md"),
        ]
        markdown = "\n".join(
            path.read_text(encoding="utf-8") for path in markdown_paths
        )
        self.assertIn("run_online.sh", markdown)
        self.assertIn("run_offline.sh", markdown)
        self.assertNotIn("run_role.sh", markdown)


if __name__ == "__main__":
    unittest.main(verbosity=2)
