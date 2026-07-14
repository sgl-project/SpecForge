"""Dependency-light contracts for the thin disaggregated examples."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ONLINE = ROOT / "examples" / "disagg" / "run_online.sh"
OFFLINE = ROOT / "examples" / "disagg" / "run_offline.sh"


class DisaggregatedWrapperTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="disagg_wrapper_")
        self.root = Path(self._tmp.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.capture = self.root / "args.txt"
        self.config = self.root / "run config.yaml"
        self.config.write_text("run_id: wrapper-test\n", encoding="utf-8")
        executable = self.bin_dir / "specforge"
        executable.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'printf "%s\\n" "$@" > "$CAPTURE_PATH"\n',
            encoding="utf-8",
        )
        executable.chmod(0o755)

    def tearDown(self):
        self._tmp.cleanup()

    def _env(self, *, include_config=True):
        env = os.environ.copy()
        env["PATH"] = f"{self.bin_dir}{os.pathsep}{env.get('PATH', '')}"
        env["CAPTURE_PATH"] = str(self.capture)
        if include_config:
            env["CONFIG"] = str(self.config)
        else:
            env.pop("CONFIG", None)
        return env

    def _run(self, wrapper, *args, include_config=True):
        return subprocess.run(
            [str(wrapper), *args],
            cwd=ROOT,
            env=self._env(include_config=include_config),
            text=True,
            capture_output=True,
            check=False,
        )

    def test_wrappers_are_executable_and_syntax_valid(self):
        for wrapper in (ONLINE, OFFLINE):
            with self.subTest(wrapper=wrapper.name):
                self.assertTrue(os.access(wrapper, os.X_OK))
                result = subprocess.run(
                    ["bash", "-n", str(wrapper)],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_wrappers_only_delegate_to_the_unified_cli(self):
        for wrapper in (ONLINE, OFFLINE):
            with self.subTest(wrapper=wrapper.name):
                result = self._run(
                    wrapper,
                    "--role",
                    "consumer",
                    "--plan",
                    "training.batch_size=2",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    self.capture.read_text(encoding="utf-8").splitlines(),
                    [
                        "train",
                        "--config",
                        str(self.config),
                        "--role",
                        "consumer",
                        "--plan",
                        "training.batch_size=2",
                    ],
                )

    def test_wrappers_validate_only_the_config_boundary(self):
        for wrapper in (ONLINE, OFFLINE):
            with self.subTest(wrapper=wrapper.name):
                result = self._run(wrapper, include_config=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn("set CONFIG", result.stderr)

    def test_topology_and_transport_logic_are_not_duplicated_in_shell(self):
        forbidden = (
            "torchrun",
            "NPROC_PER_NODE",
            "NNODES",
            "NODE_RANK",
            "DISAGG_DB",
            "DISAGG_REF_CHANNEL",
            "DISAGG_MANIFEST",
            "MOONCAKE_",
        )
        for wrapper in (ONLINE, OFFLINE):
            source = wrapper.read_text(encoding="utf-8")
            for token in forbidden:
                with self.subTest(wrapper=wrapper.name, token=token):
                    self.assertNotIn(token, source)

    def test_help_describes_auto_and_explicit_roles(self):
        for wrapper in (ONLINE, OFFLINE):
            result = self._run(wrapper, "--help")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--role", result.stdout)
            self.assertIn("producer and consumer", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
