"""Exercise CI cache snapshots and FA installation without downloading packages."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/test.yaml"
ENSURE_FA = ROOT / ".github/workflows/scripts/ensure_flash_attention.sh"


class CIDependencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cache = self.root / "persistent cache"
        self.cache.mkdir()
        self.modules = self.root / "modules"
        self.modules.mkdir()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        (self.bin / "python").symlink_to(sys.executable)
        self.env = {
            **os.environ,
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "PYTHONPATH": str(self.modules),
            "SPECFORGE_CI_CACHE": str(self.cache),
            "FLASH_ATTN_VERSION": "2.8.3.post2",
            "TEST_ROOT": str(self.root),
        }
        self.job = yaml.safe_load(WORKFLOW.read_text())["jobs"]["unit-test"]
        self.steps = {step["name"]: step for step in self.job["steps"]}

    def run_shell(self, script):
        return subprocess.run(
            ["bash", "-ec", script],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
        )

    def run_step(self, name):
        result = self.run_shell(self.steps[name]["run"])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def setup_fa(self):
        (self.modules / "torch.py").write_text(
            "import os\n"
            "__version__ = os.environ.get('TEST_TORCH', '2.13.0+cu130')\n"
            "class version: cuda = os.environ.get('TEST_CUDA', '13.0')\n"
            "class _C: _GLIBCXX_USE_CXX11_ABI = True\n"
        )
        transformers = self.modules / "transformers"
        transformers.mkdir()
        (transformers / "__init__.py").touch()
        (transformers / "modeling_flash_attention_utils.py").write_text(
            "import os\nfrom pathlib import Path\n"
            "def lazy_import_flash_attention(backend):\n"
            "    assert backend == 'flash_attention_2'\n"
            "    if not (Path(os.environ['TEST_ROOT']) / 'ready').exists():\n"
            "        raise ImportError('undefined symbol: test_abi')\n"
        )
        uv = self.bin / "uv"
        uv.write_text(
            f"#!{sys.executable}\n"
            + textwrap.dedent(
                """\
                import json
                import os
                import shutil
                import sys
                from pathlib import Path

                root = Path(os.environ["TEST_ROOT"])
                modules = root / "modules"
                cache = Path(os.environ["UV_CACHE_DIR"])
                with (root / "calls").open("a") as log:
                    log.write(json.dumps({"args": sys.argv[1:], "cache": str(cache),
                              "force": os.environ.get("FLASH_ATTENTION_FORCE_BUILD")}) + "\\n")
                if sys.argv[2] == "uninstall":
                    for entry in modules.glob("flash_attn-*.dist-info"):
                        shutil.rmtree(entry)
                else:
                    if os.environ.get("TEST_INSTALL_FAIL"):
                        sys.exit(1)
                    cache.mkdir(parents=True, exist_ok=True)
                    version = os.environ["FLASH_ATTN_VERSION"]
                    metadata = modules / f"flash_attn-{version}.dist-info"
                    metadata.mkdir()
                    (metadata / "METADATA").write_text(
                        f"Name: flash-attn\\nVersion: {version}\\n")
                    if not os.environ.get("TEST_ABI_FAIL"):
                        (root / "ready").touch()
                """
            )
        )
        uv.chmod(0o755)

    def run_fa(self, success=True):
        result = subprocess.run(
            ["bash", str(ENSURE_FA)],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)

    def calls(self):
        return [
            json.loads(line) for line in (self.root / "calls").read_text().splitlines()
        ]

    def test_save_and_restore_remove_stale_versions(self):
        old = self.cache / "sf" / "old.dist-info"
        old.mkdir(parents=True)
        (self.root / "sf").mkdir()
        (self.root / "sf" / "current.dist-info").touch()
        (self.root / "sf" / "python").symlink_to(sys.executable)
        self.run_step("Save dependency cache")
        self.assertFalse(old.exists())
        (self.root / "sf" / "stale.dist-info").touch()
        # Exercise only the venv restore; the model cache is unrelated to this test.
        restore = self.steps["Restore cache"]["run"]
        restore = restore[restore.index('if [ -d "$SPECFORGE_CI_CACHE/sf"') :]
        result = self.run_shell(restore)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            {p.name for p in (self.root / "sf").iterdir()},
            {"current.dist-info", "python"},
        )
        self.assertTrue((self.root / "sf" / "python").is_symlink())

    def test_failed_copy_preserves_previous_snapshot(self):
        (self.cache / "sf").mkdir()
        (self.cache / "sf" / "valid").touch()
        result = self.run_shell(self.steps["Save dependency cache"]["run"])
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.cache / "sf" / "valid").exists())
        self.assertEqual(list(self.cache.glob("sf.*")), [])

    def test_fa_install_then_healthy_environment_skips_install(self):
        self.setup_fa()
        self.run_fa()
        calls = self.calls()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["args"], ["pip", "uninstall", "flash-attn"])
        install = calls[1]
        self.assertEqual(install["force"], "TRUE")
        for flag in ("--no-deps", "--no-build-isolation", "--no-binary"):
            self.assertIn(flag, install["args"])
        for flag in ("--no-cache", "--reinstall-package", "--refresh"):
            self.assertNotIn(flag, install["args"])
        self.run_fa()
        self.assertEqual(self.calls(), calls)

    def test_abi_failure_repairs_then_rechecks(self):
        self.setup_fa()
        self.run_fa()
        (self.root / "ready").unlink()
        self.run_fa()
        self.assertEqual(len(self.calls()), 4)
        self.assertEqual(self.calls()[1]["cache"], self.calls()[3]["cache"])

    def test_duplicate_metadata_requires_repair_even_if_import_works(self):
        self.setup_fa()
        self.run_fa()
        old = self.modules / "flash_attn-2.8.3.post1.dist-info"
        old.mkdir()
        (old / "METADATA").write_text("Name: flash-attn\nVersion: 2.8.3.post1\n")
        self.run_fa()
        self.assertEqual(len(self.calls()), 4)
        self.assertFalse(old.exists())

    def test_build_cache_changes_with_torch_cuda_and_fa(self):
        self.setup_fa()
        self.run_fa()
        keys = {self.calls()[-1]["cache"]}
        for name, value in (
            ("TEST_TORCH", "2.13.1+cu130"),
            ("TEST_CUDA", "13.1"),
            ("FLASH_ATTN_VERSION", "2.8.4"),
        ):
            self.env[name] = value
            (self.root / "ready").unlink()
            self.run_fa()
            keys.add(self.calls()[-1]["cache"])
        self.assertEqual(len(keys), 4)

    def test_install_or_final_abi_failure_stops_ci(self):
        self.setup_fa()
        self.env["TEST_INSTALL_FAIL"] = "1"
        self.run_fa(success=False)
        del self.env["TEST_INSTALL_FAIL"]
        self.env["TEST_ABI_FAIL"] = "1"
        self.run_fa(success=False)

    def test_workflow_orders_install_check_and_snapshot_before_tests(self):
        names = list(self.steps)
        ordered = [
            "Install dependencies",
            "Ensure FlashAttention ABI",
            "Save dependency cache",
            "Run live server-capture numerics gate",
            "Run test",
        ]
        self.assertEqual(sorted(ordered, key=names.index), ordered)
        self.assertNotIn("flash-attn", self.steps["Install dependencies"]["run"])
        self.assertNotIn("/github/home/sf", self.steps["Restore cache"]["run"])


if __name__ == "__main__":
    unittest.main()
