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
                version = os.environ["FLASH_ATTN_VERSION"]
                with (root / "calls").open("a") as log:
                    log.write(json.dumps({"args": sys.argv[1:], "cache": str(cache)}) + "\\n")
                if sys.argv[2] == "uninstall":
                    for entry in modules.glob("flash_attn-*.dist-info"):
                        shutil.rmtree(entry)
                else:
                    metadata = modules / f"flash_attn-{version}.dist-info"
                    metadata.mkdir(exist_ok=True)
                    (metadata / "METADATA").write_text(
                        f"Name: flash-attn\\nVersion: {version}\\n")
                    if not os.environ.get("TEST_ABI_FAIL"):
                        (root / "ready").touch()
                """
            )
        )
        uv.chmod(0o755)
        # curl serves the PyPI JSON and an sdist tarball whose setup.py builds a
        # real (trivial) wheel once the CI script patches its -std flag. The
        # tarball is byte-stable so the JSON can carry its sha256.
        curl = self.bin / "curl"
        curl.write_text(
            f"#!{sys.executable}\n"
            + textwrap.dedent(
                """\
                import gzip
                import hashlib
                import io
                import json
                import os
                import sys
                import tarfile
                from pathlib import Path

                def sdist():
                    version = os.environ["FLASH_ATTN_VERSION"]
                    payload = (
                        "import os, sys\\n"
                        "# flags: -std=c++17\\n"
                        "if os.environ.get('TEST_INSTALL_FAIL'):\\n"
                        "    sys.exit(1)\\n"
                        "assert os.environ.get('FLASH_ATTENTION_FORCE_BUILD') == 'TRUE'\\n"
                        "src = open(__file__).read()\\n"
                        "assert '-std=' + 'c++17' not in src, \\"setup.py not patched\\"\\n"
                        "from setuptools import setup\\n"
                        "setup(name='flash-attn', version=os.environ['FLASH_ATTN_VERSION'])\\n"
                    ).encode()
                    raw = io.BytesIO()
                    with tarfile.open(fileobj=raw, mode="w") as tar:
                        info = tarfile.TarInfo(f"flash_attn-{version}/setup.py")
                        info.size = len(payload)
                        tar.addfile(info, io.BytesIO(payload))
                    return gzip.compress(raw.getvalue(), mtime=0)

                args = sys.argv[1:]
                with (Path(os.environ["TEST_ROOT"]) / "downloads").open("a") as log:
                    log.write(" ".join(args) + "\\n")
                if "-o" not in args:
                    digest = hashlib.sha256(sdist()).hexdigest()
                    if os.environ.get("TEST_BAD_SHA256"):
                        digest = "0" * 64
                    print(json.dumps({"urls": [{
                        "packagetype": "sdist",
                        "url": "https://files.example.invalid/flash_attn.tar.gz",
                        "digests": {"sha256": digest},
                    }]}))
                    sys.exit(0)
                Path(args[args.index("-o") + 1]).write_bytes(sdist())
                """
            )
        )
        curl.chmod(0o755)

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

    def sdist_downloads(self):
        downloads = self.root / "downloads"
        if not downloads.exists():
            return 0
        return downloads.read_text().count("flash_attn.tar.gz")

    def cached_wheels(self):
        return sorted(p.name for p in self.cache.glob("flash-attn/*/wheels/*"))

    def test_save_and_restore_remove_stale_versions(self):
        for environment in ("sf", "sf-capture"):
            (self.cache / environment / "old.dist-info").mkdir(parents=True)
            current = self.root / environment
            current.mkdir()
            (current / "current.dist-info").write_text(environment)
            (current / "python").symlink_to(sys.executable)
        self.run_step("Save dependency cache")
        for environment in ("sf", "sf-capture"):
            self.assertFalse((self.cache / environment / "old.dist-info").exists())
            (self.root / environment / "stale.dist-info").touch()
        # Exercise only the venv restore; the model cache is unrelated to this test.
        restore = self.steps["Restore cache"]["run"]
        restore = restore[restore.index("for environment in sf sf-capture; do") :]
        result = self.run_shell(restore)
        self.assertEqual(result.returncode, 0, result.stderr)
        for environment in ("sf", "sf-capture"):
            with self.subTest(environment=environment):
                restored = self.root / environment
                self.assertEqual(
                    {p.name for p in restored.iterdir()},
                    {"current.dist-info", "python"},
                )
                self.assertEqual(
                    (restored / "current.dist-info").read_text(), environment
                )
                self.assertTrue((restored / "python").is_symlink())

    def test_failed_copy_preserves_previous_snapshot(self):
        for environment in ("sf", "sf-capture"):
            (self.cache / environment).mkdir()
            (self.cache / environment / "valid").touch()
        result = self.run_shell(self.steps["Save dependency cache"]["run"])
        self.assertNotEqual(result.returncode, 0)
        for environment in ("sf", "sf-capture"):
            self.assertTrue((self.cache / environment / "valid").exists())
        self.assertEqual(list(self.cache.glob("sf.*")), [])

    def test_fa_install_then_healthy_environment_skips_install(self):
        self.setup_fa()
        self.run_fa()
        calls = self.calls()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["args"], ["pip", "uninstall", "flash-attn"])
        install = calls[1]
        self.assertEqual(install["args"][:2], ["pip", "install"])
        self.assertTrue(install["args"][-1].endswith(".whl"))
        self.assertIn("--no-deps", install["args"])
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

    def test_only_a_wheel_that_passes_the_abi_check_is_cached(self):
        self.setup_fa()
        self.env["TEST_ABI_FAIL"] = "1"
        self.run_fa(success=False)
        self.assertEqual(self.cached_wheels(), [])
        del self.env["TEST_ABI_FAIL"]
        self.run_fa()
        self.assertEqual(self.sdist_downloads(), 2)
        (wheel,) = self.cached_wheels()
        self.assertTrue(wheel.startswith("flash_attn-") and wheel.endswith(".whl"))

    def test_cached_wheel_that_fails_is_dropped_and_rebuilt(self):
        self.setup_fa()
        self.run_fa()
        self.assertEqual(len(self.cached_wheels()), 1)
        (self.root / "ready").unlink()
        self.env["TEST_ABI_FAIL"] = "1"
        self.run_fa(success=False)
        self.assertEqual(self.sdist_downloads(), 1)
        self.assertEqual(self.cached_wheels(), [])
        del self.env["TEST_ABI_FAIL"]
        self.run_fa()
        self.assertEqual(self.sdist_downloads(), 2)
        self.assertEqual(len(self.cached_wheels()), 1)

    def test_sdist_checksum_mismatch_stops_before_build(self):
        self.setup_fa()
        self.env["TEST_BAD_SHA256"] = "1"
        self.run_fa(success=False)
        self.assertEqual(self.cached_wheels(), [])
        self.assertFalse((self.root / "calls").exists())

    def test_trusting_the_workspace_is_idempotent(self):
        self.env["GIT_CONFIG_GLOBAL"] = str(self.root / "gitconfig")
        self.env["GITHUB_WORKSPACE"] = str(self.root / "workspace")
        for _ in range(2):
            self.run_step("Trust the workspace for later git calls")
        result = self.run_shell("git config --global --get-all safe.directory")
        self.assertEqual(result.stdout.splitlines(), [self.env["GITHUB_WORKSPACE"]])

    def test_workflow_orders_install_check_and_snapshot_before_tests(self):
        names = list(self.steps)
        ordered = [
            "Install dependencies",
            "Install pinned plugin capture environment",
            "Ensure FlashAttention ABI",
            "Save dependency cache",
            "Run live server-capture numerics gate",
            "Run test",
        ]
        self.assertEqual(sorted(ordered, key=names.index), ordered)
        self.assertNotIn("flash-attn", self.steps["Install dependencies"]["run"])
        self.assertNotIn("/github/home/sf", self.steps["Restore cache"]["run"])

    def test_both_environments_install_ci_pin_without_package_manifest(self):
        self.env["SGLANG_COMMIT"] = self.job["env"]["SGLANG_COMMIT"]
        uv = self.bin / "uv"
        uv.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "with (Path(os.environ['TEST_ROOT']) / 'calls').open('a') as log:\n"
            "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        )
        uv.chmod(0o755)
        for environment in ("sf", "sf-capture"):
            bin_dir = self.root / environment / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "activate").touch()
            # Stub only the capture environment's post-install import check.
            python = bin_dir / "python"
            python.write_text("#!/bin/sh\nexit 0\n")
            python.chmod(0o755)
        self.run_step("Install dependencies")
        self.run_step("Install pinned plugin capture environment")
        requirement = (
            "sglang @ git+https://github.com/sgl-project/sglang.git@"
            f"{self.env['SGLANG_COMMIT']}#subdirectory=python"
        )
        installs = [args for args in self.calls() if requirement in args]
        self.assertEqual(len(installs), 2)
        self.assertIn(".", installs[0])
        self.assertIn("sf-capture/bin/python", installs[1])


if __name__ == "__main__":
    unittest.main()
