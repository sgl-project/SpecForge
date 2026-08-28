"""Regression tests for cached-venv spec-capture patch transitions."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "apply_sglang_spec_capture_patch.sh"


def _patch(base: str, patched: str, sink: str) -> str:
    return f"""\
diff --git a/python/sglang/srt/example.py b/python/sglang/srt/example.py
--- a/python/sglang/srt/example.py
+++ b/python/sglang/srt/example.py
@@ -1 +1 @@
-{base}
+{patched}
diff --git a/python/sglang/srt/spec_capture_sink.py b/python/sglang/srt/spec_capture_sink.py
new file mode 100644
--- /dev/null
+++ b/python/sglang/srt/spec_capture_sink.py
@@ -0,0 +1 @@
+{sink}
"""


class ApplySglangSpecCapturePatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.site_packages = Path(self.temp_dir.name) / "site-packages"
        self.srt = self.site_packages / "sglang" / "srt"
        self.srt.mkdir(parents=True)
        self.example = self.srt / "example.py"
        self.sink = self.srt / "spec_capture_sink.py"
        self.record = self.site_packages / "sglang" / ".spec_capture_patch.applied"
        self.new_patch = Path(self.temp_dir.name) / "new.patch"
        self.new_patch.write_text(
            _patch("new base", "new patched", "new sink"), encoding="utf-8"
        )
        self.env = {
            **os.environ,
            "SPECFORGE_SGLANG_ROOT": str(self.site_packages),
            "SPECFORGE_SGLANG_VERSION": "0.5.18",
            "SPECFORGE_SPEC_CAPTURE_PATCH": str(self.new_patch),
        }

    def run_script(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPT)],
            check=False,
            capture_output=True,
            env=self.env,
            text=True,
        )

    def seed_cached_upgrade(self, installed_source: str = "new base") -> str:
        old_patch = _patch("old base", "old patched", "old sink")
        self.example.write_text(f"{installed_source}\n", encoding="utf-8")
        self.sink.write_text("old sink\n", encoding="utf-8")
        self.record.write_text(old_patch, encoding="utf-8")
        return old_patch

    def test_recovers_files_left_by_pip_upgrade_and_is_idempotent(self) -> None:
        self.seed_cached_upgrade()

        result = self.run_script()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("recovered stale spec-capture files", result.stdout)
        self.assertEqual(self.example.read_text(encoding="utf-8"), "new patched\n")
        self.assertEqual(self.sink.read_text(encoding="utf-8"), "new sink\n")
        self.assertEqual(
            self.record.read_text(encoding="utf-8"),
            self.new_patch.read_text(encoding="utf-8"),
        )

        second_result = self.run_script()
        self.assertEqual(second_result.returncode, 0, second_result.stderr)
        self.assertIn("already applied", second_result.stdout)

    def test_restores_stale_sink_when_new_patch_does_not_apply(self) -> None:
        old_patch = self.seed_cached_upgrade(installed_source="unexpected source")

        result = self.run_script()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown spec-capture patch state", result.stderr)
        self.assertEqual(
            self.example.read_text(encoding="utf-8"), "unexpected source\n"
        )
        self.assertEqual(self.sink.read_text(encoding="utf-8"), "old sink\n")
        self.assertEqual(self.record.read_text(encoding="utf-8"), old_patch)


if __name__ == "__main__":
    unittest.main()
