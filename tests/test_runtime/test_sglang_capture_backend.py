# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Offline capture keeps SGLang internals in the server process."""

import ast
import os
import subprocess
import sys
import textwrap
import unittest

_CAPTURE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "specforge", "offline_capture")
)


def _toplevel_sglang_imports(path):
    """Return module-level imports rooted at ``sglang``."""
    with open(path, "r", encoding="utf-8") as source:
        tree = ast.parse(source.read(), filename=path)
    hits = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            hits.extend(
                name.name for name in node.names if name.name.split(".")[0] == "sglang"
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                if node.module.split(".")[0] == "sglang":
                    hits.append(node.module)
    return hits


class OfflineSglangBoundaryTest(unittest.TestCase):
    def test_package_import_does_not_load_torch_or_sglang(self):
        repo_root = os.path.dirname(os.path.dirname(_CAPTURE_DIR))
        code = (
            "import sys; "
            "import specforge.offline_capture; "
            "assert 'torch' not in sys.modules; "
            "assert 'sglang' not in sys.modules; "
        )
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_offline_capture_round_trip_without_sglang(self):
        # Exercise feature capture, file publication and resume with the real
        # plugin sink and a fake transport. Any SGLang import must fail even
        # when the test environment happens to have it installed.
        code = textwrap.dedent(
            """
            import importlib.abc
            import sys
            import unittest

            class NoSglang(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "sglang" or fullname.startswith("sglang."):
                        raise ModuleNotFoundError("offline client imported " + fullname)

            sys.meta_path.insert(0, NoSglang())
            suite = unittest.defaultTestLoader.loadTestsFromName(
                "tests.test_offline_capture.test_client.OfflineCaptureTest."
                "test_algorithms_round_trip_and_resume"
            )
            result = unittest.TextTestRunner().run(suite)
            assert result.wasSuccessful()
            assert not any(n == "sglang" or n.startswith("sglang.") for n in sys.modules)
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.path.dirname(os.path.dirname(_CAPTURE_DIR)),
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_public_wrapper_has_no_toplevel_sglang_import(self):
        path = os.path.join(_CAPTURE_DIR, "client.py")
        self.assertEqual(_toplevel_sglang_imports(path), [])


if __name__ == "__main__":
    unittest.main()
