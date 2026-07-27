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
"""Offline capture is the only local SGLang version boundary."""

import ast
import os
import subprocess
import sys
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
            "import specforge.offline_capture.sglang_backend; "
            "assert 'torch' not in sys.modules; "
            "assert 'sglang' not in sys.modules; "
            "assert 'specforge.offline_capture.sglang_backend.capture' "
            "not in sys.modules"
        )
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_backend_package_does_not_eagerly_import_capture(self):
        path = os.path.join(_CAPTURE_DIR, "sglang_backend", "__init__.py")
        with open(path, encoding="utf-8") as source:
            tree = ast.parse(source.read(), filename=path)
        eager_capture_imports = [
            node
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "capture"
        ]
        self.assertEqual(eager_capture_imports, [])

    def test_public_wrapper_has_no_toplevel_sglang_import(self):
        path = os.path.join(_CAPTURE_DIR, "sglang.py")
        self.assertEqual(_toplevel_sglang_imports(path), [])

    def test_capture_backend_is_the_single_sglang_boundary(self):
        path = os.path.join(_CAPTURE_DIR, "sglang_backend", "capture.py")
        hits = _toplevel_sglang_imports(path)
        self.assertTrue(
            hits,
            "OfflineSGLangCaptureBackend is the version-pinned boundary and should "
            "own direct SGLang imports",
        )


if __name__ == "__main__":
    unittest.main()
