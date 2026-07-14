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
"""The generic target-engine layer is decoupled from the SGLang version."""

import ast
import os
import unittest

_TARGET_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "specforge", "inference", "target_engine"
    )
)


def _toplevel_sglang_imports(path):
    """Return module-level imports rooted at ``sglang``."""
    with open(path, "r", encoding="utf-8") as source:
        tree = ast.parse(source.read(), filename=path)
    hits = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            hits.extend(
                name.name
                for name in node.names
                if name.name.split(".")[0] == "sglang"
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                if node.module.split(".")[0] == "sglang":
                    hits.append(node.module)
    return hits


class EngineSglangDecouplingTest(unittest.TestCase):
    def test_generic_engine_has_no_toplevel_sglang_import(self):
        path = os.path.join(_TARGET_DIR, "sglang.py")
        self.assertEqual(_toplevel_sglang_imports(path), [])

    def test_capture_policies_have_no_toplevel_sglang_import(self):
        path = os.path.join(_TARGET_DIR, "target_capture_policy.py")
        self.assertEqual(_toplevel_sglang_imports(path), [])

    def test_capture_backend_is_the_single_sglang_boundary(self):
        path = os.path.join(_TARGET_DIR, "sglang_backend", "capture.py")
        hits = _toplevel_sglang_imports(path)
        self.assertTrue(
            hits,
            "SGLangCaptureBackend is the version-pinned boundary and should "
            "own direct SGLang imports",
        )


if __name__ == "__main__":
    unittest.main()
