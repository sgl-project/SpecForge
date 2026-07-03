# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Phase B2 — target engine decoupled from the sglang version.

The core invariant: the *algorithm* engine modules (``eagle3_target_model`` /
``dflash_target_model``) must not import sglang internals at module load — the
entire sglang-version coupling lives behind ``SGLangCaptureBackend``. That
invariant is checked here with a pure-AST scan (no torch needed, so it runs even
where torch cannot import). A sibling test asserts ``sglang_server`` is
selectable through the factory.
"""

import ast
import os
import unittest

_TARGET_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "specforge", "modeling", "target"
    )
)


def _toplevel_sglang_imports(path):
    """Module-level `import sglang…` / `from sglang… import` names (level 0)."""
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    hits = []
    for node in tree.body:  # module level only — lazy imports inside defs are OK
        if isinstance(node, ast.Import):
            hits += [n.name for n in node.names if n.name.split(".")[0] == "sglang"]
        elif isinstance(node, ast.ImportFrom):
            if (
                node.level == 0
                and node.module
                and node.module.split(".")[0] == "sglang"
            ):
                hits.append(node.module)
    return hits


class EngineSglangDecouplingTest(unittest.TestCase):
    """The engines are sglang-version-agnostic; the backend owns sglang."""

    def test_eagle3_engine_has_no_toplevel_sglang_import(self):
        hits = _toplevel_sglang_imports(
            os.path.join(_TARGET_DIR, "eagle3_target_model.py")
        )
        self.assertEqual(hits, [], f"eagle3 engine leaks sglang imports: {hits}")

    def test_dflash_engine_has_no_toplevel_sglang_import(self):
        hits = _toplevel_sglang_imports(
            os.path.join(_TARGET_DIR, "dflash_target_model.py")
        )
        self.assertEqual(hits, [], f"dflash engine leaks sglang imports: {hits}")

    def test_capture_backend_is_the_single_sglang_boundary(self):
        # The backend is *allowed* (and expected) to import sglang directly.
        hits = _toplevel_sglang_imports(
            os.path.join(_TARGET_DIR, "sglang_backend", "capture.py")
        )
        self.assertTrue(
            hits,
            "SGLangCaptureBackend is the version-pinned boundary; it should "
            "import sglang internals directly",
        )


class SglangServerBackendTest(unittest.TestCase):
    """sglang_server is selectable via the factory (capture gated by O1.3)."""

    def test_server_backend_tag_and_gated_construction(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch unavailable")
        from specforge.modeling.target import (
            SGLangServerEagle3TargetEngine,
            get_eagle3_target_model,
        )

        self.assertEqual(SGLangServerEagle3TargetEngine.backend, "sglang_server")
        # Selectable, but gated: construction raises an actionable NotImplementedError.
        with self.assertRaises(NotImplementedError):
            get_eagle3_target_model("some/model", backend="sglang_server")


if __name__ == "__main__":
    unittest.main()
