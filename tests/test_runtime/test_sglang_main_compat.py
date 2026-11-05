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
"""The offline capture backend must run on the pinned SGLang and on sglang main."""

import ast
import inspect
import textwrap
import types
import unittest
from unittest import mock

from specforge.offline_capture.sglang_backend import capture as sglang_capture
from specforge.offline_capture.sglang_backend import compat as sglang_compat


class SGLangMainCompatibilityTest(unittest.TestCase):
    def test_require_mlp_sync_supports_both_signatures(self):
        server_args = object()
        seen = []

        def pinned(args):  # 0.5.18: takes ServerArgs
            seen.append(args)
            return True

        def current():  # sglang main: reads the runtime context
            seen.append("no-args")
            return False

        with mock.patch.object(sglang_compat.sglang_utils, "require_mlp_sync", pinned):
            self.assertTrue(sglang_compat.require_mlp_sync(server_args))
        with mock.patch.object(sglang_compat.sglang_utils, "require_mlp_sync", current):
            self.assertFalse(sglang_compat.require_mlp_sync(server_args))
        self.assertEqual(seen, [server_args, "no-args"])

    def test_require_mlp_tp_gather_supports_both_signatures(self):
        server_args = object()
        with mock.patch.object(
            sglang_compat.sglang_utils,
            "require_mlp_tp_gather",
            lambda args: args is server_args,
        ):
            self.assertTrue(sglang_compat.require_mlp_tp_gather(server_args))
        with mock.patch.object(
            sglang_compat.sglang_utils, "require_mlp_tp_gather", lambda: False
        ):
            self.assertFalse(sglang_compat.require_mlp_tp_gather(server_args))

    def test_resolve_device_only_fills_missing_device(self):
        server_args = types.SimpleNamespace(device=None)
        with mock.patch.object(sglang_compat, "get_device_type", return_value="npu"):
            self.assertEqual(sglang_compat.resolve_device(server_args), "npu")
        self.assertEqual(server_args.device, "npu")

        server_args = types.SimpleNamespace(device="xpu")
        self.assertEqual(sglang_compat.resolve_device(server_args), "xpu")
        self.assertEqual(server_args.device, "xpu")

    def test_publish_runtime_context_publishes_once(self):
        server_args = object()
        publish = mock.Mock()
        module = types.SimpleNamespace(publish=publish, publish_role=lambda: None)
        with mock.patch.dict("sys.modules", {"sglang.srt.runtime_context": module}):
            self.assertTrue(sglang_compat.publish_runtime_context(server_args))
        publish.assert_called_once_with(server_args, role="scheduler")

        already = types.SimpleNamespace(
            publish=mock.Mock(), publish_role=lambda: "scheduler"
        )
        with mock.patch.dict("sys.modules", {"sglang.srt.runtime_context": already}):
            self.assertFalse(sglang_compat.publish_runtime_context(server_args))
        already.publish.assert_not_called()

        absent = types.SimpleNamespace()
        with mock.patch.dict("sys.modules", {"sglang.srt.runtime_context": absent}):
            self.assertFalse(sglang_compat.publish_runtime_context(server_args))

    def test_capture_backend_routes_drifted_calls_through_compat(self):
        source = inspect.getsource(sglang_capture)
        tree = ast.parse(source)
        compat_names = set()
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "compat":
                compat_names.update(alias.name for alias in node.names)
        for name in (
            "require_mlp_sync",
            "require_mlp_tp_gather",
            "resolve_device",
            "publish_runtime_context",
        ):
            self.assertIn(name, compat_names)
        # The drifted helpers must not also be imported straight from sglang.
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "sglang.srt.utils":
                imported = {alias.name for alias in node.names}
                self.assertNotIn("require_mlp_sync", imported)
                self.assertNotIn("require_mlp_tp_gather", imported)

        build_tree = ast.parse(
            textwrap.dedent(
                inspect.getsource(sglang_capture.OfflineSGLangCaptureBackend.build)
            )
        )
        called = [
            node.func.id
            for node in sorted(
                (
                    n
                    for n in ast.walk(build_tree)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                ),
                key=lambda n: (n.lineno, n.col_offset),
            )
        ]
        self.assertIn("resolve_device", called)
        self.assertIn("publish_runtime_context", called)
        # Both must run before the ModelRunner is constructed.
        self.assertLess(called.index("resolve_device"), called.index("SGLangRunner"))
        self.assertLess(
            called.index("publish_runtime_context"), called.index("SGLangRunner")
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
