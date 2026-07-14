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
"""Structural tests for the canonical, policy-driven target engines."""

import unittest
from unittest import mock

from specforge.inference.target_engine import (
    CustomTargetEngine,
    HFTargetEngine,
    SGLangTargetEngine,
    TargetEngine,
    available_target_engines,
    get_target_engine,
)
from specforge.inference.target_engine.base import KNOWN_BACKENDS


class TargetEngineABCTest(unittest.TestCase):
    def test_base_is_abstract(self):
        with self.assertRaises(TypeError):
            TargetEngine()

    def test_generic_backends_share_one_abc(self):
        for engine_cls in (HFTargetEngine, SGLangTargetEngine, CustomTargetEngine):
            with self.subTest(engine=engine_cls.__name__):
                self.assertTrue(issubclass(engine_cls, TargetEngine))
                self.assertIn(engine_cls.backend, KNOWN_BACKENDS)

    def test_backend_tags_are_explicit(self):
        self.assertEqual(HFTargetEngine.backend, "hf")
        self.assertEqual(SGLangTargetEngine.backend, "sglang")
        self.assertEqual(CustomTargetEngine.backend, "custom")
        self.assertEqual(TargetEngine.backend, "unknown")

    def test_factory_routes_to_generic_hf_engine(self):
        sentinel = object()
        with mock.patch.object(
            HFTargetEngine, "from_pretrained", return_value=sentinel
        ) as load:
            engine = get_target_engine(
                "some/path", strategy="eagle3", backend="hf", device="cuda"
            )
        self.assertIs(engine, sentinel)
        self.assertEqual(load.call_args.args, ("some/path",))
        self.assertEqual(load.call_args.kwargs["device"], "cuda")
        self.assertEqual(load.call_args.kwargs["policy"].spec.name, "eagle3")

    def test_factory_routes_to_generic_sglang_and_custom_engines(self):
        for backend, engine_cls in (
            ("sglang", SGLangTargetEngine),
            ("custom", CustomTargetEngine),
        ):
            with self.subTest(backend=backend), mock.patch.object(
                engine_cls, "from_pretrained", return_value=backend
            ) as load:
                self.assertEqual(
                    get_target_engine("some/path", strategy="dflash", backend=backend),
                    backend,
                )
                self.assertEqual(load.call_args.kwargs["policy"].spec.name, "dflash")

    def test_factory_lists_and_rejects_unknown_values(self):
        self.assertEqual(available_target_engines(), ["dflash", "domino", "eagle3"])
        with self.assertRaises(ValueError):
            get_target_engine("some/path", strategy="does-not-exist")
        with self.assertRaises(ValueError):
            get_target_engine("some/path", strategy="eagle3", backend="other")


if __name__ == "__main__":
    unittest.main()
