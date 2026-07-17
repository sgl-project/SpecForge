# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Phase B — TargetEngine ABC / de-EAGLE3 extraction (structure, no behavior change).

These assertions are about the *abstraction*, not the GPU forward: the generic
``TargetEngine`` ABC, the real ``.backend`` tags, the ``capture`` /
``set_capture_layers`` dispatch onto the algorithm-specific methods, the
back-compat aliases, and the generic ``get_target_engine`` factory. They do not
load a model. (They still import torch, so run on the GPU box like the rest of
tests/test_runtime.)
"""

import unittest

import torch

from specforge.inference.target_engine.base import KNOWN_BACKENDS
from specforge.modeling.target import (
    CustomEagle3TargetEngine,
    CustomEagle3TargetModel,
    Eagle3TargetEngine,
    Eagle3TargetModel,
    HFEagle3TargetEngine,
    HFEagle3TargetModel,
    SGLangEagle3TargetEngine,
    SGLangEagle3TargetModel,
    TargetEngine,
    available_target_engines,
    get_target_engine,
)


class TargetEngineABCTest(unittest.TestCase):
    def test_base_is_abstract(self):
        with self.assertRaises(TypeError):
            TargetEngine()  # abstract: from_pretrained + capture unimplemented

    def test_eagle3_hierarchy_under_target_engine(self):
        # The EAGLE3 algorithm ABC is now a TargetEngine subclass ...
        self.assertTrue(issubclass(Eagle3TargetEngine, TargetEngine))
        # ... and the concrete backends sit under it.
        for cls in (
            HFEagle3TargetEngine,
            SGLangEagle3TargetEngine,
            CustomEagle3TargetEngine,
        ):
            self.assertTrue(issubclass(cls, Eagle3TargetEngine))

    def test_backend_tags_are_real(self):
        # Previously the adapters read getattr(target, "backend", "unknown") and
        # always got "unknown" — the tag is now a real class attribute.
        self.assertEqual(SGLangEagle3TargetEngine.backend, "sglang")
        self.assertEqual(HFEagle3TargetEngine.backend, "hf")
        self.assertEqual(CustomEagle3TargetEngine.backend, "custom")
        self.assertEqual(TargetEngine.backend, "unknown")
        for cls in (
            SGLangEagle3TargetEngine,
            HFEagle3TargetEngine,
            CustomEagle3TargetEngine,
        ):
            self.assertIn(cls.backend, KNOWN_BACKENDS)

    def test_backcompat_aliases_are_identical(self):
        # Pre-Phase-B names resolve to the exact same classes (import-compatible).
        self.assertIs(Eagle3TargetModel, Eagle3TargetEngine)
        self.assertIs(HFEagle3TargetModel, HFEagle3TargetEngine)
        self.assertIs(SGLangEagle3TargetModel, SGLangEagle3TargetEngine)
        self.assertIs(CustomEagle3TargetModel, CustomEagle3TargetEngine)

    def test_capture_dispatches_to_generate_eagle3_data(self):
        calls = {}

        class FakeEagle3(Eagle3TargetEngine):
            backend = "custom"

            @classmethod
            def from_pretrained(cls, *a, **k):  # pragma: no cover - not used
                return cls()

            def generate_eagle3_data(
                self, input_ids, attention_mask, loss_mask, **kwargs
            ):
                calls["args"] = (input_ids, attention_mask, loss_mask)
                calls["kwargs"] = kwargs
                return "sentinel"

        eng = FakeEagle3()
        out = eng.capture(
            input_ids=torch.zeros(1, 3, dtype=torch.long),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            loss_mask=torch.ones(1, 3, dtype=torch.long),
            shard_returns=True,
        )
        self.assertEqual(out, "sentinel")
        # capture forwards extraction kwargs verbatim (byte-identical path).
        self.assertEqual(calls["kwargs"], {"shard_returns": True})

    def test_set_capture_layers_maps_to_aux_layers(self):
        class FakeEagle3(Eagle3TargetEngine):
            @classmethod
            def from_pretrained(cls, *a, **k):  # pragma: no cover
                return cls()

            def generate_eagle3_data(self, *a, **k):  # pragma: no cover
                raise NotImplementedError

        eng = FakeEagle3()
        eng.set_capture_layers([1, 5, 9])  # generic hook -> aux layers
        self.assertEqual(eng.aux_hidden_states_layers, [1, 5, 9])

    def test_factory_lists_and_rejects_unknown_strategy(self):
        self.assertEqual(available_target_engines(), ["dflash", "domino", "eagle3"])
        with self.assertRaises(ValueError):
            get_target_engine("some/path", strategy="does-not-exist")


if __name__ == "__main__":
    unittest.main()
