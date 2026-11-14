"""Old loss imports remain aliases of the shared numerical modules."""

import importlib
import importlib.util
import unittest
from unittest.mock import patch


class LossImportTest(unittest.TestCase):
    def assert_alias(self, old, new):
        legacy = importlib.import_module("specforge.core." + old)
        shared = importlib.import_module("specforge.losses." + new)
        self.assertIs(legacy, shared)
        with patch.object(legacy, "_compatibility_probe", True, create=True):
            self.assertTrue(shared._compatibility_probe)

    def test_portable_aliases(self):
        for old, new in (
            ("fused_loss", "fused"),
            ("domino_loss", "domino"),
            ("lk_loss", "lk"),
            ("position_weights", "position_weights"),
        ):
            with self.subTest(module=old):
                self.assert_alias(old, new)

    @unittest.skipUnless(importlib.util.find_spec("triton"), "requires Triton")
    def test_triton_aliases(self):
        for old, new in (
            ("loss", "soft_cross_entropy"),
            ("domino_loss_triton", "_domino_triton"),
        ):
            with self.subTest(module=old):
                self.assert_alias(old, new)
