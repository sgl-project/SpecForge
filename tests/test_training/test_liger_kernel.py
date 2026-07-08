from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from specforge.training import liger


def _cfg(*, enabled: bool, strategy: str = "dflash"):
    return types.SimpleNamespace(
        model=types.SimpleNamespace(use_liger_kernel=enabled),
        training=types.SimpleNamespace(strategy=strategy),
    )


class TestLigerKernelIntegration(unittest.TestCase):
    def test_disabled_config_does_not_import_or_patch_liger(self):
        with mock.patch.object(liger, "_load_liger_apply") as loader:
            liger.maybe_apply_liger_kernel(_cfg(enabled=False))
        loader.assert_not_called()

    def test_dflash_uses_the_frozen_safe_liger_patch_options(self):
        apply = mock.Mock()
        with mock.patch.object(liger, "_load_liger_apply", return_value=apply):
            liger.maybe_apply_liger_kernel(_cfg(enabled=True))
        apply.assert_called_once_with(
            rope=False,
            rms_norm=True,
            swiglu=True,
            cross_entropy=False,
            fused_linear_cross_entropy=False,
        )

    def test_non_dflash_strategy_is_rejected_before_import(self):
        with mock.patch.object(liger, "_load_liger_apply") as loader:
            with self.assertRaisesRegex(ValueError, "strategy=dflash"):
                liger.maybe_apply_liger_kernel(_cfg(enabled=True, strategy="dspark"))
        loader.assert_not_called()

    def test_missing_optional_extra_has_an_actionable_error(self):
        with mock.patch.dict(
            sys.modules,
            {"liger_kernel": None, "liger_kernel.transformers": None},
        ):
            with self.assertRaisesRegex(ImportError, r"specforge\[liger\]"):
                liger._load_liger_apply()


if __name__ == "__main__":
    unittest.main()
