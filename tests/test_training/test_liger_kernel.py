"""Runtime kernel choices are explicit and do not import Liger when disabled."""

import itertools
import unittest
from unittest.mock import patch

from specforge.training.kernels import resolve_liger


class LigerChoiceTest(unittest.TestCase):
    def test_disabled_or_inapplicable_defaults_do_not_import_liger(self):
        with patch.dict("sys.modules", {"liger_kernel": None}):
            for device, activation, default in itertools.product(
                ("cpu", "cuda", "npu", "xpu"), ("silu", "gelu"), (False, True)
            ):
                self.assertFalse(
                    resolve_liger(
                        False,
                        activations=[activation],
                        device=device,
                        default_enabled=default,
                    )
                )
                if not (default and device == "cuda" and activation == "silu"):
                    self.assertFalse(
                        resolve_liger(
                            None,
                            activations=[activation],
                            device=device,
                            default_enabled=default,
                        )
                    )
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                resolve_liger(
                    None, activations=["silu"], device="cuda", default_enabled=True
                )

    def test_explicit_incompatible_request_fails(self):
        for device, activation in (("cpu", "silu"), ("cuda", "gelu")):
            with self.assertRaisesRegex(ValueError, "Triton accelerator"):
                resolve_liger(
                    True, activations=[activation], device=device, default_enabled=False
                )
        with self.assertRaisesRegex(ValueError, "boolean"):
            resolve_liger(
                "true", activations=["silu"], device="cuda", default_enabled=False
            )


if __name__ == "__main__":
    unittest.main()
