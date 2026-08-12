"""Safe PyTorch compiler defaults for variable-length training shapes."""

import os
import unittest
from unittest import mock

from torch._inductor import config as inductor_config

from specforge.torch_environment import configure_flex_attention_inductor


class TorchEnvironmentTest(unittest.TestCase):
    def test_flex_attention_keeps_aten_as_dynamic_shape_fallback(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(
                inductor_config,
                "max_autotune_gemm_backends",
                "TRITON",
            ),
        ):
            self.assertTrue(configure_flex_attention_inductor("flex_attention"))
            self.assertEqual(
                os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"],
                "ATEN,TRITON",
            )
            self.assertEqual(
                inductor_config.max_autotune_gemm_backends,
                "ATEN,TRITON",
            )

    def test_explicit_operator_choice_remains_authoritative(self):
        with mock.patch.dict(
            os.environ,
            {"TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS": "TRITON"},
            clear=True,
        ):
            self.assertFalse(configure_flex_attention_inductor("flex_attention"))
            self.assertEqual(
                os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"],
                "TRITON",
            )

    def test_non_flex_attention_does_not_change_inductor(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(configure_flex_attention_inductor("sdpa"))
            self.assertNotIn(
                "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS",
                os.environ,
            )


if __name__ == "__main__":
    unittest.main()
