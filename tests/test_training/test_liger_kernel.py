"""Kernel selection is explicit and optional imports stay lazy."""

import unittest
from unittest.mock import patch

from specforge.modeling import build_draft_model
from specforge.modeling.draft.layers.kernels import configure_kernels
from tests.test_config.draft_fixtures import draft_config


class LigerChoiceTest(unittest.TestCase):
    def test_disabled_does_not_import_liger(self):
        model = build_draft_model(
            draft_config(target_bindings=(), target_logit_transform=None)
        )
        with patch.dict("sys.modules", {"liger_kernel": None}):
            configure_kernels(model, use_liger=False)

    def test_explicit_cpu_request_fails(self):
        model = build_draft_model(
            draft_config(target_bindings=(), target_logit_transform=None)
        )
        with self.assertRaisesRegex(ValueError, "CUDA/ROCm"):
            configure_kernels(model, use_liger=True)
