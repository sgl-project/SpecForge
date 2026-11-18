"""Unset kernel settings resolve to the validated defaults; explicit ones are strict."""

import unittest

import torch

from specforge.modeling.draft.layers.conv.grouped_causal import GroupedCausalConv
from specforge.modeling.draft.layers.kernels import configure_kernels, liger_hidden_act
from specforge.training.canonical import liger_choice
from tests.test_algorithms.test_objective_contracts import build_runtime
from tests.test_config.test_recipe_execution_defaults import protocol_chunk


class KernelDefaultsTest(unittest.TestCase):
    def test_fused_convolution_follows_what_the_trainer_supports(self):
        _, runtime = build_runtime("DFlash2DraftModel")
        model = runtime.draft_model
        convolutions = [m for m in model.modules() if isinstance(m, GroupedCausalConv)]
        self.assertTrue(convolutions)
        # CPU weights: auto keeps the eager convolution, an explicit request fails.
        self.assertFalse(configure_kernels(model, use_fused_conv=None))
        self.assertFalse(any(module.use_fused for module in convolutions))
        with self.assertRaisesRegex(ValueError, "requires CUDA/ROCm"):
            configure_kernels(model, use_fused_conv=True)
        self.assertFalse(configure_kernels(model, use_fused_conv=False))
        _, plain = build_runtime("DSparkDraftModel")
        self.assertFalse(configure_kernels(plain.draft_model, use_fused_conv=None))
        with self.assertRaisesRegex(ValueError, "requires a grouped convolution"):
            configure_kernels(plain.draft_model, use_fused_conv=True)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_convolution_is_the_cuda_default(self):
        _, runtime = build_runtime("DFlash2DraftModel", dtype=torch.bfloat16)
        model = runtime.draft_model.cuda()
        self.assertTrue(configure_kernels(model, use_fused_conv=None))
        self.assertFalse(configure_kernels(model, use_fused_conv=False))

    def test_liger_is_a_default_only_where_declared(self):
        _, runtime = build_runtime("DFlashDraftModel")
        act = liger_hidden_act(runtime.draft_model)
        self.assertIn(act, ("silu", "swish"))

        def choice(requested, by_default, platform, hidden_act=act):
            return liger_choice(
                requested,
                by_default=by_default,
                hidden_act=hidden_act,
                platform=platform,
            ).enabled

        self.assertTrue(choice(None, True, "cuda"))
        self.assertTrue(choice(None, True, "rocm"))
        self.assertFalse(choice(None, False, "cuda"))
        self.assertFalse(choice(None, True, "cpu"))
        self.assertFalse(choice(None, True, "npu"))
        self.assertFalse(choice(None, True, "cuda", hidden_act="gelu"))
        self.assertFalse(choice(False, True, "cuda"))
        self.assertTrue(choice(True, False, "cuda"))
        with self.assertRaisesRegex(ValueError, "requires a Triton GPU backend"):
            choice(True, True, "cpu")

    def test_startup_preflight_reads_nothing_where_auto_cannot_enable_liger(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from specforge.training import canonical

        cfg = SimpleNamespace(model=SimpleNamespace(use_liger_kernel=None))
        with (
            patch.object(canonical, "trainer_platform", return_value="cpu"),
            patch.object(canonical, "resolve_config") as resolve,
        ):
            canonical.preflight_kernels(cfg)
        resolve.assert_not_called()
        cfg.model.use_liger_kernel = False
        with patch.object(canonical, "resolve_config") as resolve:
            canonical.preflight_kernels(cfg)
        resolve.assert_not_called()

    def test_block_objectives_bound_the_loss_chunk_per_row(self):
        for architecture, expected in (
            ("DFlashDraftModel", (128, True)),
            ("DFlash2DraftModel", (128, True)),
            ("DominoDraftModel", (128, True)),
            ("DSparkDraftModel", (128, True)),
            ("DSpineDraftModel", (128, True)),
            ("MTPDraftModel", (4096, False)),
            ("Eagle3DraftModel", (0, False)),
        ):
            with self.subTest(architecture=architecture):
                self.assertEqual(protocol_chunk(architecture), expected)

    def test_objective_model_reports_the_implementations_it_resolved(self):
        _, runtime = build_runtime("DFlashDraftModel")
        runtime.loss_implementations.names["0:0"] = "reference"
        kernels = runtime.kernel_record()
        self.assertEqual(kernels, {"0:0": "reference"})
        # A copy: what a checkpoint compares must not change under it.
        self.assertIsNot(kernels, runtime.loss_implementations.names)


if __name__ == "__main__":
    unittest.main()
