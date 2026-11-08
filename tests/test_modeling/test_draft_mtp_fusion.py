"""MTP fusion preserves normalization order, mixed widths and gradients."""

import copy
import unittest
from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import patch

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from specforge.config.draft.architecture.features import FinalHiddenFeatureSpec
from specforge.config.draft.norm import RMSNormSpec
from specforge.modeling.draft.layers.inputs import LayerInputs
from specforge.modeling.draft.models.assembly import build_decoder_backbone
from specforge.modeling.draft.models.mtp_fusion import MTPInputFusion
from specforge.modeling.draft.mtp.qwen3_5 import Qwen3_5RMSNorm
from tests.test_config.test_draft_mtp_spec import mtp_config
from tests.test_modeling.test_draft_gqa import layout_for


class MTPFusionTest(unittest.TestCase):
    def _parity(self, device, dtype, autocast=False):
        for offset in (0, 1):
            with self.subTest(
                device=device, dtype=dtype, offset=offset, autocast=autocast
            ):
                spec = replace(
                    mtp_config().spec_config,
                    target_features=FinalHiddenFeatureSpec(hidden_size=24),
                )
                actual = MTPInputFusion(
                    spec,
                    hidden_size=32,
                    norm=RMSNormSpec(eps=1e-6, weight_offset=offset),
                    device=device,
                    dtype=dtype,
                )
                norm_cls = Qwen3_5RMSNorm if offset else Qwen3RMSNorm
                reference = nn.ModuleDict(
                    {
                        "pre_fc_norm_embedding": norm_cls(32),
                        "pre_fc_norm_hidden": norm_cls(24),
                        "fc": nn.Linear(56, 32, bias=False),
                    }
                ).to(device=device, dtype=dtype)
                reference.load_state_dict(actual.state_dict(), strict=True)
                embeddings = torch.randn(
                    2, 4, 32, device=device, dtype=dtype, requires_grad=True
                )
                features = torch.randn(
                    2, 4, 24, device=device, dtype=dtype, requires_grad=True
                )
                other_embeddings, other_features = (
                    embeddings.detach().clone().requires_grad_(),
                    features.detach().clone().requires_grad_(),
                )
                with (
                    torch.autocast(device, dtype=torch.bfloat16)
                    if autocast
                    else nullcontext()
                ):
                    output = actual(embeddings, features)
                    expected = reference["fc"](
                        torch.cat(
                            (
                                reference["pre_fc_norm_embedding"](other_embeddings),
                                reference["pre_fc_norm_hidden"](other_features),
                            ),
                            -1,
                        )
                    )
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                upstream = torch.randn_like(output)
                output.backward(upstream)
                expected.backward(upstream)
                for value, other in (
                    (embeddings, other_embeddings),
                    (features, other_features),
                ):
                    torch.testing.assert_close(value.grad, other.grad, rtol=0, atol=0)
                for name, parameter in actual.named_parameters():
                    torch.testing.assert_close(
                        parameter.grad,
                        dict(reference.named_parameters())[name].grad,
                        rtol=0,
                        atol=0,
                    )

    def test_cpu_reference(self):
        for dtype in (torch.float32, torch.bfloat16):
            self._parity("cpu", dtype)
        self._parity("cpu", torch.float32, autocast=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_reference(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self._parity("cuda", dtype)
        self._parity("cuda", torch.float32, autocast=True)

    def test_fusion_and_decoder_checkpoint_gradients(self):
        config = mtp_config()
        fusion = MTPInputFusion(
            config.spec_config, hidden_size=32, norm=config.norm_config
        )
        backbone = build_decoder_backbone(config, backend="eager")
        other_fusion, other_backbone = copy.deepcopy(fusion), copy.deepcopy(backbone)
        other_backbone.gradient_checkpointing = True
        embeddings, features = (
            torch.randn(2, 4, 32, requires_grad=True) for _ in range(2)
        )
        e2, f2 = (
            embeddings.detach().clone().requires_grad_(),
            features.detach().clone().requires_grad_(),
        )
        layout = layout_for(embeddings)
        entries = [
            LayerInputs(attention_layout=layout, position_ids=layout.key_positions)
        ]
        output = backbone(fusion(embeddings, features), entries)
        expected = other_backbone(other_fusion(e2, f2), entries)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        upstream = torch.randn_like(output)
        output.backward(upstream)
        expected.backward(upstream)
        for value, other in ((embeddings, e2), (features, f2)):
            torch.testing.assert_close(value.grad, other.grad, rtol=0, atol=0)
        for module, reference in ((fusion, other_fusion), (backbone, other_backbone)):
            for name, parameter in module.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                torch.testing.assert_close(
                    parameter.grad,
                    dict(reference.named_parameters())[name].grad,
                    rtol=0,
                    atol=0,
                )

    def test_meta_and_alignment_validation(self):
        config = mtp_config()
        meta = MTPInputFusion(
            config.spec_config,
            hidden_size=32,
            norm=config.norm_config,
            device="meta",
            dtype=torch.bfloat16,
        )
        self.assertTrue(
            all(p.is_meta and p.dtype == torch.bfloat16 for p in meta.parameters())
        )
        actual = MTPInputFusion(
            config.spec_config, hidden_size=32, norm=config.norm_config
        )
        with patch.object(actual.fc, "forward") as project:
            for shape in ((2, 3, 32), (2, 4, 16), (4, 32)):
                with self.assertRaises(ValueError):
                    actual(torch.empty(2, 4, 32), torch.empty(shape))
            project.assert_not_called()


if __name__ == "__main__":
    unittest.main()
