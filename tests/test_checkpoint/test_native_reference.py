"""Independent Qwen3.5 metadata and HF decoder references for native presets."""

import json
import unittest
from unittest.mock import Mock, patch

import torch
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding,
)

from specforge.checkpoint.formats.native.writer import export_native
from specforge.checkpoint.initialization import make_initialization_plan
from specforge.checkpoint.tensors import TensorInfo, TensorStore
from specforge.modeling import build_draft_model
from specforge.modeling.draft.layers.attention.masks import AttentionLayout
from specforge.modeling.draft.layers.inputs import LayerInputs
from tests.test_checkpoint.test_native_contracts import FIXTURE, native_fixture
from tests.test_checkpoint.test_native_presets import NATIVE_MAPPING, native_configs
from tests.test_modeling.test_draft_decoder import prepare


class NativeReferenceTest(unittest.TestCase):
    def test_real_headers_cover_initialization_and_merge(self):
        config, target = native_fixture()
        headers = json.loads((FIXTURE / "headers.json").read_text())
        store = object.__new__(TensorStore)
        store.metadata = {
            name: TensorInfo(tuple(info["shape"]), torch.bfloat16)
            for name, info in headers.items()
        }
        self.assertTrue(all(info["dtype"] == "BF16" for info in headers.values()))
        store.files = {name: FIXTURE / "model.safetensors" for name in headers}
        store.read = Mock(side_effect=AssertionError("tensor read"))
        model = build_draft_model(config, device="meta")
        plan = make_initialization_plan(
            model,
            target_store=store,
            target_config=target["text_config"],
            from_target=NATIVE_MAPPING,
        )
        self.assertEqual(
            {ref.key for _, ref in plan.weights},
            {k for k in headers if k.startswith("mtp.")},
        )
        self.assertEqual(len(plan.weights), 15)
        self.assertEqual(
            {name for name, _ in plan.bindings}, {"embed_tokens", "lm_head"}
        )
        self.assertEqual(len({ref.key for _, ref in plan.bindings}), 1)
        store.read.assert_not_called()
        # Export validates the independent names/shapes before its first data read.
        with patch(
            "specforge.checkpoint.formats.native.writer.TensorStore", return_value=store
        ):
            with self.assertRaisesRegex(AssertionError, "tensor read"):
                export_native(
                    config,
                    model.state_dict(),
                    FIXTURE,
                    "unused",
                )

    @torch.no_grad()
    def test_decoder_matches_hf_with_distinct_axis_positions(self):
        _, target = native_fixture()
        config = next(native_configs(small=True))
        # Scale only widths; every computational setting comes from the real config.
        hf = Qwen3_5TextConfig(
            **(
                target["text_config"]
                | {
                    "hidden_size": 32,
                    "vocab_size": 64,
                    "eos_token_id": None,
                    "intermediate_size": 64,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                }
            )
        )
        hf._attn_implementation = "eager"
        reference = Qwen3_5DecoderLayer(hf, layer_idx=3).eval()
        model = build_draft_model(config).eval()
        layer = model.backbone.layers[0]
        # Use nonzero norm weights as well as projections to exercise offset norms.
        with torch.random.fork_rng():
            torch.manual_seed(1163)
            for parameter in reference.parameters():
                parameter.normal_(std=0.1)
            hidden = torch.randn(1, 5, 32)
        layer.load_state_dict(reference.state_dict(), strict=True)
        positions = torch.tensor(
            [
                [[0, 1, 31, 127, 511]],
                [[3, 19, 73, 257, 1023]],
                [[7, 37, 151, 509, 2047]],
            ]
        )
        rotary = Qwen3_5TextRotaryEmbedding(hf, device="cpu")
        mask = torch.ones(5, 5, dtype=torch.bool).tril()[None, None]
        additive = torch.zeros_like(mask, dtype=hidden.dtype).masked_fill(
            ~mask, float("-inf")
        )
        expected = reference(hidden, rotary(hidden, positions), attention_mask=additive)
        indices = torch.arange(5)[None]
        layout = AttentionLayout(
            query_positions=indices, key_positions=indices, visibility=mask
        )
        inputs = LayerInputs(
            attention=prepare(layer, layout, 1),
            position_embeddings=model.backbone.rotary[0](hidden, positions),
        )
        actual = layer(hidden, inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
