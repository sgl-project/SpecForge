import json
import unittest
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from torch import nn
from transformers import DynamicCache, Qwen3Config

from scripts.gates.normalize_dflash_export import normalize_export
from specforge.algorithms.common.dflash_family_model import (
    OnlineDFlashModel,
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)
from specforge.modeling.draft.dflash import (
    DFlashDraftModel,
    dflash_attention_resume_contract,
    resolve_dflash_causal_block,
)
from specforge.modeling.draft.dspine import DSpineDraftModel
from tests.test_modeling.test_dflash_sliding import _draft_config


class DFlashAttentionDirectionTest(unittest.TestCase):

    def test_direction_and_range_masks_match_independent_reference(self):
        for window, causal in product([None, 1, 3], [None, False, True]):
            with self.subTest(window=window, causal=causal):
                anchors = torch.tensor([[2, 5], [0, 4]])
                keep = torch.tensor([[True, True], [True, False]])
                arguments = dict(
                    anchor_positions=anchors,
                    block_keep_mask=keep,
                    S=7,
                    block_size=5,
                    device="cpu",
                    sliding_window=window,
                    causal_block=causal,
                )
                dense = create_dflash_sdpa_mask(**arguments)
                sparse = create_dflash_block_mask(**arguments)
                for batch in range(2):
                    for query in range(10):
                        (block, offset) = divmod(query, 5)
                        anchor = int(anchors[batch, block])
                        expected = []
                        for key in range(17):
                            visible = False
                            if keep[batch, block]:
                                if key < 7:
                                    visible = key < anchor
                                    if window is not None:
                                        visible = (
                                            visible and anchor + offset - key < window
                                        )
                                elif (key - 7) // 5 == block:
                                    distance = offset - (key - 7) % 5
                                    visible = True
                                    if causal is True or (
                                        causal is None and window is not None
                                    ):
                                        visible = distance >= 0
                                    if causal is not None and window is not None:
                                        visible = visible and abs(distance) < window
                            expected.append(visible)
                        expected = torch.tensor(expected)
                        torch.testing.assert_close(dense[batch, 0, query], expected)
                        actual = sparse.mask_mod(
                            torch.tensor(batch),
                            torch.tensor(0),
                            torch.tensor(query),
                            torch.arange(17),
                        )
                        torch.testing.assert_close(actual, expected)

    def test_rejects_non_boolean_direction(self):
        for value in ["false", "true", 0, 1, [], {}]:
            with self.subTest(value=value):
                config = _draft_config(["full_attention"])
                config.is_causal = value
                with self.assertRaisesRegex(ValueError, "is_causal"):
                    DFlashDraftModel(config)

    def test_direction_survives_config_roundtrip_and_reaches_training(self):
        for layer_type, causal in product(
            ["full_attention", "sliding_attention"], [None, False, True]
        ):
            with self.subTest(layer_type=layer_type, causal=causal):
                config = _draft_config(
                    [layer_type], 3 if layer_type == "sliding_attention" else None
                )
                if causal is not None:
                    config.is_causal = causal
                config = Qwen3Config.from_dict(json.loads(config.to_json_string()))
                model = DFlashDraftModel(config)
                wrapper = OnlineDFlashModel(
                    draft_model=model,
                    target_lm_head=nn.Linear(8, 32),
                    target_embed_tokens=nn.Embedding(32, 8),
                    mask_token_id=31,
                    block_size=2,
                    attention_backend="sdpa",
                    num_anchors=1,
                )
                self.assertIs(resolve_dflash_causal_block(config), causal)
                self.assertIs(wrapper.causal_block, causal)
                self.assertIs(model.layers[0].self_attn.causal_block, causal)

    def test_native_inference_matches_training_mask(self):
        for cached, backend, window, causal in product(
            [False, True], ["eager", "sdpa"], [None, 2], [False, True]
        ):
            with self.subTest(
                cached=cached, backend=backend, window=window, causal=causal
            ):
                torch.manual_seed(17)
                config = _draft_config(
                    ["sliding_attention" if window else "full_attention"], window
                )
                config.block_size = 4
                config.is_causal = causal
                config._attn_implementation = backend
                model = DFlashDraftModel(config).eval()
                context = torch.randn(1, 5, model.fc.in_features)
                noise = torch.randn(1, 4, 8)
                positions = torch.arange(9).unsqueeze(0)
                mask = create_dflash_sdpa_mask(
                    anchor_positions=torch.tensor([[5]]),
                    block_keep_mask=torch.tensor([[True]]),
                    S=5,
                    block_size=4,
                    device="cpu",
                    sliding_window=window,
                    causal_block=causal,
                )
                with torch.no_grad():
                    expected = model(
                        position_ids=positions,
                        noise_embedding=noise,
                        target_hidden=context,
                        attention_mask=mask,
                    )
                    cache = DynamicCache() if cached else None
                    if cached:
                        model(
                            position_ids=torch.arange(8).unsqueeze(0),
                            noise_embedding=noise,
                            target_hidden=context[:, :4],
                            past_key_values=cache,
                            use_cache=True,
                        )
                        cache.crop(4)
                    actual = model(
                        position_ids=positions[:, 4:] if cached else positions,
                        noise_embedding=noise,
                        target_hidden=context[:, 4:] if cached else context,
                        past_key_values=cache,
                        use_cache=cached,
                        is_causal=False,
                    )
                torch.testing.assert_close(actual, expected, atol=1e-06, rtol=1e-05)

    def test_actual_forward_and_gradients_obey_direction_and_window(self):
        for backend, window, causal in product(
            ["sdpa", "flex_attention"], [None, 2], [False, True]
        ):
            with self.subTest(backend=backend, window=window, causal=causal):
                if backend == "flex_attention" and (not torch.cuda.is_available()):
                    self.skipTest("Flex Attention requires CUDA")
                device = "cuda" if backend == "flex_attention" else "cpu"
                torch.manual_seed(19)
                config = _draft_config(
                    ["sliding_attention" if window else "full_attention"], window
                )
                config.hidden_size = 32
                config.intermediate_size = 64
                config.head_dim = 16
                config.is_causal = causal
                config._attn_implementation = backend
                model = DFlashDraftModel(config).to(device)
                arguments = dict(
                    anchor_positions=torch.tensor([[3, 6]], device=device),
                    block_keep_mask=torch.tensor([[True, True]], device=device),
                    S=8,
                    block_size=4,
                    device=device,
                    sliding_window=window,
                    causal_block=causal,
                )
                builder = (
                    create_dflash_block_mask
                    if backend == "flex_attention"
                    else create_dflash_sdpa_mask
                )
                mask = builder(**arguments)
                context = torch.randn(
                    1, 8, model.fc.in_features, device=device, requires_grad=True
                )
                noise = torch.randn(1, 8, 32, device=device, requires_grad=True)
                positions = torch.tensor(
                    [[0, 1, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 6, 7, 8, 9]], device=device
                )
                output = model(
                    position_ids=positions,
                    noise_embedding=noise,
                    target_hidden=context,
                    attention_mask=mask,
                    kernel_options=(
                        {"BACKEND": "TRITON"} if backend == "flex_attention" else None
                    ),
                )
                output[0, 0, 0].backward()
                self.assertTrue(torch.isfinite(noise.grad).all())
                self.assertTrue(torch.isfinite(context.grad).all())
                self.assertFalse(noise.grad[0, 4:].any())
                self.assertFalse(context.grad[0, 3:].any())
                if causal:
                    self.assertFalse(noise.grad[0, 1:4].any())
                else:
                    self.assertGreater(noise.grad[0, 1].abs().sum(), 0)
                if window is not None:
                    self.assertFalse(noise.grad[0, 2:4].any())
                    self.assertFalse(context.grad[0, :2].any())

    def test_new_window_semantics_have_a_distinct_resume_contract(self):
        config = _draft_config(["sliding_attention"], 3)
        model = DFlashDraftModel(config)
        self.assertEqual(dflash_attention_resume_contract(model), {})
        config.is_causal = True
        causal = dflash_attention_resume_contract(model)
        config.is_causal = False
        bidirectional = dflash_attention_resume_contract(model)
        self.assertTrue(causal and bidirectional and (causal != bidirectional))
        config.is_causal = True
        config.architectures = ["DSparkDraftModel"]
        model.sliding_window = None
        self.assertEqual(dflash_attention_resume_contract(model), {})

    def test_checkpoint_versioning_preserves_legacy_cases(self):
        for window, causal, architecture in product(
            [None, 3],
            [None, False, True],
            ["DFlashDraftModel", "DSparkDraftModel", "DSpineDraftModel"],
        ):
            with self.subTest(window=window, causal=causal, architecture=architecture):
                layer_type = "sliding_attention" if window else "full_attention"
                draft = SimpleNamespace(
                    config=SimpleNamespace(
                        architectures=[architecture], is_causal=causal
                    ),
                    layer_types=(layer_type,),
                    sliding_window=window,
                )
                needs_version = causal is not None and (
                    window is not None
                    or (causal and architecture == "DFlashDraftModel")
                )
                expected = {}
                if needs_version:
                    expected["draft_attention_mask_semantics"] = (
                        2,
                        (layer_type,),
                        window,
                        causal,
                    )
                self.assertEqual(dflash_attention_resume_contract(draft), expected)

    def test_dspark_resume_keeps_legacy_causal_field_presence(self):
        for causal in [None, False, True]:
            with self.subTest(causal=causal):
                from specforge.algorithms.dspark.providers import resume_contract
                from tests.test_modeling.test_dspark_causal import build_model

                model = build_model(causal=causal)
                contract = resume_contract(None, model.draft_model, model)
                self.assertIs("dspark_causal_block" in contract, causal is True)
                if causal:
                    self.assertIs(contract["dspark_causal_block"], True)
                self.assertNotIn("draft_attention_mask_semantics", contract)

    def test_dspine_rejects_bidirectional_attention(self):
        config = _draft_config(["full_attention"])
        config.is_causal = False
        with self.assertRaisesRegex(ValueError, "DSpine requires causal"):
            DSpineDraftModel(config)

    def test_export_preserves_direction_and_rejects_unvalidated_serving(self):
        for causal in [False, True]:
            with self.subTest(causal=causal):
                with TemporaryDirectory() as directory:
                    tmp_path = Path(directory)
                    config = _draft_config(["full_attention"])
                    config.is_causal = causal
                    path = tmp_path / "config.json"
                    path.write_text(config.to_json_string())
                    self.assertIs(normalize_export(str(path), 2)["is_causal"], causal)
                    config.layer_types = ["sliding_attention"]
                    config.sliding_window = 3
                    path.write_text(config.to_json_string())
                    if causal:
                        self.assertIs(normalize_export(str(path), 2)["is_causal"], True)
                    else:
                        before = path.read_bytes()
                        with self.assertRaisesRegex(
                            ValueError, "Bidirectional sliding-window"
                        ):
                            normalize_export(str(path), 2)
                        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
