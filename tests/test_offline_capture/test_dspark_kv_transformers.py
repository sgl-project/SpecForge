import unittest

import torch
from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

from examples.prepare_dspark_kv import capture_sample
from specforge.modeling.draft.target_kv import inverse_target_kv_rope


class QwenTargetKVCaptureTest(unittest.TestCase):
    def test_real_transformers_cache_matches_post_norm_pre_rope_reference(self):
        torch.manual_seed(42)
        cfg = Qwen3_5TextConfig(
            hidden_size=16,
            intermediate_size=32,
            vocab_size=32,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            layer_types=["full_attention", "linear_attention", "full_attention"],
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 1e7,
                "partial_rotary_factor": 0.5,
                "mrope_section": [1, 0, 0],
            },
        )
        cfg._attn_implementation = "sdpa"
        model = Qwen3_5TextModel(cfg).eval().requires_grad_(False)
        seen = {}
        handles = [
            model.layers[i].self_attn.k_norm.register_forward_hook(
                lambda module, args, output, i=i: seen.update(
                    {i: output.detach().clone()}
                )
            )
            for i in (0, 2)
        ]
        for i in (0, 2):
            handles.append(
                model.layers[i].self_attn.v_proj.register_forward_hook(
                    lambda module, args, output, i=i: seen.update(
                        {f"v{i}": output.detach().clone()}
                    )
                )
            )
        handles.append(
            model.norm.register_forward_hook(
                lambda module, args, output: seen.update(
                    {"last": output.detach().clone()}
                )
            )
        )
        ids = torch.tensor([1, 3, 5, 7, 2, 11])
        try:
            captured = capture_sample(model, ids, torch.ones(6), (0, 2))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(captured["target_kv"].shape, (6, 2, 2, 2, 4))
        restored = inverse_target_kv_rope(
            captured["target_kv"].unsqueeze(0),
            torch.arange(6)[None],
            rope_theta=1e7,
            rotary_dim=2,
        )
        for index, layer in enumerate((0, 2)):
            torch.testing.assert_close(
                restored[:, :, index, 0], seen[layer], atol=2e-6, rtol=2e-6
            )
            torch.testing.assert_close(
                captured["target_kv"][:, index, 1],
                seen[f"v{layer}"].view(6, 2, 4),
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            captured["target_last_hidden_states"], seen["last"][0], rtol=0, atol=0
        )
        self.assertTrue(all(not value.requires_grad for value in captured.values()))
        again = capture_sample(model, ids, torch.ones(6), (0, 2))
        for key in captured:
            torch.testing.assert_close(again[key], captured[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
