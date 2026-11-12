"""Dense stage-1 checkpoints and config-driven inference exports."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file

from specforge.export.eagle4 import add_eagle4_svd_factors
from specforge.export.to_hf import export_to_hf
from specforge.export.to_sglang import export_to_sglang
from specforge.modeling.draft.eagle4 import LlamaForCausalLMEagle4
from specforge.modeling.draft.eagle4_structure import efficient_head_export_config
from tests.test_modeling.test_draft.test_eagle4 import _config


class Eagle4ExportTest(unittest.TestCase):
    def config(self, rank=4):
        return _config(
            efficient_lm_head_config={
                "type": "low_rank",
                "rank": rank,
                "num_candidates": 4,
            }
        )

    def test_export_preserves_dense_weights_and_emits_svd_for_both_formats(self):
        torch.manual_seed(7)
        config = self.config()
        model = LlamaForCausalLMEagle4(config)
        weights = model.state_dict()
        self.assertFalse(any("efficient_lm_head" in k for k in weights))
        u, singular, vh = torch.linalg.svd(
            weights["lm_head.weight"], full_matrices=False
        )
        expected = (u[:, :4] * singular[:4]) @ vh[:4]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config.to_json_file(str(root / "draft.json"))
            torch.save(
                {"strategy": "eagle4", "draft_state_dict": weights},
                root / "training_state.pt",
            )
            for name, exporter in (("hf", export_to_hf), ("sglang", export_to_sglang)):
                with self.subTest(format=name):
                    dest = root / name
                    exporter(str(root), str(root / "draft.json"), str(dest))
                    saved = load_file(str(dest / "model.safetensors"))
                    for key, value in weights.items():
                        if name == "sglang" and "embed" in key:
                            continue
                        torch.testing.assert_close(saved[key], value, rtol=0, atol=0)
                    a = saved["efficient_lm_head_down_proj.weight"]
                    b = saved["efficient_lm_head.weight"]
                    self.assertEqual(tuple(a.shape), (4, 8))
                    self.assertEqual(tuple(b.shape), (16, 4))
                    torch.testing.assert_close(b @ a, expected)
                    exported_config = json.loads((dest / "config.json").read_text())
                    self.assertEqual(
                        exported_config["efficient_lm_head_config"],
                        config.efficient_lm_head_config,
                    )
            original = torch.load(root / "training_state.pt", weights_only=True)
            self.assertFalse(
                any("efficient_lm_head" in k for k in original["draft_state_dict"])
            )

    def test_full_rank_reconstructs_and_bfloat16_precision_is_preserved(self):
        config = self.config(rank=8)
        w = torch.randn(16, 8).bfloat16()
        result = add_eagle4_svd_factors(config, {"lm_head.weight": w})
        self.assertIs(result["lm_head.weight"], w)
        a = result["efficient_lm_head_down_proj.weight"]
        b = result["efficient_lm_head.weight"]
        self.assertEqual(a.dtype, w.dtype)
        self.assertEqual(b.dtype, w.dtype)
        torch.testing.assert_close(
            b.float() @ a.float(), w.float(), atol=0.03, rtol=0.03
        )

    def test_dense_config_is_unchanged_and_invalid_export_metadata_fails(self):
        state = {"lm_head.weight": torch.randn(16, 8)}
        self.assertIs(add_eagle4_svd_factors(_config(), state), state)
        for head in (
            {},
            {"type": "weighted_low_rank"},
            {"type": "low_rank", "rank": 9, "num_candidates": 4},
            {"type": "low_rank", "rank": 4, "num_candidates": 17},
            {"type": "low_rank", "rank": True, "num_candidates": 4},
        ):
            with self.subTest(head=head), self.assertRaises(ValueError):
                efficient_head_export_config(_config(efficient_lm_head_config=head))

    def test_export_rejects_nonfinite_or_preexisting_factors(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            add_eagle4_svd_factors(
                self.config(), {"lm_head.weight": torch.full((16, 8), float("nan"))}
            )
        with self.assertRaisesRegex(ValueError, "without factors"):
            add_eagle4_svd_factors(
                self.config(), {"efficient_lm_head.weight": torch.zeros(16, 4)}
            )


if __name__ == "__main__":
    unittest.main()
