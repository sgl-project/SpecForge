"""Native MTP adaptation is independent of config parsing and model execution."""

import unittest

import torch

from specforge.export.mtp import _patch_text_config
from specforge.modeling.draft.models.specialized.qwen3_5_mtp.adapter import (
    Qwen3_5MTPAdapter,
)


class NativeDraftAdapterTest(unittest.TestCase):
    def test_native_keys_respect_shared_weights(self):
        state = {
            key: torch.zeros(1)
            for key in ["embed_tokens.weight", "mtp.fc.weight", "mtp.lm_head.weight"]
        }
        self.assertEqual(
            set(Qwen3_5MTPAdapter.native_state_dict(state)),
            {"mtp.fc.weight", "mtp.lm_head.weight"},
        )
        for shared in (True, False):
            with self.subTest(shared=shared):
                required = Qwen3_5MTPAdapter.required_keys(state, share_lm_head=shared)
                self.assertIn("mtp.fc.weight", required)
                self.assertEqual("mtp.lm_head.weight" in required, not shared)
                extra = Qwen3_5MTPAdapter.allowed_extra_keys(share_lm_head=shared)
                self.assertIn("mtp.embed_tokens.weight", extra)
                self.assertEqual("mtp.lm_head.weight" in extra, shared)

    def test_config_merge_preserves_target_depth_and_unrelated_fields(self):
        for nested in (True, False):
            with self.subTest(nested=nested):
                target = {"head_dim": 8, "num_hidden_layers": 40, "custom": True}
                base = (
                    {"text_config": target, "vision_config": {"x": 1}}
                    if nested
                    else target
                )
                draft = {
                    "architectures": ["Qwen3_5MTPDraftModel"],
                    "head_dim": 4,
                    "num_hidden_layers": 1,
                }
                result = _patch_text_config(base, draft)
                self.assertIs(result, base)
                self.assertEqual(
                    target, {"head_dim": 4, "num_hidden_layers": 40, "custom": True}
                )
                if nested:
                    self.assertEqual(result["vision_config"], {"x": 1})

    def test_legacy_exports_default_to_qwen_mtp(self):
        self.assertEqual(
            _patch_text_config({"head_dim": 8}, {"head_dim": 4}), {"head_dim": 4}
        )

    def test_unknown_and_ambiguous_architectures_are_rejected(self):
        for architectures, message in [
            (["DSparkDraftModel"], "No native checkpoint adapter"),
            (["unknown"], "No native checkpoint adapter"),
            (["Qwen3_5MTPDraftModel", "DSparkDraftModel"], "exactly one"),
        ]:
            with self.subTest(architectures=architectures):
                with self.assertRaisesRegex(ValueError, message):
                    _patch_text_config({}, {"architectures": architectures})
