"""Public JSON layout, component boundaries, and legacy checkpoint compatibility."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.config import AutoDraftConfig, DraftConfigClass
from specforge.modeling.draft.config.dflash import DFlashConfig
from specforge.modeling.draft.config.dspark import DSparkConfig
from tests.test_modeling.test_all_draft_configs import architecture_payloads


class DraftLayoutTest(unittest.TestCase):
    def test_every_existing_architecture_has_one_canonical_round_trip(self):
        self.assertIs(DraftConfigClass, AutoDraftConfig)
        for name, payload in architecture_payloads().items():
            with self.subTest(architecture=name):
                original = copy.deepcopy(payload)
                resolved = AutoDraftConfig.finalize(AutoDraftConfig.from_dict(payload))
                public = AutoDraftConfig.to_dict(resolved)
                self.assertIn("spec_config", public)
                self.assertIn("attention_config", public)
                self.assertEqual(public["residual_config"], {"type": "standard"})
                self.assertEqual(public["ffn_config"], {"type": "dense"})
                self.assertNotIn("block_size", public)
                self.assertNotIn("dflash_config", public)
                self.assertNotIn("eagle_config", public)
                self.assertNotIn("mtp_config", public)
                self.assertNotIn("dspine_config", public)
                restored = AutoDraftConfig.from_dict(public)
                self.assertEqual(AutoDraftConfig.to_dict(restored), public)
                self.assertEqual(payload, original)

    def test_save_pretrained_writes_public_config_and_preserves_weights(self):
        for name, payload in architecture_payloads().items():
            with (
                self.subTest(architecture=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                model = AutoDraftModel.from_config(AutoDraftConfig.from_dict(payload))
                model.save_pretrained(directory)
                saved = json.loads((Path(directory) / "config.json").read_text())
                self.assertIn("spec_config", saved)
                self.assertNotIn("dflash_config", saved)
                restored = AutoDraftModel.from_pretrained(
                    directory, local_files_only=True
                )
                self.assertIs(type(restored), type(model))
                torch.testing.assert_close(
                    model.state_dict(), restored.state_dict(), rtol=0, atol=0
                )

    def test_conflicts_and_unknown_algorithm_fields_are_rejected(self):
        payload = architecture_payloads()["DSparkDraftModel"]
        public = AutoDraftConfig.to_dict(AutoDraftConfig.from_dict(payload))
        cases = [
            ({"block_size": public["spec_config"]["block_size"] + 1}, "Conflicting"),
            ({"dflash_config": {"block_size": 99}}, "Conflicting"),
            (
                {"spec_config": {**public["spec_config"], "algorithm": "dflash"}},
                "algorithm",
            ),
            (
                {"spec_config": {**public["spec_config"], "schema_version": True}},
                "schema_version",
            ),
            (
                {"spec_config": {**public["spec_config"], "selector_rank": 5}},
                "Unknown spec_config",
            ),
            (
                {
                    "attention_config": {"type": "mla"},
                    "dflash_config": {"attention_mode": "gqa"},
                },
                "Conflicting",
            ),
        ]
        for override, message in cases:
            with (
                self.subTest(override=override),
                self.assertRaisesRegex(ValueError, message),
            ):
                AutoDraftConfig.from_dict({**public, **override})

    def test_mla_dimensions_have_one_serialized_source(self):
        payload = architecture_payloads()["DFlashDraftModel"]
        payload["dflash_config"]["attention_mode"] = "mla"
        payload.update(
            kv_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4
        )
        public = AutoDraftConfig.to_dict(AutoDraftConfig.from_dict(payload))
        self.assertEqual(public["attention_config"]["type"], "mla")
        self.assertEqual(public["attention_config"]["kv_lora_rank"], 8)
        self.assertNotIn("kv_lora_rank", public)
        model = AutoDraftModel.from_config(AutoDraftConfig.from_dict(public))
        self.assertEqual(model.layers[0].self_attn.kv_lora_rank, 8)

    def test_configuration_inheritance_does_not_change_block_semantics(self):
        self.assertTrue(issubclass(DSparkConfig, DFlashConfig))
        for architecture in ("DFlashDraftModel", "DSparkDraftModel"):
            config = AutoDraftConfig.from_dict(architecture_payloads()[architecture])
            public = AutoDraftConfig.to_dict(config)
            self.assertEqual(public["spec_config"]["block_size"], 4)
            expected = "dspark" if architecture == "DSparkDraftModel" else "dflash"
            self.assertEqual(public["spec_config"]["algorithm"], expected)

    def test_direct_hf_configs_use_the_same_serialization(self):
        from transformers import LlamaConfig, Qwen3Config

        from specforge.modeling.draft.registry import DRAFT_REGISTRY

        for name, payload in architecture_payloads().items():
            with self.subTest(architecture=name):
                base = LlamaConfig if "Eagle" in name else Qwen3Config
                # Both legacy and public layouts work without the Auto loader.
                public = AutoDraftConfig.to_dict(AutoDraftConfig.from_dict(payload))
                for raw in (payload, public):
                    config = base.from_dict(raw)
                    config._attn_implementation = "eager"
                    model = DRAFT_REGISTRY[name](config)
                    saved = json.loads(model.config.to_json_string())
                    self.assertEqual(saved["spec_config"], public["spec_config"])
                    self.assertEqual(model.config._attn_implementation, "eager")

    def test_causal_only_architectures_validate_direct_configs(self):
        from transformers import LlamaConfig

        from specforge.modeling.draft.models.eagle3 import LlamaForCausalLMEagle3

        payload = architecture_payloads()["LlamaForCausalLMEagle3"]
        for value in (False, None, 1, "true"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "is_causal"),
            ):
                LlamaForCausalLMEagle3(
                    LlamaConfig.from_dict(
                        {
                            **payload,
                            "attention_config": {"type": "gqa", "is_causal": value},
                        }
                    )
                )

    def test_serving_export_preserves_both_config_contracts(self):
        from specforge.export.checkpoint_io import write_serving_config

        for name, payload in architecture_payloads().items():
            with (
                self.subTest(architecture=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                config = AutoDraftConfig.from_dict(payload)
                model = AutoDraftModel.from_config(config)
                before = json.loads(json.dumps(AutoDraftConfig.to_dict(model.config)))
                write_serving_config(model, directory)
                legacy = json.loads((Path(directory) / "config.json").read_text())
                public = json.loads(
                    (Path(directory) / "specforge_config.json").read_text()
                )
                self.assertEqual(public, before)
                self.assertNotIn("spec_config", legacy)
                self.assertEqual(
                    json.loads(
                        json.dumps(
                            AutoDraftConfig.to_dict(AutoDraftConfig.from_dict(legacy))
                        )
                    ),
                    public,
                )
                self.assertEqual(
                    json.loads(json.dumps(AutoDraftConfig.to_dict(model.config))),
                    before,
                )


if __name__ == "__main__":
    unittest.main()
