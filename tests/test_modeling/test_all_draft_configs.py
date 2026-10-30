"""Every built-in architecture participates in the resolved-config contract."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.config import Config
from specforge.config.draft import AutoDraftConfig
from specforge.config.draft.base import DraftConfig
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.registry import DRAFT_REGISTRY
from specforge.training.model_loading import resolve_draft_config
from tests.test_modeling.test_dflash_config import VARIANTS, payload


def architecture_payloads():
    result = {name: payload(name, **method) for name, _, method in VARIANTS}
    for name in ("LlamaForCausalLMEagle3", "PEagleDraftModel", "Qwen3_5MTPDraftModel"):
        result[name] = {
            "architectures": [name],
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "max_position_embeddings": 32,
        }
    result["LlamaForCausalLMEagle3"].update(fc_norm=True, norm_output=True)
    result["PEagleDraftModel"].update(num_hidden_layers=2, norm_before_residual=True)
    result["Qwen3_5MTPDraftModel"].update(
        attn_output_gate=True,
        partial_rotary_factor=0.5,
        mtp_config={"share_lm_head": False},
    )
    result["DSpineDraftModel"] = {
        **payload("DSpineDraftModel"),
        "dspine_config": {"transfer_rank": 4, "message_size": 8, "top_k": 4},
    }
    return result


class AllDraftConfigsTest(unittest.TestCase):
    def test_dspine_settings_are_validated_before_allocation(self):
        data = architecture_payloads()["DSpineDraftModel"]
        for settings, message in [
            ({"transfer_rank": 65}, "transfer_rank"),
            ({"top_k": 0}, "top_k"),
        ]:
            with self.subTest(settings=settings):
                config = AutoDraftConfig.from_dict(
                    {**data, "dspine_config": {**data["dspine_config"], **settings}}
                )
                with mock.patch(
                    "torch.nn.Linear", side_effect=AssertionError("allocation")
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        AutoDraftConfig.finalize(config)
        data["is_causal"] = False
        with self.assertRaisesRegex(ValueError, "causal"):
            AutoDraftConfig.finalize(AutoDraftConfig.from_dict(data))

    def test_every_algorithm_resolves_its_compatible_architectures(self):
        cases = architecture_payloads()
        for name, data in cases.items():
            if "Eagle" in name:
                data["eagle_config"] = {"eagle_aux_hidden_state_layer_ids": [1, 2, 3]}
                data["mask_token_id"] = 3
            elif "dflash_config" in data:
                data["dflash_config"]["mask_token_id"] = 3
        for registration in builtin_algorithm_registry():
            for name in registration.spec.draft.compatible_architectures:
                with (
                    self.subTest(algorithm=registration.name, architecture=name),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    path = Path(directory) / "config.json"
                    path.write_text(json.dumps(cases[name]))
                    cfg = Config.model_validate(
                        {
                            "model": {
                                "target_model_path": "unused",
                                "draft_model_config": cases[name],
                            },
                            "data": {"hidden_states_path": "/unused"},
                            "training": {"strategy": registration.name},
                        }
                    )
                    resolved = resolve_draft_config(
                        cfg, provider=registration.providers.model.draft_config
                    )
                    self.assertEqual(
                        resolved.to_dict(),
                        AutoDraftConfig.from_inline(
                            {
                                **cases[name],
                                "_attn_implementation": cfg.training.attention_backend,
                            }
                        ).to_dict(),
                    )

    def test_registry_and_every_algorithm_have_config_coverage(self):
        cases = architecture_payloads()
        self.assertEqual(set(cases), set(DRAFT_REGISTRY))
        for registration in builtin_algorithm_registry():
            self.assertLessEqual(
                set(registration.spec.draft.compatible_architectures), set(cases)
            )
        for name, model in DRAFT_REGISTRY.items():
            with self.subTest(architecture=name):
                self.assertTrue(issubclass(model.draft_options_class, DraftConfig))
                with mock.patch(
                    "torch.nn.Linear", side_effect=AssertionError("allocation")
                ):
                    config = AutoDraftConfig.finalize(
                        AutoDraftConfig.from_dict(cases[name])
                    )
                self.assertEqual(config.architectures, [name])

    def test_every_architecture_round_trips_without_mutating_input(self):
        for name, data in architecture_payloads().items():
            with (
                self.subTest(architecture=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                data["future_metadata"] = {"nested": [1]}
                config = AutoDraftConfig.from_dict(data)
                original = copy.deepcopy(config.to_dict())
                resolved = AutoDraftConfig.finalize(config)
                self.assertEqual(config.to_dict(), original)
                self.assertEqual(
                    AutoDraftConfig.finalize(resolved).to_dict(), resolved.to_dict()
                )
                model = AutoDraftModel.from_config(config)
                self.assertEqual(config.to_dict(), original)
                self.assertEqual(
                    model.draft_options_class.from_model_config(model.config),
                    model.draft_options_class.from_model_config(resolved),
                )
                live_config = model.config
                before_save = copy.deepcopy(live_config.to_dict())
                self.assertFalse(hasattr(model, "draft_options"))
                for name in (
                    "block_size",
                    "mask_token_id",
                    "target_layer_ids",
                    "shift_label",
                    "layer_types",
                    "sliding_window",
                    "norm_output",
                    "norm_before_residual",
                ):
                    self.assertNotIn(name, vars(model))
                model.save_pretrained(directory)
                self.assertIs(model.config, live_config)
                self.assertEqual(model.config.to_dict(), before_save)
                if hasattr(model.config, "dflash_config"):
                    self.assertNotIn("block_size", model.config.dflash_config)
                    saved = json.loads((Path(directory) / "config.json").read_text())
                    self.assertEqual(
                        saved["block_size"], saved["dflash_config"]["block_size"]
                    )
                restored = AutoDraftModel.from_pretrained(
                    directory, local_files_only=True
                )
                self.assertEqual(
                    restored.draft_options_class.from_model_config(restored.config),
                    model.draft_options_class.from_model_config(model.config),
                )
                self.assertEqual(
                    restored.config.to_dict()["future_metadata"], {"nested": [1]}
                )
                torch.testing.assert_close(
                    model.state_dict(), restored.state_dict(), rtol=0, atol=0
                )

    def test_failed_save_cannot_mutate_live_config(self):
        from transformers import PreTrainedModel

        for name, data in architecture_payloads().items():
            with (
                self.subTest(architecture=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                model = AutoDraftModel.from_config(AutoDraftConfig.from_dict(data))
                live_config = model.config
                before = copy.deepcopy(live_config.to_dict())

                def fail_save(export_model, *args, **kwargs):
                    self.assertIsNot(export_model.config, live_config)
                    self.assertIs(
                        next(export_model.parameters()), next(model.parameters())
                    )
                    export_model.config.architectures = ["ServingOnly"]
                    if hasattr(export_model.config, "dflash_config"):
                        export_model.config.dflash_config["mask_token_id"] = 0
                    raise OSError("serialization failed")

                with mock.patch.object(PreTrainedModel, "save_pretrained", fail_save):
                    with self.assertRaisesRegex(OSError, "serialization failed"):
                        model.save_pretrained(directory)
                self.assertIs(model.config, live_config)
                self.assertEqual(live_config.to_dict(), before)

    def test_training_wrappers_read_draft_config(self):
        from specforge.algorithms.common.dflash_family_model import (
            OnlineDFlashModel,
            OnlineDominoModel,
            OnlineDSparkModel,
        )
        from specforge.algorithms.dspine.model import OnlineDSpineModel

        wrappers = {
            "DFlashDraftModel": OnlineDFlashModel,
            "DFlash2DraftModel": OnlineDFlashModel,
            "DominoDraftModel": OnlineDominoModel,
            "DSparkDraftModel": OnlineDSparkModel,
            "DSpineDraftModel": OnlineDSpineModel,
        }
        for name, wrapper_class in wrappers.items():
            with self.subTest(architecture=name):
                config = AutoDraftConfig.from_dict(architecture_payloads()[name])
                config.dflash_config["mask_token_id"] = 3
                draft = AutoDraftModel.from_config(config)
                wrapper = wrapper_class(
                    draft_model=draft,
                    target_lm_head=torch.nn.Linear(
                        config.hidden_size, config.vocab_size, bias=False
                    ),
                    target_embed_tokens=torch.nn.Embedding(
                        config.vocab_size, config.hidden_size
                    ),
                    attention_backend="sdpa",
                )
                for field in (
                    "block_size",
                    "mask_token_id",
                    "shift_label",
                    "dspine_config",
                ):
                    self.assertNotIn(field, vars(wrapper))
                if name == "DSpineDraftModel":
                    self.assertIs(draft.injection.config, draft.config.dspine_config)
                anchors = torch.tensor([[0, 1]])
                keep = torch.tensor([[True, False]])
                tokens = torch.tensor([[1, 2, 4, 6]])
                result = wrapper._create_noise_embed(tokens, anchors, keep)
                expected_ids = torch.full((1, 2 * draft.config.block_size), 3)
                expected_ids[0, 0] = 1
                torch.testing.assert_close(result, wrapper.embed_tokens(expected_ids))
                # Construction resolves a private config, not the caller's input.
                self.assertEqual(config.dflash_config["mask_token_id"], 3)

    def test_invalid_settings_fail_before_any_parameter_allocation(self):
        cases = (
            (
                "LlamaForCausalLMEagle3",
                {"target_hidden_size": -1},
                "target_hidden_size",
            ),
            ("LlamaForCausalLMEagle3", {"fc_norm": "false"}, "fc_norm"),
            ("LlamaForCausalLMEagle3", {"norm_output": 1}, "norm_output"),
            ("LlamaForCausalLMEagle3", {"num_hidden_layers": 2}, "num_hidden_layers"),
            (
                "PEagleDraftModel",
                {"norm_before_residual": "false"},
                "norm_before_residual",
            ),
            (
                "PEagleDraftModel",
                {"eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 2]}},
                "layer_ids",
            ),
            ("PEagleDraftModel", {"num_key_value_heads": 3}, "num_key_value_heads"),
            ("Qwen3_5MTPDraftModel", {"num_hidden_layers": 2}, "num_hidden_layers"),
            ("Qwen3_5MTPDraftModel", {"attn_output_gate": "false"}, "attn_output_gate"),
            ("Qwen3_5MTPDraftModel", {"partial_rotary_factor": 0.25}, "rotary"),
            ("Qwen3_5MTPDraftModel", {"partial_rotary_factor": float("nan")}, "rotary"),
            (
                "Qwen3_5MTPDraftModel",
                {"mtp_config": {"share_lm_head": "false"}},
                "share_lm_head",
            ),
        )
        for name, overrides, message in cases:
            with self.subTest(architecture=name, overrides=overrides):
                config = AutoDraftConfig.from_dict(architecture_payloads()[name])
                for key, value in overrides.items():
                    setattr(config, key, value)
                with (
                    mock.patch(
                        "torch.nn.Linear", side_effect=AssertionError("allocation")
                    ),
                    mock.patch(
                        "torch.nn.Embedding", side_effect=AssertionError("allocation")
                    ),
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        AutoDraftConfig.finalize(config)
                    with self.assertRaisesRegex(ValueError, message):
                        AutoDraftModel.from_config(config)

    def test_eagle_supports_independent_projection_width(self):
        data = architecture_payloads()["LlamaForCausalLMEagle3"]
        data.update(hidden_size=20, num_attention_heads=3, num_key_value_heads=1)
        config = AutoDraftConfig.finalize(AutoDraftConfig.from_dict(data))
        model = AutoDraftModel.from_config(config)
        self.assertEqual(model.midlayer.self_attn.q_proj.weight.shape, (12, 40))
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = AutoDraftModel.from_pretrained(directory)
            torch.testing.assert_close(
                model.state_dict(), restored.state_dict(), rtol=0, atol=0
            )

    def test_nested_settings_preserve_unknown_metadata(self):
        for name, method in (
            ("PEagleDraftModel", "eagle_config"),
            ("Qwen3_5MTPDraftModel", "mtp_config"),
        ):
            with self.subTest(architecture=name):
                data = architecture_payloads()[name]
                data.setdefault(method, {})["future_field"] = {"value": [1]}
                config = AutoDraftConfig.finalize(AutoDraftConfig.from_dict(data))
                self.assertEqual(
                    getattr(config, method)["future_field"], {"value": [1]}
                )

    def test_peagle_reads_normalized_config_without_constructor_overrides(self):
        config = AutoDraftConfig.from_dict(architecture_payloads()["PEagleDraftModel"])
        config.norm_before_residual = False
        original = copy.deepcopy(config.to_dict())
        model = AutoDraftModel.from_config(config)
        self.assertFalse(model.config.norm_before_residual)
        self.assertIs(model.layers[0].config, model.config)
        self.assertEqual(config.to_dict(), original)
        with self.assertRaises(TypeError):
            AutoDraftModel.from_config(config, norm_before_residual=True)

    def test_inline_eagle_settings_validate_without_task_overrides(self):
        from tests.draft_config_utils import draft_settings

        for strategy in ("eagle3", "peagle"):
            data = draft_settings(strategy)
            data["norm_before_residual"] = False
            cfg = Config.model_validate(
                {
                    "model": {
                        "target_model_path": "unused",
                        "draft_model_config": data,
                    },
                    "data": {"hidden_states_path": "/unused"},
                    "training": {"strategy": strategy},
                }
            )
            provider = (
                builtin_algorithm_registry()
                .resolve(strategy)
                .providers.model.draft_config
            )
            resolved = resolve_draft_config(cfg, provider=provider)
            self.assertEqual(
                resolved.eagle_config["eagle_aux_hidden_state_layer_ids"], (1, 5, 8)
            )
            cfg.model.draft_model_config["eagle_config"][
                "eagle_aux_hidden_state_layer_ids"
            ] = [1]
            with self.assertRaisesRegex(ValueError, "layer_ids"):
                resolve_draft_config(cfg, provider=provider)

    def test_all_checked_in_draft_configs_finalize(self):
        directory = Path(__file__).resolve().parents[2] / "configs"
        for path in directory.glob("*.json"):
            data = json.loads(path.read_text())
            if data.get("architectures", [None])[0] not in DRAFT_REGISTRY:
                continue
            with self.subTest(config=path.name):
                resolved = AutoDraftConfig.finalize(AutoDraftConfig.from_dict(data))
                self.assertEqual(
                    AutoDraftConfig.finalize(resolved).to_dict(), resolved.to_dict()
                )


if __name__ == "__main__":
    unittest.main()
