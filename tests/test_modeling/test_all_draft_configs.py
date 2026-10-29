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
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.config import AutoDraftConfig
from specforge.modeling.draft.config.base import DraftConfig
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
    return result


class AllDraftConfigsTest(unittest.TestCase):
    def test_every_algorithm_resolves_its_compatible_architectures(self):
        cases = architecture_payloads()
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
                                "draft_model_config": str(path),
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
                        AutoDraftConfig.finalize(
                            AutoDraftConfig.from_dict(cases[name])
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
                    model.draft_options,
                    model.draft_options_class.from_model_config(resolved),
                )
                model.save_pretrained(directory)
                restored = AutoDraftModel.from_pretrained(
                    directory, local_files_only=True
                )
                self.assertEqual(restored.draft_options, model.draft_options)
                self.assertEqual(restored.config.future_metadata, {"nested": [1]})
                torch.testing.assert_close(
                    model.state_dict(), restored.state_dict(), rtol=0, atol=0
                )

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

    def test_peagle_constructor_override_is_validated_without_mutation(self):
        config = AutoDraftConfig.from_dict(architecture_payloads()["PEagleDraftModel"])
        config.norm_before_residual = "invalid"
        original = copy.deepcopy(config.to_dict())
        model = AutoDraftModel.from_config(config, norm_before_residual=False)
        self.assertFalse(model.draft_options.norm_before_residual)
        self.assertFalse(model.layers[0].norm_before_residual)
        self.assertEqual(config.to_dict(), original)
        with self.assertRaisesRegex(ValueError, "norm_before_residual"):
            AutoDraftModel.from_config(config, norm_before_residual="false")

    def test_run_overrides_precede_eagle_family_validation(self):
        for strategy, name in (
            ("eagle3", "LlamaForCausalLMEagle3"),
            ("peagle", "PEagleDraftModel"),
        ):
            with (
                self.subTest(strategy=strategy),
                tempfile.TemporaryDirectory() as directory,
            ):
                data = architecture_payloads()[name]
                data["eagle_config"] = {"eagle_aux_hidden_state_layer_ids": [1]}
                if strategy == "peagle":
                    data["norm_before_residual"] = "invalid"
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(data))
                cfg = Config.model_validate(
                    {
                        "model": {
                            "target_model_path": "unused",
                            "draft_model_config": str(path),
                            "aux_hidden_state_layer_ids": [1, 2, 3],
                        },
                        "data": {"hidden_states_path": "/unused"},
                        "training": {
                            "strategy": strategy,
                            "norm_before_residual": False,
                        },
                    }
                )
                provider = (
                    builtin_algorithm_registry()
                    .resolve(strategy)
                    .providers.model.draft_config
                )
                resolved = resolve_draft_config(cfg, provider=provider)
                self.assertEqual(
                    resolved.eagle_config["eagle_aux_hidden_state_layer_ids"], [1, 2, 3]
                )
                if strategy == "peagle":
                    self.assertFalse(resolved.norm_before_residual)
                cfg.model.aux_hidden_state_layer_ids = None
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
