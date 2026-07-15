from __future__ import annotations

import subprocess
import sys
import unittest
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.algorithms.common.providers import (
    MODEL_PROVENANCE_CONTRACT_KEY,
    OMITTED_STATE_FINGERPRINT_CONTRACT_KEY,
    STEP_OPTIONS_CONTRACT_KEY,
    AlgorithmProviders,
    DraftConfigProvider,
    ServerCaptureLayout,
    ServerStreamingProvider,
    StepRuntimeConfig,
)
from specforge.algorithms.contracts import AlgorithmSpec, FeatureMode

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILTINS = ("dflash", "domino", "dspark", "eagle3", "peagle")


class BuiltinProviderContractTest(unittest.TestCase):
    def setUp(self):
        self.registry = builtin_algorithm_registry()

    def test_five_builtins_are_explicit_and_instance_owned(self):
        self.assertEqual(BUILTINS, self.registry.names)
        self.assertIsNot(builtin_algorithm_registry(), self.registry)

    def test_every_registration_pairs_contract_and_providers(self):
        for registration in self.registry:
            with self.subTest(algorithm=registration.name):
                self.assertIsInstance(registration.spec, AlgorithmSpec)
                self.assertIsInstance(registration.providers, AlgorithmProviders)
                self.assertEqual(
                    registration.name,
                    registration.providers.algorithm_name,
                )
                contract_keys = {c.key for c in registration.spec.feature_contracts}
                provider_keys = {
                    *(
                        (FeatureMode.OFFLINE, p.modality)
                        for p in registration.providers.offline
                    ),
                    *(
                        (FeatureMode.STREAMING, p.modality)
                        for p in registration.providers.server_streaming
                    ),
                }
                self.assertEqual(contract_keys, provider_keys)

    def test_algorithm_metadata_has_no_factories_or_topology_flags(self):
        field_names = {field.name for field in fields(AlgorithmSpec)}
        self.assertEqual(
            {"name", "draft", "feature_contracts", "capabilities"},
            field_names,
        )
        forbidden = {
            "supports_online",
            "supported_deployments",
            "supported_target_backends",
            "deployment_mode",
            "target_backend",
        }
        for registration in self.registry:
            with self.subTest(algorithm=registration.name):
                self.assertTrue(forbidden.isdisjoint(vars(registration.spec)))
                self.assertEqual(
                    registration.spec.supports_online,
                    bool(registration.providers.server_streaming),
                )

    def test_builtin_online_providers_are_server_streaming_only(self):
        for registration in self.registry:
            with self.subTest(algorithm=registration.name):
                providers = registration.providers
                self.assertGreaterEqual(len(providers.server_streaming), 1)
                self.assertFalse(hasattr(providers, "colocated"))
                self.assertFalse(hasattr(providers, "target_backend"))
                self.assertFalse(hasattr(providers, "deployment"))

    def test_draft_resolution_stays_outside_algorithm_spec(self):
        provider_fields = {field.name for field in fields(DraftConfigProvider)}
        self.assertEqual(
            {
                "architecture",
                "target_defaults",
                "expected_auto_map_model",
                "apply_overrides",
            },
            provider_fields,
        )
        forbidden = {
            "config_class",
            "model_class",
            "load_config",
            "resolve_config",
            "resolve_model",
        }
        self.assertTrue(forbidden.isdisjoint(provider_fields))
        for registration in self.registry:
            with self.subTest(algorithm=registration.name):
                self.assertFalse(hasattr(registration.spec, "draft_config_resolver"))

    def test_target_derived_defaults_and_overrides_are_provider_owned(self):
        expected = {
            "eagle3": ("llama", 1, 32000, False),
            "peagle": ("llama", 4, 32000, False),
            "dflash": ("qwen3", 1, None, True),
        }
        for name, (model_type, layers, vocab_size, has_override) in expected.items():
            with self.subTest(algorithm=name):
                policy = self.registry.resolve(name).providers.model.draft_config
                defaults = policy.target_defaults
                self.assertIsNotNone(defaults)
                self.assertEqual(model_type, defaults.model_type)
                self.assertEqual(layers, defaults.num_hidden_layers)
                self.assertEqual(vocab_size, defaults.draft_vocab_size)
                self.assertEqual(has_override, policy.apply_overrides is not None)

        for name in ("domino", "dspark"):
            with self.subTest(algorithm=name):
                policy = self.registry.resolve(name).providers.model.draft_config
                self.assertIsNone(policy.target_defaults)
                self.assertIsNone(policy.apply_overrides)

    def test_vlm_is_not_registered_as_a_builtin(self):
        for registration in self.registry:
            modalities = {
                contract.modality for contract in registration.spec.feature_contracts
            }
            self.assertEqual({"text"}, modalities, registration.name)

    def test_builtin_resume_contracts_cover_resolved_objective_semantics(self):
        training = SimpleNamespace(
            attention_backend="flex_attention",
            compact_teacher=True,
            compact_teacher_chunk_size=1024,
            lambda_base_start=0.75,
            lambda_base_decay_ratio=0.25,
        )
        config = SimpleNamespace(training=training)
        draft = SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=2),
            layers=[object(), object()],
            norm_before_residual=True,
            target_layer_ids=[3, 7],
            pure_draft_prefix_len=2,
        )
        dflash_family = SimpleNamespace(
            block_size=16,
            mask_token_id=0,
            attention_backend="flex_attention",
            num_anchors=64,
            loss_decay_gamma=2.0,
            loss_type="dpace",
            dpace_alpha=0.4,
            shift_label=True,
            dspark_ce_loss_alpha=0.1,
            dspark_l1_loss_alpha=0.8,
            dspark_confidence_head_alpha=0.2,
        )
        models = {
            "eagle3": SimpleNamespace(
                length=7,
                attention_backend="flex_attention",
                lk_loss_type="lambda",
                kl_scale=0.9,
                kl_decay=0.8,
            ),
            "peagle": SimpleNamespace(
                num_depths=5,
                down_sample_ratio=0.7,
                down_sample_ratio_min=0.3,
                mask_token_id=31,
            ),
            "dflash": dflash_family,
            "domino": dflash_family,
            "dspark": dflash_family,
        }
        expected_keys = {
            "eagle3": {
                "eagle3_ttt_length",
                "eagle3_lk_loss_type",
                "eagle3_kl_scale",
                "eagle3_kl_decay",
                "eagle3_compact_teacher",
            },
            "peagle": {
                "peagle_num_draft_layers",
                "peagle_num_depths",
                "peagle_down_sample_ratio",
                "peagle_mask_token_id",
            },
            "dflash": {
                "dflash_block_size",
                "dflash_num_anchors",
                "dflash_loss_type",
                "dflash_dpace_alpha",
            },
            "domino": {
                "domino_block_size",
                "domino_shift_label",
                "domino_pure_draft_prefix_len",
                "domino_lambda_base_start",
                "domino_lambda_base_decay_ratio",
            },
            "dspark": {
                "dspark_block_size",
                "dspark_ce_loss_alpha",
                "dspark_l1_loss_alpha",
                "dspark_confidence_head_alpha",
            },
        }

        for name in BUILTINS:
            with self.subTest(algorithm=name):
                step = self.registry.resolve(name).providers.step
                contract = step.resume_contract(config, draft, models[name])
                self.assertTrue(expected_keys[name] <= contract.keys())
                self.assertTrue(all(key.startswith(f"{name}_") for key in contract))

    def test_step_runtime_config_binds_options_contract_and_missing_key_policy(self):
        step = self.registry.resolve("eagle3").providers.step
        draft = SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=1),
            named_parameters=lambda: (),
            state_dict=lambda: {},
        )
        config = SimpleNamespace(
            training=SimpleNamespace(
                attention_backend="flex_attention",
                compact_teacher=False,
                compact_teacher_chunk_size=None,
            )
        )
        model = SimpleNamespace(
            length=7,
            attention_backend="flex_attention",
            lk_loss_type=None,
            kl_scale=1.0,
            kl_decay=1.0,
        )

        model_provenance = {"target_model": ("reference", "model-id")}
        runtime = step.bind_runtime(
            config,
            draft,
            model,
            model_provenance=model_provenance,
        )

        self.assertIsInstance(runtime, StepRuntimeConfig)
        self.assertEqual(dict(runtime), step.options(config))
        self.assertEqual(
            {
                key: value
                for key, value in runtime.resume_contract.items()
                if key not in {STEP_OPTIONS_CONTRACT_KEY, MODEL_PROVENANCE_CONTRACT_KEY}
            },
            step.resume_contract(config, draft, model),
        )
        self.assertEqual(
            runtime.resume_contract[STEP_OPTIONS_CONTRACT_KEY],
            (
                ("compact_teacher", False),
                ("compact_teacher_chunk_size", None),
            ),
        )
        self.assertEqual(
            runtime.resume_contract[MODEL_PROVENANCE_CONTRACT_KEY],
            (("target_model", ("reference", "model-id")),),
        )
        self.assertEqual(runtime.allowed_missing_checkpoint_keys, frozenset())

    def test_missing_checkpoint_policy_requires_a_runtime_fingerprint(self):
        with self.assertRaisesRegex(
            ValueError,
            OMITTED_STATE_FINGERPRINT_CONTRACT_KEY,
        ):
            StepRuntimeConfig(
                options={},
                resume_contract={},
                allowed_missing_checkpoint_keys={"embed_tokens.weight"},
            )

    def test_step_options_are_canonical_resume_metadata(self):
        runtime = StepRuntimeConfig(
            options={"objective_scale": 0.5},
            resume_contract={},
            allowed_missing_checkpoint_keys=frozenset(),
        )
        self.assertEqual(
            runtime.resume_contract[STEP_OPTIONS_CONTRACT_KEY],
            (("objective_scale", 0.5),),
        )

        with self.assertRaisesRegex(ValueError, STEP_OPTIONS_CONTRACT_KEY):
            StepRuntimeConfig(
                options={"objective_scale": 0.5},
                resume_contract={
                    STEP_OPTIONS_CONTRACT_KEY: (("objective_scale", 1.0),),
                },
                allowed_missing_checkpoint_keys=frozenset(),
            )

    def test_streaming_provider_constructs_a_generic_input_adapter(self):
        config = object()
        calls = []

        class GenericInputAdapter:
            def load_input_tools(self, received_config):
                return ("tools", received_config)

            def prepare_prompts(
                self,
                received_config,
                input_tools,
                *,
                draft_config,
            ):
                return [
                    {
                        "config": received_config,
                        "input_tools": input_tools,
                        "draft_config": draft_config,
                    }
                ]

            def build_request_inputs(self, tasks):
                return {"generic_model_inputs": list(tasks)}

        adapter = GenericInputAdapter()

        def build_input_adapter(received_config):
            calls.append(received_config)
            return adapter

        provider = ServerStreamingProvider(
            modality="vision_language",
            capture_method="eagle3",
            target_representation="hidden_state",
            layout=ServerCaptureLayout(
                aux_feature="hidden_state",
                last_hidden_feature="target",
                passthrough=(("position_ids", "position_ids", (3,)),),
            ),
            build_collator=lambda: None,
            build_input_adapter=build_input_adapter,
        )

        self.assertEqual("vision_language", provider.modality)
        self.assertIs(adapter, provider.create_input_adapter(config))
        self.assertEqual([config], calls)
        self.assertEqual(
            {"generic_model_inputs": ["task"]},
            adapter.build_request_inputs(["task"]),
        )

    def test_streaming_provider_rejects_incomplete_input_adapters(self):
        required_methods = (
            "load_input_tools",
            "prepare_prompts",
            "build_request_inputs",
        )
        for invalid_method in required_methods:
            methods = {
                name: (lambda *args, **kwargs: None) for name in required_methods
            }
            methods[invalid_method] = object()
            incomplete = SimpleNamespace(**methods)
            provider = ServerStreamingProvider(
                modality="plugin_modality",
                capture_method="eagle3",
                target_representation="hidden_state",
                layout=ServerCaptureLayout(
                    aux_feature="hidden_state",
                    last_hidden_feature="target",
                    passthrough=(),
                ),
                build_collator=lambda: None,
                build_input_adapter=lambda _config: incomplete,
            )

            with (
                self.subTest(method=invalid_method),
                self.assertRaisesRegex(
                    TypeError,
                    f"missing callable methods: \\['{invalid_method}'\\]",
                ),
            ):
                provider.create_input_adapter(object())

    def test_building_catalog_does_not_import_training_or_torch(self):
        code = (
            "import sys; "
            "from specforge.algorithms.builtin import builtin_algorithm_registry; "
            "r=builtin_algorithm_registry(); assert len(r)==5; "
            "assert 'torch' not in sys.modules; "
            "assert 'specforge.training.strategies.registry' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr or result.stdout)


if __name__ == "__main__":
    unittest.main()
