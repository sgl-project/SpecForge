from __future__ import annotations

import subprocess
import sys
import unittest
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.algorithms.common.providers import (
    AlgorithmProviders,
    DraftConfigProvider,
    ServerCaptureLayout,
    ServerStreamingProvider,
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
