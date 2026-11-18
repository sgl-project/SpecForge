"""Runtime contracts remain independent of model and objective registries."""

import unittest
from types import SimpleNamespace

from specforge.algorithms.common.providers import (
    OMITTED_STATE_FINGERPRINT_CONTRACT_KEY,
    STEP_OPTIONS_CONTRACT_KEY,
    ServerCaptureLayout,
    ServerStreamingProvider,
    StepRuntimeConfig,
)
from tests.test_algorithms.fixtures import builtin_algorithm_registry


class RuntimeProviderTest(unittest.TestCase):
    def setUp(self):
        self.registry = builtin_algorithm_registry()

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
