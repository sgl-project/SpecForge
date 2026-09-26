from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.algorithms.common.vlm_input import (
    VlmServerInputAdapter,
    build_vlm_input_adapter,
)
from specforge.algorithms.contracts import FeatureMode

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch-free dev boxes
    torch = None
    TORCH_AVAILABLE = False


def _multimodal_provider(registry):
    return registry.resolve("dspark").providers.server_streaming_for("multimodal")


def _prompt_config():
    return SimpleNamespace(
        model=SimpleNamespace(
            target_model_path="target",
            cache_dir=None,
            trust_remote_code=False,
        ),
        data=SimpleNamespace(
            prompts_path=None,
            train_data_path="train.jsonl",
            chat_template="qwen3.5",
            max_length=64,
            max_prompts=None,
        ),
    )


class DSparkMultimodalRegistrationTest(unittest.TestCase):
    def setUp(self):
        self.registry = builtin_algorithm_registry()

    def test_dspark_registers_the_multimodal_streaming_contract(self):
        registration = self.registry.resolve("dspark")
        contract = registration.spec.feature_contract(
            FeatureMode.STREAMING, "multimodal"
        )
        self.assertEqual(
            set(contract.required_tensors),
            {
                "input_ids",
                "loss_mask",
                "hidden_states",
                "target_last_hidden_states",
            },
        )
        self.assertEqual({"hidden_state"}, set(contract.allowed_target_representations))
        self.assertEqual("hidden_state", contract.default_target_representation)

    def test_multimodal_provider_mirrors_the_text_capture_layout(self):
        providers = self.registry.resolve("dspark").providers
        provider = providers.server_streaming_for("multimodal")
        text = providers.server_streaming_for("text")
        self.assertEqual("dspark", provider.capture_method)
        self.assertEqual("hidden_state", provider.target_representation)
        self.assertEqual(text.layout, provider.layout)
        self.assertEqual("hidden_states", provider.layout.aux_feature)
        self.assertEqual(
            "target_last_hidden_states", provider.layout.last_hidden_feature
        )
        self.assertIs(text.build_collator, provider.build_collator)

    def test_input_adapter_factory_builds_a_valid_adapter(self):
        provider = _multimodal_provider(self.registry)
        adapter = provider.create_input_adapter(
            SimpleNamespace(model=SimpleNamespace(), data=SimpleNamespace())
        )
        self.assertIsInstance(adapter, VlmServerInputAdapter)

    def test_input_adapter_injects_dspark_minimum_loss_tokens(self):
        from specforge.algorithms.dspark import providers as dspark_providers

        build = _multimodal_provider(self.registry).build_input_adapter
        self.assertIs(build_vlm_input_adapter, build.func)
        self.assertIs(
            dspark_providers.minimum_loss_tokens,
            build.keywords["minimum_loss_tokens"],
        )


class DSparkMinLossTokensTest(unittest.TestCase):
    def _run_prepare_prompts(self, adapter, config, draft_config):
        captured = {}

        def fake_build_payloads(path, tokenizer, processor, **kwargs):
            captured.update(kwargs)
            return [{"input_ids": [1]}]

        with (
            mock.patch(
                "transformers.AutoProcessor.from_pretrained",
                return_value=object(),
            ),
            mock.patch(
                "specforge.data.vlm_preprocessing.build_vlm_prompt_payloads",
                side_effect=fake_build_payloads,
            ),
        ):
            prompts = adapter.prepare_prompts(
                config, object(), draft_config=draft_config
            )
        self.assertEqual([{"input_ids": [1]}], prompts)
        return captured

    def test_prepare_prompts_uses_the_injected_minimum_loss_tokens(self):
        calls = []

        def floor(config, draft_config):
            calls.append((config, draft_config))
            return 7

        config = _prompt_config()
        draft_config = SimpleNamespace(block_size=2)
        adapter = build_vlm_input_adapter(config, minimum_loss_tokens=floor)
        captured = self._run_prepare_prompts(adapter, config, draft_config)
        self.assertEqual(7, captured["min_loss_tokens"])
        self.assertEqual([(config, draft_config)], calls)

    def test_prepare_prompts_defaults_to_the_dflash_floor(self):
        adapter = VlmServerInputAdapter(config=_prompt_config())
        captured = self._run_prepare_prompts(
            adapter, adapter._config, SimpleNamespace(block_size=2)
        )
        self.assertEqual(2, captured["min_loss_tokens"])
        with self.assertRaisesRegex(ValueError, "block_size >= 2"):
            self._run_prepare_prompts(
                adapter, adapter._config, SimpleNamespace(block_size=1)
            )

    def test_registered_adapter_resolves_the_dspark_floor(self):
        registry = builtin_algorithm_registry()
        config = _prompt_config()
        adapter = _multimodal_provider(registry).create_input_adapter(config)
        captured = self._run_prepare_prompts(
            adapter, config, SimpleNamespace(block_size=2)
        )
        self.assertEqual(2, captured["min_loss_tokens"])
        with self.assertRaisesRegex(ValueError, "block_size >= 2"):
            self._run_prepare_prompts(adapter, config, SimpleNamespace(block_size=1))


class DSparkVlmRequestInputsTest(unittest.TestCase):
    def test_build_request_inputs_uses_collapsed_ids_and_image_data(self):
        registry = builtin_algorithm_registry()
        adapter = _multimodal_provider(registry).create_input_adapter(SimpleNamespace())
        tasks = [
            SimpleNamespace(
                payload={
                    "input_ids": [1, 2, 2, 2, 3],
                    "request_input_ids": [1, 2, 3],
                    "image_data": "aGVsbG8=",
                }
            ),
            SimpleNamespace(
                payload={
                    "input_ids": [7, 8],
                    "request_input_ids": [7, 8],
                    "image_data": None,
                }
            ),
        ]
        request = adapter.build_request_inputs(tasks)
        self.assertEqual(request["input_ids"], [[1, 2, 3], [7, 8]])
        self.assertEqual(request["image_data"], ["aGVsbG8=", None])


@unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
class DSparkMultimodalCollatorTest(unittest.TestCase):
    """Multimodal batches are isomorphic to text batches (plain-rope design):
    the multimodal provider reuses the DSpark collator verbatim."""

    def test_multimodal_collator_pads_all_four_features(self):
        short = {
            "input_ids": torch.tensor([[1, 2]]),
            "loss_mask": torch.ones(1, 2, dtype=torch.long),
            "hidden_states": torch.ones(1, 2, 4),
            "target_last_hidden_states": torch.ones(1, 2, 3),
        }
        long = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "loss_mask": torch.ones(1, 3, dtype=torch.long),
            "hidden_states": torch.ones(1, 3, 4),
            "target_last_hidden_states": torch.ones(1, 3, 3),
        }
        collate = _multimodal_provider(builtin_algorithm_registry()).build_collator()
        batch = collate([short, long])
        self.assertEqual((2, 3), tuple(batch["input_ids"].shape))
        self.assertEqual((2, 3, 4), tuple(batch["hidden_states"].shape))
        self.assertEqual((2, 3, 3), tuple(batch["target_last_hidden_states"].shape))
        self.assertTrue(torch.all(batch["target_last_hidden_states"][0, 2:] == 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
