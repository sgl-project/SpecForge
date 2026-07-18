from __future__ import annotations

import unittest

import torch

from specforge.algorithms.builtin import builtin_algorithm_registry


class BuiltinProviderParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = builtin_algorithm_registry()

    def test_builtin_contract_matrix_preserves_pr678_capabilities(self):
        expected = {
            "dflash": (
                "DFlashDraftModel",
                {"input_ids", "loss_mask", "hidden_states"},
                {"eager", "sdpa", "flex_attention"},
                None,
            ),
            "domino": (
                "DominoDraftModel",
                {"input_ids", "loss_mask", "hidden_states"},
                {"eager", "sdpa", "flex_attention"},
                None,
            ),
            "dspark": (
                "DSparkDraftModel",
                {
                    "input_ids",
                    "loss_mask",
                    "hidden_states",
                    "target_last_hidden_states",
                },
                {"eager", "sdpa", "flex_attention"},
                None,
            ),
            "eagle3": (
                "LlamaForCausalLMEagle3",
                {
                    "input_ids",
                    "attention_mask",
                    "loss_mask",
                    "hidden_state",
                    "target",
                },
                {"sdpa", "flex_attention", "fa", "usp"},
                None,
            ),
            "peagle": (
                "PEagleDraftModel",
                {
                    "input_ids",
                    "attention_mask",
                    "loss_mask",
                    "hidden_state",
                    "target",
                },
                {"flex_attention"},
                1,
            ),
        }
        for name, (architecture, features, backends, batch_size) in expected.items():
            with self.subTest(algorithm=name):
                registration = self.registry.resolve(name)
                self.assertEqual(
                    architecture,
                    registration.providers.model.draft_config.architecture,
                )
                streaming = registration.spec.feature_contract("streaming", "text")
                self.assertEqual(features, streaming.required_tensors)
                self.assertEqual(
                    backends,
                    registration.spec.capabilities.attention_backends,
                )
                self.assertEqual(
                    batch_size,
                    registration.spec.capabilities.required_batch_size,
                )

    def test_existing_server_capture_layouts_are_preserved(self):
        expected = {
            "dflash": (
                "hidden_states",
                None,
                (("input_ids", "input_ids", ()), ("loss_mask", "loss_mask", ())),
                None,
            ),
            "domino": (
                "hidden_states",
                None,
                (("input_ids", "input_ids", ()), ("loss_mask", "loss_mask", ())),
                None,
            ),
            "dspark": (
                "hidden_states",
                "target_last_hidden_states",
                (("input_ids", "input_ids", ()), ("loss_mask", "loss_mask", ())),
                None,
            ),
            "eagle3": (
                "hidden_state",
                "target",
                (("input_ids", "input_ids", ()), ("loss_mask", "loss_mask", ())),
                "attention_mask",
            ),
            "peagle": (
                "hidden_state",
                "target",
                (("input_ids", "input_ids", ()), ("loss_mask", "loss_mask", ())),
                "attention_mask",
            ),
        }
        for name, layout_values in expected.items():
            with self.subTest(algorithm=name):
                layout = (
                    self.registry.resolve(name)
                    .providers.server_streaming_for("text")
                    .layout
                )
                self.assertEqual(
                    layout_values,
                    (
                        layout.aux_feature,
                        layout.last_hidden_feature,
                        layout.passthrough,
                        layout.attention_mask_feature,
                    ),
                )

    def test_peagle_reuses_the_eagle_server_contract_explicitly(self):
        peagle = self.registry.resolve("peagle").providers
        eagle3 = self.registry.resolve("eagle3").providers
        peagle_stream = peagle.server_streaming_for("text")
        eagle_stream = eagle3.server_streaming_for("text")

        self.assertEqual("eagle3", peagle_stream.capture_method)
        self.assertEqual(eagle_stream.layout, peagle_stream.layout)
        self.assertEqual("hidden_state", peagle_stream.target_representation)
        self.assertTrue(peagle.step.uses_external_target_head)

    def test_step_factories_preserve_concrete_strategy_types(self):
        expected = {
            "eagle3": "Eagle3TrainStrategy",
            "peagle": "PEagleTrainStrategy",
            "dflash": "DFlashTrainStrategy",
            "domino": "DominoTrainStrategy",
            "dspark": "DSparkTrainStrategy",
        }
        model = object()
        for name, class_name in expected.items():
            with self.subTest(algorithm=name):
                provider = self.registry.resolve(name).providers.step
                instance = provider.build(model, target_head=None)
                self.assertEqual(class_name, type(instance).__name__)

    def test_dflash_family_collator_matches_existing_padding_contract(self):
        short = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "loss_mask": torch.tensor([[1, 1, 1]]),
            "hidden_states": torch.ones(1, 3, 4),
        }
        long = {
            "input_ids": torch.tensor([[4, 5, 6, 7, 8]]),
            "loss_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "hidden_states": torch.ones(1, 5, 4),
        }
        for name in ("dflash", "domino"):
            with self.subTest(algorithm=name):
                collate = (
                    self.registry.resolve(name)
                    .providers.server_streaming_for("text")
                    .build_collator()
                )
                batch = collate([short, long])
                self.assertEqual((2, 5), tuple(batch["input_ids"].shape))
                self.assertEqual((2, 5, 4), tuple(batch["hidden_states"].shape))
                self.assertEqual([1, 1, 1, 0, 0], batch["loss_mask"][0].tolist())

    def test_dspark_collator_preserves_target_last_hidden_states(self):
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
        collate = (
            self.registry.resolve("dspark")
            .providers.server_streaming_for("text")
            .build_collator()
        )
        batch = collate([short, long])
        self.assertEqual((2, 3, 3), tuple(batch["target_last_hidden_states"].shape))
        self.assertTrue(torch.all(batch["target_last_hidden_states"][0, 2:] == 0))

    def test_small_normalizers_match_retained_implementations(self):
        from specforge.data.preprocessing import (
            process_offline_dflash_sample,
            process_offline_eagle3_sample,
        )

        eagle_raw = {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "loss_mask": torch.ones(4, dtype=torch.long),
            "hidden_state": torch.arange(24).reshape(1, 4, 6),
            "aux_hidden_state": torch.arange(48).reshape(1, 4, 12),
        }
        dflash_raw = {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "loss_mask": torch.ones(4, dtype=torch.long),
            "hidden_states": torch.arange(24).reshape(1, 4, 6),
        }
        cases = (
            ("eagle3", eagle_raw, process_offline_eagle3_sample),
            ("dflash", dflash_raw, process_offline_dflash_sample),
            ("domino", dflash_raw, process_offline_dflash_sample),
        )
        for name, raw, retained in cases:
            with self.subTest(algorithm=name):
                provider = self.registry.resolve(name).providers.offline_for("text")
                actual = provider.build_normalizer(3)(raw)
                expected = retained(raw, 3)
                self.assertEqual(set(expected), set(actual))
                for key in expected:
                    self.assertTrue(torch.equal(expected[key], actual[key]), key)


if __name__ == "__main__":
    unittest.main()
