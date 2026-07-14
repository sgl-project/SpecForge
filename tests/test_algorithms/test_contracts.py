from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from specforge.algorithms import (
    AlgorithmCapabilities,
    AlgorithmRegistry,
    AlgorithmSpec,
    DraftArchitectureContract,
    FeatureContract,
    FeatureMode,
)


def _resolve_mixed_attention(raw_config):
    return tuple(layer["attention"] for layer in raw_config["layers"])


def _spec(name: str = "eagle3") -> AlgorithmSpec:
    return AlgorithmSpec(
        name=name,
        draft=DraftArchitectureContract(
            architecture=f"{name}-draft",
            resolve_config=_resolve_mixed_attention,
        ),
        feature_contracts=(
            FeatureContract(
                mode=FeatureMode.OFFLINE,
                required_tensors={"input_ids", "hidden_states", "loss_mask"},
                optional_tensors={"position_ids"},
                allowed_target_representations={"hidden_state"},
            ),
            FeatureContract(
                mode=FeatureMode.STREAMING,
                required_tensors={"input_ids", "hidden_states", "loss_mask"},
                allowed_target_representations={"hidden_state", "logits"},
            ),
        ),
        capabilities=AlgorithmCapabilities(
            modalities={"text", "vision_language"},
            attention_backends={"sdpa", "flex_attention"},
            supports_compact_teacher=True,
            supports_vocab_mapping=True,
        ),
    )


class AlgorithmContractTest(unittest.TestCase):
    def test_draft_structure_is_algorithm_owned_and_opaque(self):
        spec = _spec()

        resolved = spec.draft.resolve(
            {
                "layers": [
                    {"attention": "swa"},
                    {"attention": "swa"},
                    {"attention": "full"},
                ]
            }
        )

        self.assertEqual(("swa", "swa", "full"), resolved)

    def test_draft_resolver_receives_a_read_only_mapping(self):
        def mutate(raw_config):
            raw_config["layers"][0]["attention"] = "full"

        contract = DraftArchitectureContract("draft", mutate)
        raw_config = {"layers": [{"attention": "swa"}]}

        with self.assertRaises(TypeError):
            contract.resolve(raw_config)
        self.assertEqual("swa", raw_config["layers"][0]["attention"])

    def test_feature_contract_normalizes_names_and_derives_modes(self):
        spec = _spec()

        offline = spec.feature_contract("offline")

        self.assertIsInstance(offline.required_tensors, frozenset)
        self.assertEqual(
            frozenset({FeatureMode.OFFLINE, FeatureMode.STREAMING}),
            spec.feature_modes,
        )

    def test_required_and_optional_tensors_must_be_disjoint(self):
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            FeatureContract(
                mode=FeatureMode.OFFLINE,
                required_tensors={"input_ids"},
                optional_tensors={"input_ids"},
            )

    def test_algorithm_has_at_most_one_contract_per_mode(self):
        contract = FeatureContract(
            mode=FeatureMode.OFFLINE,
            required_tensors={"input_ids"},
        )

        with self.assertRaisesRegex(ValueError, "duplicate modes"):
            AlgorithmSpec(
                name="eagle3",
                draft=DraftArchitectureContract("draft", lambda raw: raw),
                feature_contracts=(contract, contract),
                capabilities=AlgorithmCapabilities(
                    modalities={"text"},
                    attention_backends={"sdpa"},
                ),
            )

    def test_algorithm_rejects_untyped_contract_parts(self):
        with self.assertRaisesRegex(TypeError, "DraftArchitectureContract"):
            AlgorithmSpec(
                name="eagle3",
                draft=object(),
                feature_contracts=(_spec().feature_contract(FeatureMode.OFFLINE),),
                capabilities=_spec().capabilities,
            )

    def test_required_batch_size_is_positive(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            AlgorithmCapabilities(
                modalities={"text"},
                attention_backends={"sdpa"},
                required_batch_size=0,
            )

    def test_contracts_are_immutable(self):
        spec = _spec()

        with self.assertRaises(FrozenInstanceError):
            spec.name = "dflash"

    def test_missing_feature_mode_reports_supported_modes(self):
        offline_only = _spec().__class__(
            name="offline_only",
            draft=_spec().draft,
            feature_contracts=(_spec().feature_contract(FeatureMode.OFFLINE),),
            capabilities=_spec().capabilities,
        )

        with self.assertRaisesRegex(KeyError, r"supported modes: \['offline'\]"):
            offline_only.feature_contract(FeatureMode.STREAMING)


class AlgorithmRegistryTest(unittest.TestCase):
    def test_registry_is_deterministic_and_explicit(self):
        registry = AlgorithmRegistry((_spec("peagle"), _spec("eagle3")))

        self.assertEqual(("eagle3", "peagle"), registry.names)
        self.assertEqual("peagle", registry.resolve("peagle").name)
        self.assertEqual(["eagle3", "peagle"], [spec.name for spec in registry])

    def test_registry_instances_do_not_share_state(self):
        empty = AlgorithmRegistry()
        populated = empty.with_spec(_spec())

        self.assertEqual((), empty.names)
        self.assertEqual(("eagle3",), populated.names)

    def test_duplicate_registration_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "duplicate algorithm"):
            AlgorithmRegistry((_spec(), _spec()))

    def test_unknown_algorithm_lists_registered_names(self):
        registry = AlgorithmRegistry((_spec("eagle3"),))

        with self.assertRaisesRegex(KeyError, r"registered algorithms: \['eagle3'\]"):
            registry.resolve("missing")


if __name__ == "__main__":
    unittest.main()
