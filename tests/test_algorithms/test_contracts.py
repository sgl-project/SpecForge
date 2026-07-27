from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, fields, is_dataclass
from enum import Enum

from specforge.algorithms import (
    AlgorithmCapabilities,
    AlgorithmSpec,
    DraftRequirement,
    FeatureContract,
    FeatureMode,
    OfflineStorageContract,
)


def _offline(modality: str = "text") -> FeatureContract:
    return FeatureContract(
        mode=FeatureMode.OFFLINE,
        modality=modality,
        required_tensors={"input_ids", "hidden_states", "loss_mask"},
        allowed_target_representations={"hidden_state"},
        default_target_representation="hidden_state",
        storage=OfflineStorageContract(
            format="manifest_v1",
            required_tensors={"input_ids", "hidden_states", "loss_mask"},
            normalizer="eagle3_offline_v1",
        ),
    )


def _streaming(modality: str = "text") -> FeatureContract:
    required = {"input_ids", "hidden_state", "target", "loss_mask"}
    if modality == "vision_language":
        required.add("position_ids")
    return FeatureContract(
        mode=FeatureMode.STREAMING,
        modality=modality,
        required_tensors=required,
        allowed_target_representations={"hidden_state"},
        default_target_representation="hidden_state",
    )


def _spec(*contracts: FeatureContract) -> AlgorithmSpec:
    return AlgorithmSpec(
        name="eagle3",
        draft=DraftRequirement(
            compatible_architectures={"LlamaForCausalLMEagle3"},
            default_architecture="LlamaForCausalLMEagle3",
            supported_overrides={"attention_layout", "num_hidden_layers"},
        ),
        feature_contracts=contracts or (_offline(), _streaming()),
        capabilities=AlgorithmCapabilities(
            attention_backends={"sdpa", "flex_attention"},
            supports_compact_teacher=True,
            supports_vocab_mapping=True,
        ),
    )


def _assert_contract_is_pure(value: object) -> None:
    if isinstance(value, type) or callable(value):
        raise AssertionError(f"executable value leaked into contract: {value!r}")
    if value is None or isinstance(value, (str, int, float, bool, Enum)):
        return
    if isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            _assert_contract_is_pure(item)
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _assert_contract_is_pure(getattr(value, field.name))
        return
    raise AssertionError(f"opaque value leaked into contract: {type(value).__name__}")


class AlgorithmContractTest(unittest.TestCase):
    def test_spec_is_recursively_pure(self):
        _assert_contract_is_pure(_spec())

    def test_same_mode_can_define_multiple_modalities(self):
        spec = _spec(_streaming("text"), _streaming("vision_language"))

        self.assertTrue(spec.supports("streaming", "text"))
        self.assertIn(
            "position_ids",
            spec.feature_contract("streaming", "vision_language").required_tensors,
        )

    def test_online_capability_is_derived_from_streaming_contract(self):
        self.assertFalse(_spec(_offline()).supports_online)
        self.assertTrue(_spec(_streaming()).supports_online)

    def test_duplicate_mode_and_modality_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _spec(_streaming(), _streaming())

    def test_offline_requires_storage_contract(self):
        with self.assertRaisesRegex(ValueError, "require a storage contract"):
            FeatureContract(
                mode="offline",
                modality="text",
                required_tensors={"input_ids"},
            )

    def test_streaming_rejects_offline_storage_contract(self):
        with self.assertRaisesRegex(ValueError, "cannot define storage"):
            FeatureContract(
                mode="streaming",
                modality="text",
                required_tensors={"input_ids"},
                storage=OfflineStorageContract(
                    format="manifest_v1",
                    required_tensors={"input_ids"},
                    normalizer="text_v1",
                ),
            )

    def test_contracts_are_immutable(self):
        spec = _spec()

        with self.assertRaises(FrozenInstanceError):
            spec.name = "dflash"

    def test_draft_requirement_accepts_flexible_layout_override(self):
        requirement = _spec().draft

        self.assertIn("attention_layout", requirement.supported_overrides)


if __name__ == "__main__":
    unittest.main()
