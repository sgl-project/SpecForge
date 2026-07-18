from __future__ import annotations

import unittest
from dataclasses import dataclass

from specforge.algorithms import (
    AlgorithmCapabilities,
    AlgorithmRegistration,
    AlgorithmRegistry,
    AlgorithmSpec,
    DraftRequirement,
    FeatureContract,
    FeatureMode,
)


@dataclass(frozen=True)
class _Providers:
    algorithm_name: str


def _registration(name: str) -> AlgorithmRegistration:
    return AlgorithmRegistration(
        spec=AlgorithmSpec(
            name=name,
            draft=DraftRequirement(
                compatible_architectures={f"{name}-draft"},
                default_architecture=f"{name}-draft",
            ),
            feature_contracts=(
                FeatureContract(
                    mode=FeatureMode.STREAMING,
                    modality="text",
                    required_tensors={"input_ids"},
                ),
            ),
            capabilities=AlgorithmCapabilities(attention_backends={"sdpa"}),
        ),
        providers=_Providers(name),
    )


class AlgorithmRegistryTest(unittest.TestCase):
    def test_registry_is_deterministic_and_instance_owned(self):
        empty = AlgorithmRegistry()
        registry = AlgorithmRegistry((_registration("peagle"), _registration("eagle3")))

        self.assertEqual((), empty.names)
        self.assertEqual(("eagle3", "peagle"), registry.names)
        self.assertEqual("peagle", registry.resolve("peagle").name)

    def test_with_registration_returns_new_registry(self):
        empty = AlgorithmRegistry()
        populated = empty.with_registration(_registration("eagle3"))

        self.assertEqual((), empty.names)
        self.assertEqual(("eagle3",), populated.names)

    def test_duplicate_registration_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "duplicate algorithm"):
            AlgorithmRegistry((_registration("eagle3"), _registration("eagle3")))

    def test_provider_name_must_match_spec(self):
        registration = _registration("eagle3")

        with self.assertRaisesRegex(ValueError, "must match"):
            AlgorithmRegistration(
                spec=registration.spec,
                providers=_Providers("dflash"),
            )

    def test_unknown_algorithm_lists_registered_names(self):
        registry = AlgorithmRegistry((_registration("eagle3"),))

        with self.assertRaisesRegex(
            KeyError,
            r"registered algorithms: \['eagle3'\]",
        ):
            registry.resolve("missing")


if __name__ == "__main__":
    unittest.main()
