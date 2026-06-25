# coding=utf-8
"""Contract dataclasses + the no-tensor invariant (CPU-only)."""

import dataclasses
import unittest

import torch

from specforge.runtime.contracts import (
    FeatureSpec,
    PromptTask,
    SampleRef,
    TrainBatch,
    assert_no_tensors,
)


class TestContracts(unittest.TestCase):
    def _sample_ref(self) -> SampleRef:
        return SampleRef(
            sample_id="s0",
            run_id="r0",
            source_task_id=None,
            feature_store_uri="mem://x/s0",
            feature_keys={"input_ids": "s0/input_ids"},
            feature_specs={
                "input_ids": FeatureSpec("input_ids", (1, 8), "int64"),
                "target": FeatureSpec(
                    "target",
                    (1, 8, 16),
                    "bfloat16",
                    target_repr="hidden_state",
                    target_meta={"lm_head_key": "lm_head.weight"},
                ),
            },
            strategy="eagle3",
            num_tokens=8,
        )

    def test_roundtrip_asdict(self):
        ref = self._sample_ref()
        d = dataclasses.asdict(ref)
        self.assertEqual(d["sample_id"], "s0")
        self.assertEqual(d["feature_specs"]["target"]["target_repr"], "hidden_state")

    def test_frozen(self):
        ref = self._sample_ref()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ref.sample_id = "nope"

    def test_assert_no_tensors_passes_on_metadata_records(self):
        assert_no_tensors(self._sample_ref())
        assert_no_tensors(
            PromptTask(
                task_id="t",
                run_id="r",
                source_id="s",
                payload={"input_ids": [1, 2, 3], "nested": {"a": [4, 5]}},
                max_length=8,
            )
        )

    def test_assert_no_tensors_catches_tensor_in_metadata(self):
        ref = self._sample_ref()
        # smuggle a tensor into metadata -> must be caught
        bad = dataclasses.replace(ref, metadata={"sneaky": torch.zeros(3)})
        with self.assertRaises(TypeError):
            assert_no_tensors(bad)

    def test_assert_no_tensors_catches_tensor_in_payload(self):
        bad = PromptTask(
            task_id="t",
            run_id="r",
            source_id="s",
            payload={"ids": torch.zeros(3)},
            max_length=8,
        )
        with self.assertRaises(TypeError):
            assert_no_tensors(bad)

    def test_trainbatch_holds_tensors(self):
        batch = TrainBatch(
            sample_ids=["s0"],
            strategy="eagle3",
            tensors={"input_ids": torch.zeros(1, 8, dtype=torch.long)},
            metadata={},
        )
        # TrainBatch is the ONLY contract allowed to carry tensors.
        self.assertIn("input_ids", batch.tensors)
        with self.assertRaises(TypeError):
            assert_no_tensors(batch)


if __name__ == "__main__":
    unittest.main(verbosity=2)
