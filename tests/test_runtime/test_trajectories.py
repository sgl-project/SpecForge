"""A recorded training trajectory is reproducible and pins the optimizer window."""

import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from tests.test_runtime.test_canonical_end_to_end import recomputed_tolerance
from tests.test_runtime.trajectories import (
    BATCHES_PER_EPOCH,
    _vary_supervision,
    cases,
    differences,
    record,
)


class TrajectoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = cases()
        cls.first = record(cls.cases)
        cls.second = record(cls.cases)

    def test_second_recording_reproduces_the_first(self):
        # Loss, weights, optimizer, RNG and data position of every update.
        # Two accelerator runs are held to the end-to-end test's tolerance.
        self.assertEqual(
            differences(self.second, self.first, atol=recomputed_tolerance()), []
        )

    def test_window_spans_the_epoch_boundary(self):
        """Three batches an epoch, two epochs, accumulation two: three updates."""
        for case in self.cases:
            with self.subTest(case=case.name):
                positions = [
                    divmod(step * case.accumulation_steps - 1, BATCHES_PER_EPOCH)
                    for step in range(1, case.updates + 1)
                ]
                self.assertEqual(
                    [
                        (
                            update["checkpoint"],
                            update["shared"]["global_step"],
                            update["shared"]["epoch"],
                            update["shared"]["epoch_batch"],
                        )
                        for update in self.first[case.name]
                    ],
                    [
                        (step, step, epoch, batch + 1)
                        for step, (epoch, batch) in enumerate(positions, start=1)
                    ],
                )

    def test_every_rank_logs_and_applies_the_same(self):
        for case in self.cases:
            with self.subTest(case=case.name):
                for update in self.first[case.name]:
                    for name in ("logged", "applied_lr"):
                        self.assertEqual(len(update[name]), case.world_size)
                        for other in update[name][1:]:
                            self.assertEqual(differences(other, update[name][0]), [])

    def test_every_replica_holds_the_saved_weights(self):
        """DDP saves rank 0's weights; every rank's process holds the same bits."""
        for case in self.cases:
            if case.sharding != "NO_SHARD":
                continue
            with self.subTest(case=case.name):
                for update in self.first[case.name]:
                    saved = update["shared"]["draft_state_dict"]
                    self.assertEqual(len(update["replicas"]), case.world_size)
                    for replica in update["replicas"]:
                        held = {
                            name.removeprefix("draft_model."): tensor
                            for name, tensor in replica["weights"].items()
                        }
                        self.assertEqual(
                            differences({name: held[name] for name in saved}, saved),
                            [],
                        )
                        self.assertEqual(
                            differences(replica, update["replicas"][0]), []
                        )

    def test_records_differ_in_supervised_length(self):
        """Otherwise every rank holds the same loss denominator."""
        with TemporaryDirectory() as temp:
            features = Path(temp)
            for index in range(3):
                torch.save(
                    {"loss_mask": torch.tensor([0, 1, 1, 1])}, features / str(index)
                )
            _vary_supervision(features)
            self.assertEqual(
                [
                    int(torch.load(features / str(index))["loss_mask"].sum())
                    for index in range(3)
                ],
                [3, 2, 1],
            )


class DifferencesTest(unittest.TestCase):
    def test_names_each_changed_value(self):
        expected = {"run": [{"loss": torch.tensor(1.0), "step": 1, "lr": 0.5}]}
        changed = copy.deepcopy(expected)
        changed["run"][0]["loss"] += 1
        changed["run"][0]["lr"] = 0.25
        self.assertEqual(
            differences(changed, expected),
            [
                "recording.run[0].loss: values differ",
                "recording.run[0].lr: 0.25 != 0.5",
            ],
        )

    def test_tolerance_reaches_floating_tensors_only(self):
        expected = {"weight": torch.tensor([1.0]), "lr": 1.0, "step": torch.tensor(1)}
        near = {"weight": torch.tensor([1.05]), "lr": 1.05, "step": torch.tensor(2)}
        self.assertEqual(
            differences(near, expected, atol=0.1),
            ["recording.lr: 1.05 != 1.0", "recording.step: values differ"],
        )

    def test_exact_comparison_is_bitwise(self):
        nan = torch.tensor([float("nan"), float("inf")])
        self.assertEqual(differences(nan.clone(), nan), [])
        self.assertEqual(differences(float("inf"), float("inf")), [])
        self.assertEqual(
            differences(torch.tensor([-0.0]), torch.tensor([0.0])),
            ["recording: values differ"],
        )
        self.assertEqual(differences(-0.0, 0.0), ["recording: -0.0 != 0.0"])

    def test_a_value_of_another_type_differs(self):
        for other in (True, 1.0, torch.tensor(1), torch.tensor([1, 1])):
            self.assertEqual(len(differences({"step": other}, {"step": 1})), 1)
        self.assertEqual(differences((1, 2), [1, 2]), ["recording: tuple != list"])
        self.assertEqual(
            differences({"a": 1}, {"b": 1}),
            ["recording: keys differ in ['a', 'b']"],
        )
        self.assertEqual(
            differences({True: "x"}, {1: "x"}),
            ["recording: keys differ in ['1', 'True']"],
        )


if __name__ == "__main__":
    unittest.main()
