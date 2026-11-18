"""A recorded training trajectory is reproducible and pins the optimizer window."""

import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from tests.test_runtime.test_canonical_end_to_end import recomputed_tolerance
from tests.test_runtime.trajectories import (
    BATCHES_PER_EPOCH,
    _vary_records,
    cases,
    differences,
    record,
)


class TrajectoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = cases()
        # An accelerator this host cannot see would leave nothing to check.
        assert cls.cases
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
                logged = []
                for update in self.first[case.name]:
                    self.assertEqual(update["shared"]["world_size"], case.world_size)
                    for name in ("logged", "evaluated", "applied_lr"):
                        self.assertEqual(len(update[name]), case.world_size)
                        for other in update[name][1:]:
                            self.assertEqual(differences(other, update[name][0]), [])
                    logged.append(update["logged"][0] is not None)
                self.assertEqual(
                    logged,
                    [
                        step % case.log_interval == 0
                        for step in range(1, case.updates + 1)
                    ],
                )

    def test_evaluated_cases_report_a_pooled_loss_after_every_update(self):
        evaluated = [case for case in self.cases if case.evaluated]
        self.assertTrue(evaluated)
        for case in evaluated:
            with self.subTest(case=case.name):
                for update in self.first[case.name]:
                    loss = update["evaluated"][0]["eval/avg_loss"]
                    self.assertTrue(torch.isfinite(loss))

    def test_runs_warm_up(self):
        """A schedule without its warmup would otherwise record the same rates."""
        for case in self.cases:
            with self.subTest(case=case.name):
                rates = [update["applied_lr"][0][0] for update in self.first[case.name]]
                self.assertLess(rates[0], max(rates))

    def test_every_replica_holds_the_saved_state(self):
        """DDP saves rank 0's state; every rank's process holds the same bits."""
        for recording in (self.first, self.second):
            for case in self.cases:
                if case.sharding != "NO_SHARD":
                    continue
                with self.subTest(case=case.name):
                    for update in recording[case.name]:
                        shared = update["shared"]
                        saved = shared["draft_state_dict"]
                        masters = shared["replicated_optimizer_state"]["fp32_params"]
                        self.assertEqual(len(update["replicas"]), case.world_size)
                        for replica in update["replicas"]:
                            held = {
                                name.removeprefix("draft_model."): tensor
                                for name, tensor in replica["weights"].items()
                            }
                            self.assertEqual(
                                differences(
                                    {name: held[name] for name in saved}, saved
                                ),
                                [],
                            )
                            self.assertEqual(
                                differences(replica["optimized"], masters), []
                            )
                            self.assertEqual(
                                differences(replica, update["replicas"][0]), []
                            )

    def test_records_differ_in_length_and_supervised_length(self):
        """Otherwise every rank holds the same loss denominator."""
        with TemporaryDirectory() as temp:
            features = Path(temp)
            for index in range(3):
                record = {
                    "loss_mask": torch.tensor([0, 1, 1, 1, 1, 1]),
                    "hidden_states": torch.zeros(6, 2),
                }
                torch.save(record, features / str(index))
            _vary_records(features)
            records = [torch.load(features / str(index)) for index in range(3)]
            self.assertEqual(
                [
                    (len(record["hidden_states"]), int(record["loss_mask"].sum()))
                    for record in records
                ],
                [(6, 5), (5, 3), (4, 1)],
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

    def test_tolerance_is_not_rounded_to_the_tensor_dtype(self):
        zero = torch.zeros(1, dtype=torch.bfloat16)
        above = torch.tensor([0.00010013580322265625], dtype=torch.bfloat16)
        self.assertEqual(len(differences(above, zero, atol=0.0001)), 1)

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
