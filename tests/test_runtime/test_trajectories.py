"""A recorded training trajectory is reproducible and pins the optimizer window."""

import copy
import math
import pickle
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from specforge.lr_scheduler import CosineAnnealingWarmupLR
from specforge.training.checkpoint import STATE_FILE
from tests.test_runtime.test_canonical_end_to_end import (
    LEARNING_RATE,
    drafts,
    recomputed_tolerance,
    write_run,
)
from tests.test_runtime.trajectories import (
    WARMUP_RATIO,
    Case,
    _evaluation,
    _logged,
    _read_checkpoint,
    _vary_records,
    cases,
    differences,
    main,
    record,
)


def _is_ddp(case):
    """The reference runtime replicating: DDP, whose replicas the recorder sees."""
    return case.runtime == "fsdp" and case.sharding == "NO_SHARD"


def _scheduled_states(case):
    optimizer = torch.optim.SGD([torch.zeros(())], lr=LEARNING_RATE)
    scheduler = CosineAnnealingWarmupLR(
        optimizer, case.updates, warmup_steps=int(case.updates * WARMUP_RATIO)
    )
    for _ in range(case.updates):
        optimizer.step()
        scheduler.step()
        yield scheduler.state_dict()


def _optimizer_differences(saved, replica, scheduler_state):
    expected = {
        "fp32_params": replica["optimized"],
        "optimizer_state_dict": replica["optimizer"],
        "scheduler_state_dict": scheduler_state,
    }
    actual = {key: saved[key] for key in expected if key in saved}
    return differences(actual, expected, where="optimizer")


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
        self.assertTrue(any(case.batch_size > 1 for case in self.cases))
        for case in self.cases:
            with self.subTest(case=case.name):
                positions = [
                    divmod(step * case.window_size - 1, case.batches_per_epoch)
                    for step in range(1, case.updates + 1)
                ]
                self.assertEqual(
                    [
                        (
                            update["checkpoint"],
                            update["shared"]["global_step"],
                            update["shared"]["epoch"],
                            update["shared"]["epoch_batch"],
                            update["shared"]["epoch_samples"],
                        )
                        for update in self.first[case.name]
                    ],
                    [
                        (step, step, epoch, batch + 1, (batch + 1) * case.batch_size)
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
                    self.assertTrue(math.isfinite(loss))
                if case.empty_eval_rank is not None:
                    # Both runs start from the same weights and training data.
                    # Ignoring the empty shard override would make these equal.
                    full = replace(case, empty_eval_rank=None)
                    self.assertNotEqual(
                        self.first[case.name][0]["evaluated"][0]["eval/avg_loss"],
                        self.first[full.name][0]["evaluated"][0]["eval/avg_loss"],
                    )

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
                if not _is_ddp(case):
                    continue
                with self.subTest(case=case.name):
                    for update, schedule in zip(
                        recording[case.name], _scheduled_states(case), strict=True
                    ):
                        shared = update["shared"]
                        saved = shared["draft_state_dict"]
                        saved_optimizer = shared["replicated_optimizer_state"]
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
                                _optimizer_differences(
                                    saved_optimizer, replica, schedule
                                ),
                                [],
                            )
                            self.assertEqual(
                                differences(replica, update["replicas"][0]), []
                            )

    def test_saved_optimizer_corruption_is_detected(self):
        case = next(case for case in self.cases if _is_ddp(case))
        update = self.first[case.name][0]
        saved = update["shared"]["replicated_optimizer_state"]
        replica = update["replicas"][0]
        schedule = next(_scheduled_states(case))
        self.assertEqual(_optimizer_differences(saved, replica, schedule), [])

        state_key = next(iter(saved["optimizer_state_dict"]["state"]))
        for path in (
            ("optimizer_state_dict", "state", state_key, "exp_avg"),
            ("optimizer_state_dict", "param_groups", 0, "lr"),
            ("scheduler_state_dict", "last_epoch"),
        ):
            with self.subTest(field=path):
                changed = copy.deepcopy(saved)
                parent = changed
                for key in path[:-1]:
                    parent = parent[key]
                parent[path[-1]] += 1
                found = _optimizer_differences(changed, replica, schedule)
                self.assertEqual(len(found), 1)
                self.assertIn(f".{path[-1]}:", found[0])

    def test_every_checkpoint_records_offline_resume_metadata(self):
        required = {"sampler_seed", "offline_sampler_version", "source_dataset_size"}
        for case, updates in self.first.items():
            for update in updates:
                self.assertLessEqual(required, update["shared"].keys(), case)

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


class CheckpointRecordingTest(unittest.TestCase):
    def test_compares_offline_resume_metadata(self):
        metadata = {
            "sampler_seed": 42,
            "offline_sampler_version": 1,
            "source_dataset_size": 6,
        }
        with TemporaryDirectory() as temp:
            root = Path(temp)
            torch.save({}, root / "training_state_rank0.pt")
            torch.save(metadata, root / STATE_FILE)
            reference = _read_checkpoint(root, 1)
            self.assertEqual(reference["shared"], metadata)
            for key, value in metadata.items():
                with self.subTest(field=key):
                    torch.save(metadata | {key: value + 1}, root / STATE_FILE)
                    self.assertEqual(
                        differences(_read_checkpoint(root, 1), reference),
                        [f"recording.shared.{key}: {value + 1} != {value}"],
                    )
            del metadata["sampler_seed"]
            torch.save(metadata, root / STATE_FILE)
            self.assertEqual(
                differences(_read_checkpoint(root, 1), reference),
                ["recording.shared: keys differ in ['sampler_seed']"],
            )

    def test_compare_loads_only_tensor_and_primitive_data(self):
        with TemporaryDirectory() as temp:
            plain, unsupported = Path(temp) / "plain.pt", Path(temp) / "date.pt"
            torch.save({"case": [torch.tensor(1), 2.0]}, plain)
            torch.save({"case": date(2020, 1, 1)}, unsupported)
            self.assertEqual(main(["compare", str(plain), str(plain)]), 0)
            for args in [(plain, unsupported), (unsupported, plain)]:
                with self.subTest(args=args), self.assertRaises(pickle.UnpicklingError):
                    main(["compare", *map(str, args)])


class EvaluationFixtureTest(unittest.TestCase):
    def test_eval_features_are_independent_and_one_rank_can_have_no_samples(self):
        case = Case("DFlashDraftModel", world_size=2, evaluated=True, empty_eval_rank=1)
        draft = drafts()[0]
        with TemporaryDirectory() as temp:
            root = Path(temp)
            write_run(root, draft, records=6)
            _vary_records(root / "features")
            overrides, rank_overrides = _evaluation(root, case, draft)
            eval_dir = root / "evaluation"
            self.assertIn(f"data.eval_hidden_states_path={eval_dir}", overrides)
            files = sorted(eval_dir.glob("*.ckpt"))
            self.assertEqual(len(files), 4)
            for path in files:
                trained = torch.load(root / "features" / path.name)
                evaluated = torch.load(path)
                self.assertFalse(
                    torch.equal(trained["hidden_states"], evaluated["hidden_states"])
                )
            empty = root / "empty-evaluation"
            self.assertEqual(
                rank_overrides, {1: [f"data.eval_hidden_states_path={empty}"]}
            )
            self.assertEqual(list(empty.iterdir()), [])


class DifferencesTest(unittest.TestCase):
    @staticmethod
    def logged_update(train_metric, eval_metric, lr=0.5):
        case = Case("Eagle3DraftModel", evaluated=True)
        events = []
        for step in range(1, case.updates + 1):
            events.append((step, {"lr": lr, "acc": train_metric}))
            events.append((step, {"eval/per_position_acc": eval_metric}))
        trained, evaluated = _logged(events, case)
        return {"checkpoint": 1, "logged": [trained[1]], "evaluated": [evaluated[1]]}

    def test_logging_preserves_scalar_and_sequence_types(self):
        reference = self.logged_update(1.0, [0.5])
        changed = self.logged_update(True, torch.tensor([0.5], dtype=torch.float64))
        for atol in (0.0, 0.1):
            self.assertEqual(
                differences(changed, reference, atol=atol),
                [
                    "recording.logged[0].metrics.acc: bool != float",
                    "recording.evaluated[0].eval/per_position_acc: Tensor != list",
                ],
            )

    def test_log_step_types_cannot_alias_integer_checkpoint_numbers(self):
        case = Case("DFlashDraftModel")
        events = [(step, {"lr": 0.5}) for step in range(1, case.updates + 1)]
        for first_step in (True, 1.0):
            with (
                self.subTest(step_type=type(first_step)),
                self.assertRaisesRegex(TypeError, "logger step must be int"),
            ):
                _logged([(first_step, events[0][1]), *events[1:]], case)

    def test_tolerance_applies_to_logged_metrics_and_keeps_rates_exact(self):
        reference = self.logged_update(0.5, [0.5])
        near = self.logged_update(0.51, [0.51], lr=0.51)
        self.assertEqual(len(differences(near, reference)), 3)
        self.assertEqual(
            differences(near, reference, atol=0.02),
            ["recording.logged[0].lr: 0.51 != 0.5"],
        )

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
