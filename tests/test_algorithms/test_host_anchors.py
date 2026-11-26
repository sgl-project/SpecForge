"""Block anchors drawn on the host: same selection law, keyed on the sample visit."""

import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from specforge.algorithms.objectives.base import ObjectiveResult
from specforge.algorithms.objectives.inputs import (
    anchor_count,
    host_anchor_inputs,
    prepare_blocks,
    sample_anchors,
    sample_seed,
    select_anchors,
)
from specforge.runtime.contracts import TrainBatch
from specforge.training.contracts import EPOCH_METADATA_KEY, EVALUATION_EPOCH
from specforge.training.controller import TrainerController
from specforge.training.runner import DataParallelRunner
from tests.test_algorithms.test_objective_contracts import build_runtime
from tests.test_algorithms.test_objectives import configurations, training_batch
from tests.test_modeling.dspine_fixtures import training_inputs, training_model
from tests.test_runtime._fixtures import FakeProgram, fake_spec
from tests.test_runtime.test_async_durable_ack import _Backend


def _batch(lengths, width=None):
    """input_ids / loss_mask rows of the given supervised lengths, right-padded."""
    width = width or max(lengths)
    mask = torch.zeros(len(lengths), width, dtype=torch.long)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1
    return {"input_ids": torch.randint(0, 30, mask.shape), "loss_mask": mask}


def _valid(batch):
    supervised = batch["loss_mask"].bool()
    return supervised[:, :-1] & supervised[:, 1:]


class SelectionTest(unittest.TestCase):
    def test_device_and_host_paths_share_the_selection(self):
        batch = _batch([9, 6])
        valid = _valid(batch)
        scores = torch.rand(valid.shape)
        with patch("torch.rand", return_value=scores):
            device = sample_anchors(valid, num_anchors=4)
        torch.testing.assert_close(
            device,
            select_anchors(valid, scores, anchor_count(valid, 4)),
            rtol=0,
            atol=0,
        )

    def test_host_anchors_are_the_selection_of_each_visits_scores(self):
        batch = _batch([9, 6])
        valid = _valid(batch)
        ids = ["a", "b"]
        scores = [
            np.random.Generator(np.random.PCG64(sample_seed(5, sample_id, 2))).random(
                valid.shape[1]
            )
            for sample_id in ids
        ]
        anchors, keep = select_anchors(
            valid,
            torch.from_numpy(np.stack(scores)),
            anchor_count(valid, 4),
            stable=True,
        )
        drawn = host_anchor_inputs(
            batch, num_anchors=4, seed=5, sample_ids=ids, epoch=2
        )
        self.assertTrue(torch.equal(drawn["anchors"], anchors))
        self.assertTrue(torch.equal(drawn["anchor_keep"], keep))

    def test_a_short_row_keeps_today_padding(self):
        drawn = host_anchor_inputs(
            _batch([9, 3]), num_anchors=4, seed=0, sample_ids=["a", "b"], epoch=0
        )
        # Row "b" has two valid anchor positions out of four requested.
        self.assertEqual(drawn["anchor_keep"].sum(-1).tolist(), [4, 2])
        self.assertTrue((drawn["anchors"][1][~drawn["anchor_keep"][1]] == 0).all())


class KeyTest(unittest.TestCase):
    def test_a_sample_draws_the_same_anchors_in_any_batch(self):
        torch.manual_seed(0)
        alone = _batch([10])
        # Second row of a batch whose first row pads it by twenty positions.
        together = _batch([30, 10])
        together["loss_mask"][1] = torch.cat(
            [alone["loss_mask"][0], torch.zeros(20, dtype=torch.long)]
        )
        first = host_anchor_inputs(
            alone, num_anchors=5, seed=3, sample_ids=["x"], epoch=1, draws_per_anchor=1
        )
        second = host_anchor_inputs(
            together,
            num_anchors=5,
            seed=3,
            sample_ids=["y", "x"],
            epoch=1,
            draws_per_anchor=1,
        )
        self.assertTrue(torch.equal(first["anchors"][0], second["anchors"][1]))
        self.assertTrue(
            torch.equal(first["anchor_draws"][0], second["anchor_draws"][1])
        )

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_a_loss_mask_on_the_training_device_is_read_on_the_host(self):
        # A pooled feature read places the loss mask on the device; the ids
        # and the normalizer's attention mask stay on the host.
        batch = _batch([12, 9])
        batch["attention_mask"] = torch.ones_like(batch["input_ids"], dtype=torch.bool)
        expected = host_anchor_inputs(
            batch, num_anchors=3, seed=0, sample_ids=["a", "b"], epoch=0
        )
        actual = host_anchor_inputs(
            {**batch, "loss_mask": batch["loss_mask"].cuda()},
            num_anchors=3,
            seed=0,
            sample_ids=["a", "b"],
            epoch=0,
        )
        self.assertTrue(torch.equal(actual["anchors"], expected["anchors"]))

    def test_each_epoch_and_each_seed_draws_anew(self):
        batch = _batch([40])
        draw = lambda seed, epoch: host_anchor_inputs(  # noqa: E731
            batch, num_anchors=6, seed=seed, sample_ids=["x"], epoch=epoch
        )["anchors"]
        self.assertTrue(torch.equal(draw(1, 0), draw(1, 0)))
        self.assertFalse(torch.equal(draw(1, 0), draw(1, 1)))
        self.assertFalse(torch.equal(draw(1, 0), draw(2, 0)))

    def test_the_seed_is_stable_and_separates_its_parts(self):
        self.assertEqual(sample_seed(42, "s", 0), sample_seed(42, "s", 0))
        self.assertNotEqual(sample_seed(4, "2s", 0), sample_seed(42, "s", 0))
        self.assertLess(sample_seed(42, "s", 0), 1 << 128)
        # 128 bits: these two ids share the low 32 bits of a 64-bit digest's
        # MT19937 seed under the old scheme; their streams now differ.
        draws = [
            np.random.Generator(np.random.PCG64(sample_seed(42, sample_id, 0))).random(
                4
            )
            for sample_id in ("specforge-run:00039032", "specforge-run:00093417")
        ]
        self.assertFalse(np.array_equal(*draws))

    def test_equal_scores_select_by_position_whatever_the_padding(self):
        valid = torch.tensor([[False, True, True, False, True]])
        scores = torch.tensor([[0.5, 0.25, 0.75, 0.5, 0.25]], dtype=torch.float64)
        padded_valid = torch.cat([valid, torch.zeros(1, 3, dtype=torch.bool)], 1)
        padded_scores = torch.cat([scores, torch.zeros(1, 3, dtype=torch.float64)], 1)
        for _ in range(20):
            short, _ = select_anchors(valid, scores, 1, stable=True)
            long, _ = select_anchors(padded_valid, padded_scores, 1, stable=True)
            self.assertEqual(short.tolist(), [[1]])
            self.assertEqual(long.tolist(), [[1]])

    def test_ids_must_match_the_rows(self):
        with self.assertRaisesRegex(ValueError, "sample ids"):
            host_anchor_inputs(
                _batch([5, 5]), num_anchors=2, seed=0, sample_ids=["a"], epoch=0
            )


class LawTest(unittest.TestCase):
    def test_every_valid_position_is_equally_likely_on_both_paths(self):
        # Ten valid anchor positions, three anchors: each position has p = 0.3.
        batch = _batch([11])
        valid = _valid(batch)
        trials = 4000
        host = torch.zeros(valid.shape[1])
        device = torch.zeros(valid.shape[1])
        torch.manual_seed(1)
        for trial in range(trials):
            drawn = host_anchor_inputs(
                batch, num_anchors=3, seed=trial, sample_ids=["x"], epoch=0
            )
            host[drawn["anchors"][0]] += 1
            anchors, _ = sample_anchors(valid, 3)
            device[anchors[0]] += 1
        # Binomial(4000, 0.3) has SD 29; six SDs is about 0.044 in frequency.
        for counts in (host, device):
            torch.testing.assert_close(
                counts / trials, torch.full_like(counts, 0.3), rtol=0, atol=0.045
            )

    def test_every_subset_is_equally_likely_on_both_paths(self):
        # Five valid anchor positions, two anchors: ten subsets, p = 0.1 each.
        # Equal marginals alone would not show a sampler that skips subsets.
        batch = _batch([6])
        valid = _valid(batch)
        trials = 5000
        host, device = {}, {}
        torch.manual_seed(2)
        for trial in range(trials):
            drawn = host_anchor_inputs(
                batch, num_anchors=2, seed=trial, sample_ids=["x"], epoch=0
            )
            key = tuple(drawn["anchors"][0].tolist())
            host[key] = host.get(key, 0) + 1
            key = tuple(sample_anchors(valid, 2)[0][0].tolist())
            device[key] = device.get(key, 0) + 1
        # Binomial(5000, 0.1) has SD 21; six SDs is about 0.025 in frequency.
        for counts in (host, device):
            self.assertEqual(len(counts), 10)
            for count in counts.values():
                self.assertLess(abs(count / trials - 0.1), 0.025)


class BlocksTest(unittest.TestCase):
    def test_prepare_blocks_uses_host_anchors_as_given(self):
        config, runtime = build_runtime("DFlashDraftModel")
        batch = _batch([12, 9])
        batch["hidden_states"] = torch.randn(2, 12, 8)
        drawn = host_anchor_inputs(
            batch, num_anchors=3, seed=0, sample_ids=["a", "b"], epoch=0
        )
        with patch(
            "specforge.algorithms.objectives.inputs.sample_anchors",
            return_value=(drawn["anchors"], drawn["anchor_keep"]),
        ):
            expected = prepare_blocks(config, batch, num_anchors=3)
        actual = prepare_blocks(config, {**batch, **drawn}, num_anchors=3)
        for name in ("anchors", "keep", "labels", "label_positions", "weights"):
            self.assertTrue(torch.equal(getattr(actual, name), getattr(expected, name)))


class ObjectiveModelTest(unittest.TestCase):
    def test_host_sampling_is_for_block_objectives_only(self):
        with self.assertRaisesRegex(ValueError, "block objectives only"):
            build_runtime("Eagle3DraftModel", anchor_sampling="host", anchor_seed=0)

    def test_host_sampling_needs_a_seed_and_the_epoch(self):
        with self.assertRaisesRegex(ValueError, "anchor_seed"):
            build_runtime("DFlashDraftModel", anchor_sampling="host")
        _, runtime = build_runtime(
            "DFlashDraftModel", anchor_sampling="host", anchor_seed=0
        )
        batch = _batch([12, 9])
        with self.assertRaisesRegex(ValueError, "epoch"):
            runtime.host_inputs(batch, sample_ids=["a", "b"], epoch=None)
        drawn = runtime.host_inputs(batch, sample_ids=["a", "b"], epoch=0)
        self.assertEqual(
            drawn["anchors"].shape[1], min(runtime.objective.num_anchors, 11)
        )

    def test_a_host_model_refuses_a_batch_without_its_draws(self):
        _, runtime = build_runtime(
            "DFlashDraftModel", anchor_sampling="host", anchor_seed=0
        )
        config = runtime.draft_model.config.draft
        batch = training_batch(config)
        with self.assertRaisesRegex(ValueError, "host_inputs"):
            runtime(batch, step=0, total_steps=10)
        model = training_model(chunk_size=1, anchor_sampling="host", anchor_seed=0)
        inputs = training_inputs()
        drawn = model.host_inputs(inputs, sample_ids=["a", "b"], epoch=0)
        del drawn["anchor_draws"]
        with self.assertRaisesRegex(ValueError, "anchor_draws"):
            model({**inputs, **drawn}, step=0, total_steps=100)

    def test_refusal_comes_before_the_draft_is_changed(self):
        from specforge.algorithms.objectives import build_objective_model
        from specforge.modeling import build_draft_model

        config = next(
            c for c in configurations() if c.architecture == "DFlashDraftModel"
        )
        draft = build_draft_model(config)
        trainable = {n for n, p in draft.named_parameters() if p.requires_grad}
        with self.assertRaisesRegex(ValueError, "anchor_seed"):
            build_objective_model(
                draft, frozen=("layers",), anchor_sampling="host", anchor_seed=None
            )
        self.assertEqual(
            {n for n, p in draft.named_parameters() if p.requires_grad}, trainable
        )

    def test_device_sampling_draws_nothing_on_the_host(self):
        _, runtime = build_runtime("DFlashDraftModel")
        drawn = runtime.host_inputs(_batch([12, 9]), sample_ids=["a", "b"], epoch=0)
        self.assertNotIn("anchors", drawn)

    def test_dspine_replaces_from_the_host_draws(self):
        inputs = training_inputs()
        masks = []

        def record(_module, _args, kwargs):
            masks.append(kwargs["replace_masks"][0].clone())

        model = training_model(
            chunk_size=1,
            replacement_probability=0.5,
            anchor_sampling="host",
            anchor_seed=0,
        )
        model.draft_model.register_forward_pre_hook(record, with_kwargs=True)
        drawn = model.host_inputs(inputs, sample_ids=["a", "b"], epoch=0)
        for value, expected in ((0.0, True), (0.99, False)):
            draws = torch.full_like(drawn["anchor_draws"], value)
            with torch.no_grad():
                model(
                    {**inputs, **drawn, "anchor_draws": draws},
                    step=0,
                    total_steps=100,
                )
            replaced = masks[-1]
            keep = drawn["anchor_keep"]
            self.assertTrue(bool((replaced[keep] == expected).all()))
            self.assertFalse(bool(replaced[~keep].any()))


class _Recorder(FakeProgram):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))
        self.seen = []

    def host_inputs(self, tensors, *, sample_ids, epoch):
        self.seen.append((tuple(sample_ids), epoch))
        return {}

    def forward(self, tensors, *, step, total_steps, collect_detailed_metrics, **_):
        loss = (self.w * tensors["x"].sum()).abs()
        one = torch.ones(())
        return ObjectiveResult(loss, one, ratios={"accuracy": (torch.tensor(0.5), one)})


class RunLoopTest(unittest.TestCase):
    def test_each_batch_carries_the_epoch_it_was_pulled_in(self):
        # Three batches an epoch, windows of two: the second window spans the
        # epoch boundary, and each of its batches keeps its own epoch.
        program = _Recorder()
        stream = [
            TrainBatch([f"s{index}"], "fake", {"x": torch.ones(1)})
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as directory:
            controller = TrainerController(
                DataParallelRunner(program, _Backend(program), fake_spec("x")),
                strategy_name="test",
                accumulation_steps=2,
                run_id="run",
                output_dir=directory,
                num_epochs=2,
                rank=0,
                world_size=1,
            )
            controller.fit(stream)
        self.assertEqual(
            program.seen,
            [(("s0",), 0), (("s1",), 0), (("s2",), 0)]
            + [(("s0",), 1), (("s1",), 1), (("s2",), 1)],
        )
        # The loader's own batches are not changed.
        self.assertNotIn(EPOCH_METADATA_KEY, stream[0].metadata)

    def test_a_resumed_loop_stamps_the_epoch_it_resumes_in(self):
        # Resumed in the second epoch after one batch: the rest of that epoch
        # carries epoch 1, as it would have without the interruption.
        program = _Recorder()
        stream = [
            TrainBatch([f"s{index}"], "fake", {"x": torch.ones(1)})
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as directory:
            controller = TrainerController(
                DataParallelRunner(program, _Backend(program), fake_spec("x")),
                strategy_name="test",
                accumulation_steps=2,
                run_id="run",
                output_dir=directory,
                num_epochs=2,
                rank=0,
                world_size=1,
            )
            controller.epoch, controller._epoch_batch = 1, 1
            controller.fit(stream)
        self.assertEqual(program.seen, [(("s1",), 1), (("s2",), 1)])

    def test_evaluation_draws_with_one_epoch_for_every_pass(self):
        program = _Recorder()
        runner = DataParallelRunner(program, _Backend(program), fake_spec("x"))
        batch = TrainBatch(["e0"], "fake", {"x": torch.ones(1)})
        from specforge.training.contracts import StepContext

        with runner.evaluation():
            runner.evaluate(batch, StepContext())
        self.assertEqual(program.seen, [(("e0",), EVALUATION_EPOCH)])

    def test_dspine_host_sampling_end_to_end_replay(self):
        from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
        from specforge.training.program import program_spec

        model = training_model(
            chunk_size=1,
            replacement_probability=0.5,
            anchor_sampling="host",
            anchor_seed=11,
        )
        spec = program_spec(model)
        self.assertTrue(spec.replay_statistics)
        backend = FSDPTrainingBackend(
            ParallelConfig(),
            optimizer_factory=lambda draft: torch.optim.SGD(
                draft.parameters(), lr=0.05
            ),
        )
        wrapped = backend.prepare_model(
            model,
            wrap=False,
            optimizer_target=model.draft_model,
            frozen_modules=model.frozen_modules,
        )
        runner = DataParallelRunner(wrapped, backend, spec)

        seen_passes = []
        orig_forward = model.forward

        def record_forward(tensors, **kwargs):
            seen_passes.append(
                (
                    bool(kwargs.get("prepass", False)),
                    tensors["anchors"].clone(),
                    tensors["anchor_draws"].clone(),
                )
            )
            return orig_forward(tensors, **kwargs)

        torch.manual_seed(0)
        b0 = training_inputs()
        b1 = training_inputs()
        stream = [
            TrainBatch(["s0", "s1"], "dspine", b0),
            TrainBatch(["s2", "s3"], "dspine", b1),
        ]
        before = [
            p.detach().clone()
            for p in model.draft_model.parameters()
            if p.requires_grad
        ]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(model, "forward", side_effect=record_forward),
        ):
            controller = TrainerController(
                runner,
                strategy_name="dspine",
                accumulation_steps=2,
                run_id="dspine-host",
                output_dir=directory,
                num_epochs=1,
                max_steps=1,
                rank=0,
                world_size=1,
            )
            self.assertEqual(controller.fit(stream), 1)
        # Two micro-batches in prepass, then the exact same two in forward_backward.
        self.assertEqual(
            [is_pre for is_pre, _, _ in seen_passes], [True, True, False, False]
        )
        for mb in range(2):
            self.assertTrue(torch.equal(seen_passes[mb][1], seen_passes[mb + 2][1]))
            self.assertTrue(torch.equal(seen_passes[mb][2], seen_passes[mb + 2][2]))
        after = [p.detach() for p in model.draft_model.parameters() if p.requires_grad]
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, after)))

    def test_host_anchor_sampling_checkpoint_resume_contract(self):
        from pathlib import Path

        from tests.test_runtime import test_checkpoint_resume as resume_tests

        harness = resume_tests.TestTrainerResumeEntrypoint()
        if not torch.distributed.is_initialized():
            harness.setUpClass()
            self.addCleanup(harness.tearDownClass)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = resume_tests._write_feature_files(str(root / "features"), n=4)
            trainer, _, _ = harness._make_trainer(
                str(root / "run"),
                feat_dir=features,
                max_steps=1,
                checkpoint_extra={"canonical_anchor_sampling": ("host", 42)},
            )
            self.assertEqual(trainer.fit(), 1)
            source = str((root / "run" / "rz-latest").resolve())

            # Matching ("host", 42) resumes cleanly.
            resumed, _, _ = harness._make_trainer(
                str(root / "run"),
                feat_dir=features,
                max_steps=2,
                checkpoint_extra={"canonical_anchor_sampling": ("host", 42)},
                resume_from=source,
            )
            self.assertEqual(resumed.fit(), 2)

            # Changed seed or switching back to "device" is rejected.
            for mismatch in (("host", 99), "device"):
                with self.assertRaisesRegex(ValueError, "canonical_anchor_sampling"):
                    harness._make_trainer(
                        str(root / "run"),
                        feat_dir=features,
                        max_steps=2,
                        checkpoint_extra={"canonical_anchor_sampling": mismatch},
                        resume_from=source,
                    )


if __name__ == "__main__":
    unittest.main()
