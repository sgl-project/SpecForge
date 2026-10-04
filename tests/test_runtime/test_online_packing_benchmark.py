"""Fairness checks for the overlapping online packing benchmark."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from specforge.benchmarks.benchmark_online_sequence_packing import (
    _observe_first_warmup_step,
    _optimizer_coverage,
    _optimizer_step_evidence,
    _parameter_inventory,
    _parameter_sample_indices,
    _parameter_samples,
    _parameter_update_evidence,
    _prompts,
    _TimedSource,
    parse_args,
)
from specforge.launch import _iter_epoch_online_prompt_batches
from specforge.optimizer import BF16Optimizer


class OnlinePackingBenchmarkTests(unittest.TestCase):
    def _model(self):
        model = torch.nn.Module()
        model.draft_model = torch.nn.Module()
        model.draft_model.layers = torch.nn.ModuleList(
            [torch.nn.Linear(2, 2, bias=False) for _ in range(2)]
        )
        # This factor has a legitimate zero gradient and no first-step change.
        model.draft_model.zero_factor = torch.nn.Parameter(torch.zeros(2))
        model.embed_tokens = torch.nn.Embedding(4, 2).requires_grad_(False)
        model.lm_head = torch.nn.Linear(2, 4, bias=False)
        model.lm_head.weight = model.embed_tokens.weight
        return model

    def _args(self, *extra):
        return parse_args(
            [
                "--server-url",
                "http://localhost:31012",
                "--target-model",
                "unused",
                "--work-dir",
                "unused",
                "--output",
                "unused.json",
                *extra,
            ]
        )

    def test_default_warmup_replays_full_corpus(self):
        args = self._args("--steps", "7")
        self.assertEqual(args.warmup_steps, 7)
        self.assertTrue(args.teacher_metrics)
        self.assertEqual(args.log_interval, 50)
        self.assertEqual(args.objective_chunk_blocks, 128)

    def test_sample_indices_keep_large_embedding_endpoint_in_bounds(self):
        # Only allocate <=128 indices, never the 389-million-element embedding.
        for numel in (0, 1, 2, 127, 128, 129, 388956160, 2**40):
            with self.subTest(numel=numel):
                indices = _parameter_sample_indices(numel)
                self.assertEqual(indices.dtype, torch.int64)
                self.assertEqual(indices.numel(), min(128, numel))
                if numel:
                    self.assertEqual(indices[0].item(), 0)
                    self.assertEqual(indices[-1].item(), numel - 1)
                    self.assertTrue(bool(((indices >= 0) & (indices < numel)).all()))
                    self.assertTrue(bool((indices[1:] > indices[:-1]).all()))

    def test_inventory_counts_tied_target_once_and_checks_optimizer_identity(self):
        model = self._model()
        named, inventory = _parameter_inventory(model)
        self.assertEqual(inventory["draft_layer_count"], 2)
        self.assertEqual(inventory["whole_model_unique"]["total"], 18)
        self.assertEqual(inventory["whole_model_unique"]["trainable"], 10)
        self.assertEqual(inventory["whole_model_unique"]["frozen"], 8)
        self.assertEqual(inventory["components"]["target_embedding"]["total"], 8)
        self.assertEqual(inventory["components"]["target_lm_head"]["total"], 8)
        tied = next(row for row in inventory["parameters"] if not row["trainable"])
        self.assertEqual(
            set(tied["aliases"]), {"embed_tokens.weight", "lm_head.weight"}
        )
        optimizer = BF16Optimizer(model.draft_model, lr=0.01, total_steps=2)
        coverage = _optimizer_coverage(model, named, optimizer)
        self.assertEqual(coverage["optimized_elements"], 10)
        self.assertTrue(coverage["target_parameters_excluded"])
        optimizer.model_params[-1] = model.embed_tokens.weight
        with self.assertRaisesRegex(AssertionError, "no target Parameter"):
            _optimizer_coverage(model, named, optimizer)

    def test_full_update_evidence_accepts_legitimate_zero_gradient_parameters(self):
        model = self._model()
        named, _ = _parameter_inventory(model)
        before = _parameter_samples(named)
        optimizer = BF16Optimizer(model.draft_model, lr=0.01, total_steps=2)
        gradient_evidence = _observe_first_warmup_step(optimizer, named)
        sum(
            parameter.square().sum() for parameter in model.draft_model.parameters()
        ).backward()
        optimizer.step()
        self.assertTrue(gradient_evidence["performed"])
        self.assertTrue(all(gradient_evidence["gradient_present"].values()))
        self.assertTrue(torch.isfinite(gradient_evidence["_gradient_norm"]))
        self.assertTrue(
            all(parameter.grad is None for parameter in optimizer.model_params)
        )
        updates = _parameter_update_evidence(named, before, 2)
        self.assertTrue(updates["all_decoder_layers_have_sampled_updates"])
        self.assertTrue(updates["all_trainable_elements_finite"])
        self.assertTrue(updates["all_frozen_parameter_samples_unchanged"])
        self.assertFalse(
            updates["parameters"]["draft_model.zero_factor"]["sampled_update_observed"]
        )
        steps = _optimizer_step_evidence(optimizer, named, 1)
        self.assertTrue(steps["all_trainable_parameters_received_every_optimizer_step"])
        self.assertEqual(
            steps["adamw_steps_by_parameter"]["draft_model.zero_factor"], 1
        )

    def test_evidence_rejects_nonfinite_parameters_and_changed_frozen_samples(self):
        model = self._model()
        named, _ = _parameter_inventory(model)
        before = _parameter_samples(named)
        with torch.no_grad():
            model.draft_model.zero_factor[0] = float("inf")
        with self.assertRaisesRegex(AssertionError, "nonfinite"):
            _parameter_update_evidence(named, before, 2)
        with torch.no_grad():
            model.draft_model.zero_factor[0] = 0
            model.embed_tokens.weight[0, 0] += 1
        with self.assertRaisesRegex(AssertionError, "frozen"):
            _parameter_update_evidence(named, before, 2)

    def test_real_masks_and_batch_order_survive_canonical_shuffle(self):
        rows = [
            {
                "input_ids": [index] * (index + 4),
                "loss_mask": [0] * (index + 2) + [1, 1],
            }
            for index in range(8)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows))
            args = self._args("--prompts-path", str(path), "--steps", "2")
            prompts, digest, tokens, supervised = _prompts(args, 2, 16)
            ordered = [
                prompt
                for batch in _iter_epoch_online_prompt_batches(
                    prompts, 0, 1, seed=args.seed, batch_size=3
                )
                for prompt in batch
            ]
            self.assertEqual([prompt["payload"] for prompt in ordered], rows)
            self.assertEqual(
                [prompt["task_id"] for prompt in ordered],
                [f"prompt-{index:08d}" for index in range(8)],
            )
            self.assertEqual(tokens, sum(len(row["input_ids"]) for row in rows))
            self.assertEqual(supervised, 16)
            self.assertEqual(_prompts(args, 2, 16)[1], digest)
            # Warmup prefixes use the same input order despite a different
            # shuffle length, making short debugging runs deterministic too.
            warm = _prompts(args, 1, 16)[0]
            warm_ordered = [
                prompt
                for batch in _iter_epoch_online_prompt_batches(
                    warm, 0, 1, seed=args.seed
                )
                for prompt in batch
            ]
            self.assertEqual([prompt["payload"] for prompt in warm_ordered], rows[:4])

    def test_request_hash_checks_masks_but_ignores_transport_namespaces(self):
        def capture(run, mask):
            sent = []
            adapter = SimpleNamespace(post_fn=lambda url, **kw: sent.append(kw) or [])
            source = _TimedSource(adapter)
            adapter.post_fn(
                "http://localhost/generate",
                timeout=1,
                json_body={
                    "input_ids": [[1, 2, 3]],
                    "extra_key": [run],
                    "sampling_params": {"max_new_tokens": 0},
                    "spec_capture": [
                        {
                            "store_id": run,
                            "sample_id": f"{run}:prompt-00000000",
                            "passthrough": [{"data": mask}],
                        }
                    ],
                },
            )
            self.assertEqual(len(sent), 1)
            self.assertEqual(source.calls[0]["samples"], 1)
            self.assertLessEqual(source.first_dispatch, source.calls[0]["end"])
            return source.request_digest.hexdigest()

        self.assertEqual(capture("run-a", [0, 1, 1]), capture("run-b", [0, 1, 1]))
        self.assertNotEqual(capture("run-a", [0, 1, 1]), capture("run-b", [1, 1, 1]))


if __name__ == "__main__":
    unittest.main()
