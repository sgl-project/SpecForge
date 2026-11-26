"""Canonical EAGLE3 runs a configured objective step on each declared backend."""

import unittest
import warnings
from dataclasses import replace
from unittest.mock import patch

import torch
from torch import nn

from specforge.algorithms.objectives import build_objective_model
from specforge.algorithms.objectives.registry import (
    protocol_registrations,
    resolve_objective,
)
from specforge.modeling import build_draft_model
from specforge.training.sequence_parallel import SequenceParallel
from tests.test_algorithms.test_objectives import eagle_config, training_batch
from tests.test_modeling.test_canonical_eagle3 import install_mapping

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Ranks(str):
    """Stands in for a process group: its name, and how many ranks it holds."""

    def __new__(cls, name, size=2):
        group = super().__new__(cls, name)
        group.ranks = size
        return group

    def size(self):
        return self.ranks


def sequence_groups(size=2):
    return SequenceParallel("ring", "ulysses", Ranks("sequence", size))


def objective_model(backend, **selection):
    config = eagle_config()
    # FlexAttention's CUDA kernel needs head dimensions of at least 16.
    config = replace(
        config,
        attention_configs={
            name: replace(node, head_dim=16)
            for name, node in config.attention_configs.items()
        },
    )
    torch.manual_seed(0)
    model = build_draft_model(config, backend=backend, device=DEVICE)
    install_mapping(model)
    torch.manual_seed(1)
    head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(DEVICE)
    selection.setdefault("attention_backend", backend)
    runtime = build_objective_model(model, target_head=head, **selection)
    return config, runtime


def forward_loss(config, runtime):
    torch.manual_seed(2)
    batch = training_batch(config, device=DEVICE)
    with torch.no_grad():
        return runtime(batch, step=0, total_steps=1).loss


class EagleBackendTest(unittest.TestCase):
    def test_dense_backends_agree_through_the_objective(self):
        declared = protocol_registrations()["eagle3"].training.attention_backends
        self.assertEqual(declared, {"eager", "sdpa", "flex_attention", "fa", "usp"})
        # Compiled FlexAttention only runs on CUDA.
        backends = ["eager", "sdpa"] + ["flex_attention"] * torch.cuda.is_available()
        losses = {
            backend: forward_loss(*objective_model(backend)) for backend in backends
        }
        for backend in backends[1:]:
            with self.subTest(backend=backend):
                torch.testing.assert_close(
                    losses[backend], losses["eager"], rtol=1e-4, atol=1e-5
                )

    def test_flex_runs_through_the_cached_rollout(self):
        # The uncached branch builds a tensor visibility without active
        # queries, which Flex preparation rejects; the rollout supplies both.
        config, runtime = objective_model("sdpa", attention_backend="flex_attention")
        model = runtime.draft_model
        original, requested = model.new_rollout, []

        def new_rollout(backend, **groups):
            requested.append(backend)
            return original("sdpa", **groups)  # stands in for Flex off CUDA

        with patch.object(model, "new_rollout", new_rollout):
            loss = forward_loss(config, runtime)
        self.assertEqual(requested, ["flex_attention"])
        torch.testing.assert_close(
            loss, forward_loss(*objective_model("sdpa")), rtol=1e-5, atol=1e-6
        )

    def test_rollout_takes_the_groups_the_model_was_built_with(self):
        def requested_groups(backend, **construction):
            config, runtime = objective_model(
                "sdpa", attention_backend=backend, **construction
            )
            model = runtime.draft_model
            original, requested = model.new_rollout, []

            def new_rollout(backend, **groups):
                requested.append(groups)
                return original("sdpa")  # stands in for the backend off CUDA

            with (
                patch.object(model, "new_rollout", new_rollout),
                patch(
                    "specforge.training.sequence_parallel.reduce_statistics",
                    lambda accepted, count, denominator, *, group: (
                        accepted,
                        count,
                        denominator,
                    ),
                ),
            ):
                forward_loss(config, runtime)
            return requested

        self.assertEqual(
            requested_groups("flex_attention"),
            [{"ring_group": None, "ulysses_group": None}],
        )
        self.assertEqual(
            requested_groups("usp", sequence_parallel=sequence_groups()),
            [{"ring_group": "ring", "ulysses_group": "ulysses"}],
        )

    def test_sequence_parallel_groups_are_read_once_from_the_process(self):
        from specforge.training import sequence_parallel

        with (
            patch("specforge.distributed.get_sp_ring_group", return_value="ring") as a,
            patch(
                "specforge.distributed.get_sp_ulysses_group", return_value="ulysses"
            ) as b,
            patch(
                "specforge.distributed.get_draft_sp_group", return_value="draft"
            ) as c,
            patch.object(
                sequence_parallel, "reduce_statistics", return_value="reduced"
            ) as reduce,
        ):
            groups = sequence_parallel.sequence_parallel_groups()
            self.assertEqual(groups.reduce_statistics(1, 2, 3), "reduced")
            self.assertEqual(groups.reduce_statistics(4, 5, 6), "reduced")
        self.assertEqual(
            (groups.ring_group, groups.ulysses_group, groups.sequence_group),
            ("ring", "ulysses", "draft"),
        )
        # A sequence is split over the ranks of the product group.
        self.assertEqual(sequence_groups(size=4).size, 4)
        # Statistics of one sequence are reduced over the product group.
        reduce.assert_called_with(4, 5, 6, group="draft")
        for getter in (a, b, c):
            getter.assert_called_once_with()

    def test_missing_sequence_group_is_refused(self):
        """A statistic reduced over no group would be pooled over every sample."""
        from specforge.training import sequence_parallel

        with (
            patch("specforge.distributed.get_sp_ring_group", return_value="ring"),
            patch("specforge.distributed.get_sp_ulysses_group", return_value="ulysses"),
            patch("specforge.distributed.get_draft_sp_group", return_value=None),
            self.assertRaisesRegex(ValueError, "without its sequence_group"),
        ):
            sequence_parallel.sequence_parallel_groups()
        # However the value is built.
        with self.assertRaisesRegex(ValueError, "without its ring_group"):
            SequenceParallel(None, "ulysses", "sequence")

    def test_objective_refuses_a_backend_or_axis_it_does_not_declare(self):
        from tests.test_algorithms.test_objective_contracts import build_runtime

        with self.assertRaisesRegex(ValueError, "does not train with attention.*'usp'"):
            build_runtime("PEagleDraftModel", attention_backend="usp")
        groups = sequence_groups()
        with self.assertRaisesRegex(ValueError, "does not support sequence parallel"):
            build_runtime("PEagleDraftModel", sequence_parallel=groups)
        # What the CLI refuses, the builder refuses.
        with self.assertRaisesRegex(ValueError, "needs the sequence-parallel groups"):
            objective_model("sdpa", attention_backend="usp")
        # The groups come with the USP backend and with no other: a backend
        # that does not split the sequence would still reduce over them.
        for backend in ("sdpa", "flex_attention", None):
            with (
                self.subTest(backend=backend),
                self.assertRaisesRegex(ValueError, "need attention backend 'usp'"),
            ):
                objective_model(
                    "sdpa", attention_backend=backend, sequence_parallel=groups
                )
        with self.assertRaisesRegex(ValueError, "D-PACE weighting is not supported"):
            objective_model(
                "sdpa",
                attention_backend="usp",
                sequence_parallel=groups,
                weighting={"kind": "dpace"},
            )
        # As the CLI does, the builder takes D-PACE while the sequence is whole.
        whole = objective_model(
            "sdpa",
            attention_backend="usp",
            sequence_parallel=sequence_groups(size=1),
            weighting={"kind": "dpace"},
        )[1]
        self.assertEqual(whole.weighting.kind, "dpace")

    def test_assembled_usp_run_hands_the_process_groups_to_the_objective(self):
        """The handoff in ``build_bundle``, up to the objective model's construction."""
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from specforge.application.composition import resolve_run
        from specforge.config import load_config
        from specforge.training.assembly import build_model_bundle
        from specforge.training.model_loading import training_target_config
        from tests.test_runtime.test_canonical_end_to_end import drafts, write_run

        class Reached(Exception):
            pass

        received = {}

        def construction(model, **arguments):
            received.update(arguments)
            raise Reached

        groups = sequence_groups()
        draft = next(d for d in drafts() if d.architecture == "Eagle3DraftModel")
        with TemporaryDirectory() as temp:
            cfg = load_config(
                str(write_run(Path(temp), draft)),
                [
                    "training.attention_backend=usp",
                    "training.sp_ring_size=2",
                    "training.batch_size=1",
                ],
            )
            resolved = resolve_run(cfg, None, target_config=training_target_config(cfg))
            with (
                patch(
                    "specforge.training.sequence_parallel.sequence_parallel_groups",
                    return_value=groups,
                ),
                patch(
                    "specforge.algorithms.objectives.build_objective_model",
                    construction,
                ),
                self.assertRaises(Reached),
            ):
                build_model_bundle(resolved.config, algorithm=resolved.algorithm)
        self.assertEqual(received["attention_backend"], "usp")
        self.assertIs(received["sequence_parallel"], groups)

    def test_trimming_is_skipped_when_statistics_cross_ranks(self):
        # Trimming leaves each sequence-parallel rank a different number of
        # units; with a chunk size the ranks would then issue a different
        # number of collectives. The reducer here stands in for the group's.
        calls = []

        def reduce(accepted, count, denominator, *, group):
            calls.append(group)
            return accepted, count, denominator

        loss = {
            "terms": {"token": {"kind": "lk_alpha", "options": {"target": "teacher"}}},
            "execution": {"trim": True, "chunk_size": 2, "checkpoint": False},
        }
        results = {}
        for name, reducer in (("local", None), ("reduced", reduce)):
            config, runtime = objective_model("eager", loss=loss)
            with (
                warnings.catch_warnings(record=True) as caught,
                patch("specforge.training.sequence_parallel.reduce_statistics", reduce),
                # Only the reducer differs between the two runs: a model built
                # with the groups also runs the USP rollout over a shorter span.
                patch.object(
                    runtime, "sequence_parallel", reducer and sequence_groups()
                ),
            ):
                warnings.simplefilter("always")
                results[name] = forward_loss(config, runtime)
            trimming_disabled = [
                w for w in caught if "Loss trimming is disabled" in str(w.message)
            ]
            self.assertEqual(bool(trimming_disabled), reducer is not None)
        # one reduction per chunk of the untrimmed units, for each rollout depth
        length = training_batch(eagle_config())["input_ids"].shape[1]
        depths = resolve_objective("Eagle3DraftModel").objective.ttt_length
        self.assertEqual(calls, ["sequence"] * (depths * -(-length // 2)))
        torch.testing.assert_close(results["reduced"], results["local"])


if __name__ == "__main__":
    unittest.main()
