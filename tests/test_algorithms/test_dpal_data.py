"""D-PAL teacher states survive capture, batching, quiet steps and resume."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from pydantic import ValidationError
from torch import nn

from specforge.algorithms.common.providers import resolve_server_capture_layout
from specforge.application import bind_run, resolve_run
from specforge.config import Config
from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import DFlashTrainStrategy, StepContext

LOSSES = ("dpal-renyi", "dpal-kl", "dpal-la", "dpal-ce-tv")


def config(kind):
    return Config(
        model={"target_model_path": "target"},
        data={"hidden_states_path": "features"},
        training={
            "strategy": "dflash",
            "loss_type": kind,
            "dflash_teacher_metrics": False,
        },
    )


class DPALDataTest(unittest.TestCase):
    def test_config_and_feature_contract_follow_the_objective(self):
        raw = {
            "input_ids": torch.tensor([1, 2, 3]),
            "loss_mask": torch.ones(3),
            "aux_hidden_states": torch.ones(1, 3, 12),
            "last_hidden_states": torch.arange(12).reshape(1, 3, 4),
        }
        for kind in ("dflash", "dpace", *LOSSES):
            with self.subTest(kind=kind):
                cfg = config(kind)
                self.assertEqual(cfg.training.dpal_rho, 0.3)
                run = resolve_run(cfg)
                run = bind_run(cfg, run.algorithm)
                provider = run.algorithm.providers.offline_for("text")
                record = provider.capture_layout.materialize(raw)
                expected = {"input_ids", "loss_mask", "hidden_states"}
                if kind in LOSSES:
                    expected.add("target_last_hidden_states")
                self.assertEqual(set(record), expected)
                self.assertEqual(
                    run.algorithm.spec.feature_contract(
                        "offline", "text"
                    ).required_tensors,
                    expected,
                )
                normalized = provider.build_normalizer(3)(record)
                batch = provider.build_collator()(
                    [normalized, provider.build_normalizer(2)(record)]
                )
                self.assertEqual(set(batch), expected)
                layout = resolve_server_capture_layout(
                    run.algorithm, cfg, modality="text"
                )
                self.assertEqual(layout.last_hidden_feature is not None, kind in LOSSES)
                reader = provider.build_reader(
                    "features", run_id="test", ttt_length=1, max_len=3
                )
                self.assertEqual(set(reader.feature_keys), expected)
                if kind in LOSSES:
                    self.assertEqual(
                        batch["target_last_hidden_states"].shape, (2, 3, 4)
                    )
                    self.assertEqual(batch["target_last_hidden_states"][1, 2].sum(), 0)
                    del record["target_last_hidden_states"]
                    with self.assertRaisesRegex(KeyError, "target_last_hidden_states"):
                        provider.build_normalizer(3)(record)

    def test_rejects_legacy_names_and_incompatible_objectives(self):
        for kind in (
            "dpard",
            "dpala",
            "dpakl",
            "dynamic-nla",
            "dynamic-kl",
            "dynamic-ce-tv",
        ):
            with self.subTest(kind=kind), self.assertRaises(ValidationError):
                config(kind)
        for kind in LOSSES:
            payload = config(kind).model_dump()
            for rho in (-0.1, 1.1, float("nan")):
                payload["training"]["dpal_rho"] = rho
                with (
                    self.subTest(kind=kind, rho=rho),
                    self.assertRaises(ValidationError),
                ):
                    Config.model_validate(payload)
            payload["training"]["dpal_rho"] = 0.3
            for strategy in ("eagle3", "domino", "dspark", "mtp"):
                payload["training"]["strategy"] = strategy
                with (
                    self.subTest(kind=kind, strategy=strategy),
                    self.assertRaises(ValidationError),
                ):
                    Config.model_validate(payload)
            payload["training"]["strategy"] = "dflash"
            payload["training"]["lk_loss_type"] = "tv"
            with self.assertRaises(ValidationError):
                Config.model_validate(payload)

    def test_quiet_step_keeps_teacher_through_a_wrapper(self):
        class Subject(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()))
                self.teacher_metrics = False
                self.requires_target_hidden_states = True
                self.received_teacher = None

            def forward(self, **inputs):
                self.received_teacher = inputs.get("target_last_hidden_states")
                return self.weight, self.weight.detach(), {}

        class Wrapper(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, **kwargs):
                return self.module(**kwargs)

        subject = Subject()
        tensors = {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "loss_mask": torch.ones(1, 4),
            "hidden_states": torch.ones(1, 4, 12),
            "target_last_hidden_states": torch.ones(1, 4, 4),
        }
        strategy = DFlashTrainStrategy(Wrapper(subject))
        strategy.forward_loss(
            TrainBatch(sample_ids=["test"], strategy="dflash", tensors=tensors),
            StepContext(collect_detailed_metrics=False),
        )
        torch.testing.assert_close(
            subject.received_teacher, tensors["target_last_hidden_states"]
        )

    def test_capture_cli_selects_dpal_teacher_states(self):
        from scripts.prepare_hidden_states import (
            parse_args,
            resolve_offline_capture_plan,
        )

        for kind in LOSSES:
            argv = [
                "prepare_hidden_states.py",
                "--target-model-path",
                "target",
                "--data-path",
                "data.jsonl",
                "--strategy",
                "dflash",
                "--loss-type",
                kind,
                "--draft-model-config",
                str(
                    Path(__file__).resolve().parents[2] / "configs/qwen3-4b-dflash.json"
                ),
            ]
            with patch("sys.argv", argv):
                args = parse_args()
            plan = resolve_offline_capture_plan(
                args, SimpleNamespace(num_hidden_layers=36, hidden_size=2560)
            )
            self.assertEqual(
                plan.layout.last_hidden_feature, "target_last_hidden_states"
            )


if __name__ == "__main__":
    unittest.main()
