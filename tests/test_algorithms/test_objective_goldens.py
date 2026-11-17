"""Canonical losses and gradients stay at their recorded reference values.

``tests/fixtures/objective_goldens.json`` holds the loss and gradient norm of
each case below. They were recorded while the same constructions matched the
legacy training wrappers (``test_objective_legacy_parity.py``,
``test_acceptance_losses.py``), so they remain a reference for the formulas the
recipes were trained with. Regenerate after a deliberate numerical change with
``SPECFORGE_UPDATE_GOLDENS=1`` and say in the commit what moved and why.
"""

import copy
import json
import os
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from specforge.algorithms.objectives import build_objective_model
from specforge.algorithms.objectives.losses import PositionWeighting
from specforge.losses.position_weights import DPACE_LOSS_TYPES
from specforge.modeling import build_draft_model
from tests.test_algorithms.test_objectives import (
    dflash2_config,
    domino_config,
    draft_config,
    dspark_config,
    dspine_config,
    eagle_config,
    peagle_config,
    training_batch,
)
from tests.test_modeling.test_canonical_eagle3 import install_mapping

GOLDENS = Path(__file__).resolve().parents[1] / "fixtures/objective_goldens.json"


def assert_golden(case, model, loss):
    """Pin one case's canonical loss and gradients; ``loss`` is backpropagated."""
    if all(p.grad is None for p in model.draft_model.parameters()):
        loss.backward()
    gradients = [p.grad for p in model.draft_model.parameters() if p.grad is not None]
    actual = {
        "loss": loss.item(),
        "gradient_norm": torch.cat([g.double().flatten() for g in gradients])
        .norm()
        .item(),
        "gradient_tensors": len(gradients),
    }
    recorded = json.loads(GOLDENS.read_text()) if GOLDENS.exists() else {}
    if os.environ.get("SPECFORGE_UPDATE_GOLDENS"):
        recorded[case] = actual
        GOLDENS.write_text(json.dumps(recorded, indent=1, sort_keys=True) + "\n")
        return
    expected = recorded[case]
    assert actual["gradient_tensors"] == expected["gradient_tensors"], case
    # Loose enough for another BLAS; a changed formula moves these by percents.
    for name in ("loss", "gradient_norm"):
        torch.testing.assert_close(
            torch.tensor(actual[name]),
            torch.tensor(expected[name]),
            rtol=1e-4,
            atol=1e-6,
            msg=lambda message: f"{case} {name}: {message}",
        )


def block_runtime(kind, *, shifted=False, loss=None):
    """The canonical model and objective of one block-family reference case."""
    configs = {
        "dflash": lambda: draft_config(target_bindings=(), target_logit_transform=None),
        "dflash2": dflash2_config,
        "domino": lambda: domino_config(shifted=shifted),
        "dspark": dspark_config,
        "dspine": dspine_config,
    }
    config = configs[kind]()
    config = replace(
        config,
        spec_config=replace(
            config.spec_config, block=replace(config.spec_config.block, block_size=4)
        ),
    )
    model = build_draft_model(config)
    for name in config.target_bindings:
        model.get_submodule(name).load_target_weight(
            torch.randn(config.vocab_size, config.hidden_size)
        )
    if kind == "dspine":
        model.initialize_transfer_space()
    return build_objective_model(
        model,
        options={"num_anchors": 2},
        target_head=copy.deepcopy(model.lm_head),
        loss=loss,
    )


class ObjectiveGoldenTest(unittest.TestCase):
    def test_block_objectives(self):
        for kind, shifted in (
            ("dflash", False),
            ("dflash2", False),
            ("dspark", False),
            ("domino", False),
            ("domino", True),
            ("dspine", False),
        ):
            for step, evaluation in ((0, False), (5, False), (1, True)):
                case = (
                    f"{kind}{'-shifted' if shifted else ''}"
                    f"-step{step}-{'eval' if evaluation else 'train'}"
                )
                with self.subTest(case=case):
                    torch.manual_seed(7)
                    runtime = block_runtime(kind, shifted=shifted)
                    runtime.draft_model.train(not evaluation)
                    torch.manual_seed(13)
                    batch = training_batch(runtime.draft_model.config.draft)
                    batch["loss_mask"][0, 4:6] = 0
                    if kind == "dflash2":
                        runtime.objective.selector_warmup_steps = 1
                        runtime.objective.selector_ramp_steps = 5
                    torch.manual_seed(11)
                    result = runtime(batch, step=step, total_steps=11)
                    assert_golden(case, runtime, result.loss)

    def test_acceptance_losses_and_chain_weighting(self):
        for weighting in ("uniform", *sorted(DPACE_LOSS_TYPES)):
            for mode in ("alpha", "tv", "lambda"):
                case = f"dflash-lk_{mode}-{weighting}"
                with self.subTest(case=case):
                    options = {"kl_scale": 0.7, "kl_decay": 1.3}
                    token = {
                        "kind": f"lk_{mode}",
                        "options": options if mode == "lambda" else {},
                    }
                    torch.manual_seed(7)
                    runtime = block_runtime("dflash", loss={"terms": {"token": token}})
                    runtime.weighting = PositionWeighting(kind=weighting)
                    torch.manual_seed(13)
                    batch = {
                        k: v.repeat(2, *([1] * (v.ndim - 1)))
                        for k, v in training_batch(
                            runtime.draft_model.config.draft
                        ).items()
                    }
                    batch["loss_mask"][1, 3:] = 0  # unequal valid anchor counts
                    torch.manual_seed(11)
                    result = runtime(batch, step=1, total_steps=10)
                    assert_golden(case, runtime, result.loss)

    def test_rollout_objectives(self):
        for kind in ("eagle3", "peagle"):
            for length in (7, 9):
                case = f"{kind}-length{length}"
                with self.subTest(case=case):
                    torch.manual_seed(7)
                    config = eagle_config() if kind == "eagle3" else peagle_config()
                    model = build_draft_model(config)
                    install_mapping(model)
                    options = (
                        {"ttt_length": 3} if kind == "eagle3" else {"num_depths": 3}
                    )
                    runtime = build_objective_model(model, options=options)
                    torch.manual_seed(13)
                    batch = {
                        k: v[:, :length].detach()
                        for k, v in training_batch(config).items()
                    }
                    batch["target_logits"] = torch.randn(1, length, config.vocab_size)
                    batch["loss_mask"][:, -1] = 0
                    torch.manual_seed(11)
                    result = runtime(batch, step=0, total_steps=None)
                    assert_golden(case, runtime, result.loss)


if __name__ == "__main__":
    unittest.main()
