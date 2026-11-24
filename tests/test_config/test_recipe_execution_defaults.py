"""Bundled recipes keep the execution defaults their legacy training route resolved.

``tests/fixtures/recipe_execution_defaults.json`` was recorded at dddef4b3, the
last commit whose recipes ran the legacy training wrappers. For every recipe it
holds what a CUDA trainer resolved there: whether the loss used the fused
kernels, the loss chunk (in flattened row x block units), chunk checkpointing,
Liger, the Triton grouped convolution and the feature-loader workers. The
canonical route has to resolve the same values. A deliberate change updates the
fixture in the same commit and says why.
"""

import json
import unittest
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from specforge.algorithms.objectives import resolve_objective
from specforge.config import Config
from specforge.config.draft.codec import parse_draft
from specforge.config.draft.conv import GroupedCausalConvSpec
from specforge.config.draft.mlp.dense import DenseFFNSpec
from specforge.training.assembly import _dataloader_num_workers
from specforge.training.canonical import liger_choice

ROOT = Path(__file__).resolve().parents[2]
RECIPES = ROOT / "examples/configs"
EXPECTED = json.loads(
    (ROOT / "tests/fixtures/recipe_execution_defaults.json").read_text()
)["recipes"]


@cache
def protocol_chunk(architecture):
    """(chunk size for one batch row, whether it scales with the rows)."""
    from tests.test_algorithms.test_objective_contracts import build_runtime
    from tests.test_algorithms.test_objectives import training_batch

    config, runtime = build_runtime(architecture)
    batch = training_batch(config)
    one = runtime.objective(runtime, batch, step=0, total_steps=1).chunk_size
    if not one:
        return 0, False
    doubled = {name: torch.cat((value, value)) for name, value in batch.items()}
    two = runtime.objective(runtime, doubled, step=0, total_steps=1).chunk_size
    return one, two == 2 * one


def resolved_defaults(cfg, draft):
    resolved = resolve_objective(
        draft.architecture,
        name=cfg.training.objective,
        options=cfg.training.objective_options,
        loss=cfg.training.loss,
        weighting=cfg.training.weighting,
    )
    requirements = resolved.registration.training
    chunk = resolved.execution.chunk_size
    if chunk is None:
        chunk, per_row = protocol_chunk(draft.architecture)
        chunk *= cfg.training.batch_size if per_row else 1
    activations = {
        node.hidden_act
        for node in draft.mlp_configs.values()
        if isinstance(node, DenseFFNSpec)
    }
    groups = [
        node.group_size
        for node in draft.conv_configs.values()
        if isinstance(node, GroupedCausalConvSpec)
    ]
    provider = SimpleNamespace(
        default_dataloader_num_workers=requirements.dataloader_workers
    )
    return {
        "fused_loss": resolved.execution.fused is not False,
        "loss_chunk": chunk,
        "loss_checkpoint": resolved.execution.checkpoint,
        "liger": liger_choice(
            cfg.model.use_liger_kernel,
            by_default=requirements.liger_by_default,
            hidden_act=(
                None
                if not activations
                else "silu" if activations <= {"silu", "swish"} else "other"
            ),
            platform="cuda",
        ).enabled,
        # the Triton kernel's rule: a power-of-two group size up to 1024
        "fused_conv": cfg.model.use_fused_conv is not False
        and bool(groups)
        and all(size & (size - 1) == 0 and size <= 1024 for size in groups),
        "dataloader_workers": _dataloader_num_workers(
            cfg, SimpleNamespace(providers=SimpleNamespace(model=provider))
        ),
        "attention_backend": cfg.training.attention_backend,
    }


class RecipeExecutionDefaultsTest(unittest.TestCase):
    def test_every_recipe_is_recorded(self):
        recipes = sorted(
            str(path.relative_to(RECIPES)) for path in RECIPES.rglob("*.yaml")
        )
        self.assertEqual(recipes, sorted(EXPECTED))

    def test_canonical_route_resolves_the_recorded_defaults(self):
        for name, expected in EXPECTED.items():
            with self.subTest(recipe=name):
                cfg = Config.model_validate(
                    yaml.safe_load((RECIPES / name).read_text())
                )
                payload = (
                    yaml.safe_load((ROOT / cfg.draft).read_text())
                    if isinstance(cfg.draft, str)
                    else cfg.draft
                )
                actual = resolved_defaults(
                    cfg, parse_draft(payload, target_config=None)
                )
                if expected["loss_checkpoint"] is None:  # never split
                    self.assertEqual(expected["loss_chunk"], 0)
                    actual["loss_checkpoint"] = None
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
