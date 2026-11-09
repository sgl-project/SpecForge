"""Historical persisted layouts, recorded before retiring the old constructors.

Shapes and dtypes are fixtures; values are synthetic. This tests import mappings
without rebuilding a legacy training/model API inside the test suite.
"""

import json
from pathlib import Path

import torch


def legacy_state(architecture):
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1] / "fixtures/legacy_weight_shapes.json"
        ).read_text()
    )
    generator = torch.Generator().manual_seed(19)
    state = {}
    for key, info in manifest[architecture].items():
        dtype = getattr(torch, info["dtype"])
        shape = info["shape"]
        if dtype.is_floating_point:
            state[key] = torch.randn(shape, generator=generator, dtype=dtype)
        else:
            state[key] = torch.zeros(shape, dtype=dtype)
    if "t2d" in state:
        state["t2d"][: state["d2t"].numel()] = True
    return state
