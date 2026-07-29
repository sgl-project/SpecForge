"""Generic position-ID contracts for the EAGLE3 training model."""

import importlib.util
import inspect
import sys
import types
import unittest
from pathlib import Path

import torch

_loss_module = types.ModuleType("specforge.core.loss")
_loss_module.LogSoftmaxLoss = object
_model_spec = importlib.util.spec_from_file_location(
    "test_eagle3_position_model",
    Path(__file__).resolve().parents[2]
    / "specforge"
    / "algorithms"
    / "eagle3"
    / "model.py",
)
assert _model_spec is not None and _model_spec.loader is not None
_model_module = importlib.util.module_from_spec(_model_spec)
_missing = object()
_previous_loss_module = sys.modules.get("specforge.core.loss", _missing)
sys.modules["specforge.core.loss"] = _loss_module
try:
    _model_spec.loader.exec_module(_model_module)
finally:
    if _previous_loss_module is _missing:
        sys.modules.pop("specforge.core.loss", None)
    else:
        sys.modules["specforge.core.loss"] = _previous_loss_module
OnlineEagle3Model = _model_module.OnlineEagle3Model


class Eagle3PositionIdsTest(unittest.TestCase):
    def setUp(self):
        self.model = OnlineEagle3Model(draft_model=object(), attention_backend="sdpa")

    def _prepare(self, position_ids, *, seq_length=4):
        return self.model._prepare_position_ids(
            position_ids,
            seq_length=seq_length,
            past_key_values_length=0,
            device=torch.device("cpu"),
        )

    def test_accepts_supplied_2d_position_ids(self):
        position_ids = torch.arange(8, dtype=torch.int32).reshape(2, 4)

        prepared = self._prepare(position_ids)

        self.assertEqual(prepared.shape, (2, 4))
        self.assertEqual(prepared.dtype, torch.long)
        self.assertTrue(torch.equal(prepared, position_ids.long()))

    def test_accepts_generic_3d_position_ids(self):
        position_ids = torch.arange(16).reshape(2, 2, 4)

        prepared = self._prepare(position_ids)

        self.assertEqual(prepared.shape, (2, 2, 4))
        self.assertTrue(torch.equal(prepared, position_ids))

    def test_usp_preserves_ulysses_expanded_position_ids(self):
        self.model.attention_backend = "usp"
        position_ids = torch.arange(8, dtype=torch.int32).reshape(1, 8)

        prepared = self._prepare(position_ids, seq_length=4)

        self.assertIs(prepared, position_ids)
        self.assertEqual(prepared.shape, (1, 8))
        self.assertEqual(prepared.dtype, torch.int32)

    def test_rejects_position_ids_with_an_invalid_shape(self):
        for position_ids in (torch.arange(4), torch.zeros(2, 3, dtype=torch.long)):
            with self.subTest(shape=tuple(position_ids.shape)):
                with self.assertRaisesRegex(ValueError, "position_ids"):
                    self._prepare(position_ids)

    def test_forward_surface_has_no_builtin_vlm_arguments(self):
        parameters = inspect.signature(OnlineEagle3Model.forward).parameters

        self.assertNotIn("is_vlm", parameters)
        self.assertNotIn("image_grid_thw", parameters)
        self.assertFalse(
            any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
