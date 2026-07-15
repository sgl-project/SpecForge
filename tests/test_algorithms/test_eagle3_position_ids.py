"""Generic position-ID contracts for the EAGLE3 training model."""

import inspect
import unittest

import torch

from specforge.algorithms.eagle3.model import OnlineEagle3Model


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
