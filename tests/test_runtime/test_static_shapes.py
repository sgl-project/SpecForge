"""``training.static_shapes``: one batch length and one anchor count per run, for compiled blocks."""

import types
import unittest

import torch

from specforge.algorithms.builtin import builtin_algorithm_registry

ALGORITHM = builtin_algorithm_registry().resolve("dflash")


def _dflash_sample(length, width=4):
    return {
        "input_ids": torch.arange(length).unsqueeze(0),
        "loss_mask": torch.ones(1, length, dtype=torch.long),
        "hidden_states": torch.randn(1, length, width),
    }


class TestPadToCollation(unittest.TestCase):
    def test_pad_and_concatenate_pads_to_the_static_length(self):
        from specforge.algorithms.common.collation import pad_and_concatenate_features

        batch = pad_and_concatenate_features(
            [_dflash_sample(5), _dflash_sample(3)],
            sequence_axes={"input_ids": 1, "loss_mask": 1, "hidden_states": 1},
            required_keys=("input_ids", "loss_mask", "hidden_states"),
            pad_to=8,
        )
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 8))
        self.assertEqual(tuple(batch["hidden_states"].shape), (2, 8, 4))
        self.assertEqual(int(batch["loss_mask"][1, 3:].sum()), 0)

    def test_pad_to_rejects_longer_samples(self):
        from specforge.algorithms.common.collation import pad_and_concatenate_features

        with self.assertRaisesRegex(ValueError, "static batch length"):
            pad_and_concatenate_features(
                [_dflash_sample(5)],
                sequence_axes={"input_ids": 1, "loss_mask": 1, "hidden_states": 1},
                required_keys=("input_ids", "loss_mask", "hidden_states"),
                pad_to=4,
            )

    def test_dflash_family_collators_accept_pad_to(self):
        from specforge.algorithms.common.hidden_states_data import (
            build_collator,
            build_dspark_collator,
        )

        batch = build_collator(pad_to=8)([_dflash_sample(5), _dflash_sample(3)])
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 8))
        dspark = [
            dict(_dflash_sample(5), target_last_hidden_states=torch.randn(1, 5, 4)),
            dict(_dflash_sample(2), target_last_hidden_states=torch.randn(1, 2, 4)),
        ]
        batch = build_dspark_collator(pad_to=8)(dspark)
        self.assertEqual(tuple(batch["target_last_hidden_states"].shape), (2, 8, 4))
        # The default stays pad-to-longest.
        batch = build_collator()([_dflash_sample(5), _dflash_sample(3)])
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 5))

    def test_eagle3_offline_collator_accepts_pad_to(self):
        from specforge.data.utils import DataCollatorWithPadding

        def item(n):
            ones = torch.ones(1, n, dtype=torch.long)
            return {"input_ids": ones, "attention_mask": ones.clone(), "loss_mask": ones.clone()}

        batch = DataCollatorWithPadding(pad_to=8)([item(5), item(3)])
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 8))
        self.assertEqual(tuple(batch["loss_mask"].shape), (2, 8))
        with self.assertRaisesRegex(ValueError, "static batch length"):
            DataCollatorWithPadding(pad_to=4)([item(5)])


class TestStaticAnchorCount(unittest.TestCase):
    """``num_anchors`` slots are always sampled; slots past a row's valid anchors are masked."""

    @staticmethod
    def _mask():
        # Row 0 supervises positions 0-2 (anchors 0, 1 valid); row 1 positions 0-3 (anchors 0-2).
        mask = torch.zeros(2, 6)
        mask[0, :3] = 1
        mask[1, :4] = 1
        return mask

    def _sample(self, static, seed=0):
        from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel

        model = types.SimpleNamespace(num_anchors=4, static_anchor_count=static)
        torch.manual_seed(seed)
        return OnlineDFlashModel._sample_anchor_positions(
            model, 6, self._mask(), torch.device("cpu"), max_valid_anchors=3
        )

    def test_static_width_is_num_anchors_and_extra_slots_are_masked(self):
        anchors, keep = self._sample(static=True)
        self.assertEqual(tuple(anchors.shape), (2, 4))
        self.assertEqual(keep.sum(dim=1).tolist(), [2, 3])
        dyn_anchors, dyn_keep = self._sample(static=False)
        self.assertEqual(tuple(dyn_anchors.shape), (2, 3))
        for row in range(2):
            self.assertEqual(
                sorted(anchors[row][keep[row]].tolist()),
                sorted(dyn_anchors[row][dyn_keep[row]].tolist()),
            )

    def test_no_valid_anchor_still_raises(self):
        from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel

        model = types.SimpleNamespace(num_anchors=4, static_anchor_count=True)
        with self.assertRaisesRegex(ValueError, "consecutive supervised"):
            OnlineDFlashModel._sample_anchor_positions(
                model, 6, torch.zeros(1, 6), torch.device("cpu"), max_valid_anchors=0
            )


class TestLaunchStaticCollators(unittest.TestCase):
    def test_static_or_dynamic_collator(self):
        from specforge.launch import _static_or_dynamic_collator, _static_pad_length

        self.assertEqual(_static_or_dynamic_collator(lambda: "dynamic", None), "dynamic")
        self.assertEqual(_static_or_dynamic_collator(lambda pad_to=None: pad_to, 8), 8)
        with self.assertRaisesRegex(ValueError, "static_shapes"):
            _static_or_dynamic_collator(lambda: "no pad_to", 8)
        self.assertIsNone(_static_pad_length(False, None))
        self.assertEqual(_static_pad_length(True, 2048), 2048)
        with self.assertRaisesRegex(ValueError, "max_len"):
            _static_pad_length(True, None)

    def test_streaming_and_offline_collators_pad_to_max_len(self):
        from specforge.launch import _offline_io, _streaming_collate

        collate = _streaming_collate(ALGORITHM, "text", None, pad_to=8)
        batch = collate([_dflash_sample(5), _dflash_sample(3)])
        self.assertEqual(tuple(batch["hidden_states"].shape), (2, 8, 4))
        collate, _ = _offline_io(
            ALGORITHM, "text", 8, ttt_length=7, use_usp_preprocess=False, static_shapes=True
        )
        self.assertEqual(
            tuple(collate([_dflash_sample(5), _dflash_sample(3)])["input_ids"].shape), (2, 8)
        )
        collate, _ = _offline_io(ALGORITHM, "text", 8, ttt_length=7, use_usp_preprocess=False)
        self.assertEqual(
            tuple(collate([_dflash_sample(5), _dflash_sample(3)])["input_ids"].shape), (2, 5)
        )


class TestStaticShapesConfig(unittest.TestCase):
    def test_compile_blocks_requires_static_shapes(self):
        from specforge.config.schema import TrainingConfig

        self.assertFalse(TrainingConfig().static_shapes)
        with self.assertRaisesRegex(ValueError, "static_shapes"):
            TrainingConfig(backend="fsdp2", compile_blocks=True)
        cfg = TrainingConfig(backend="fsdp2", compile_blocks=True, static_shapes=True)
        self.assertTrue(cfg.static_shapes)
        # static_shapes alone is allowed on either backend.
        self.assertTrue(TrainingConfig(static_shapes=True).static_shapes)


if __name__ == "__main__":
    unittest.main()
