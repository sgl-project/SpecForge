"""Every objective that declares flex_attention hands Flex the active queries it requires."""

import unittest

import torch

from specforge.algorithms.objectives import resolve_objective
from specforge.algorithms.objectives.inputs import prepare_blocks, sequence_inputs
from tests.test_algorithms.test_objectives import (
    dflash2_config,
    domino_config,
    draft_config,
    dspark_config,
    training_batch,
)


class FlexLayoutTest(unittest.TestCase):
    def test_block_layouts_carry_active_queries(self):
        for make in (
            lambda: draft_config(target_bindings=(), target_logit_transform=None),
            dflash2_config,
            domino_config,
            dspark_config,
        ):
            config = make()
            with self.subTest(architecture=config.architecture):
                declared = resolve_objective(
                    config.architecture
                ).registration.training.attention_backends
                self.assertIn("flex_attention", declared)
                blocks = prepare_blocks(config, training_batch(config), num_anchors=4)
                layout = blocks.layer_inputs[0].attention_layout
                self.assertIsNotNone(layout.active_queries)
                torch.testing.assert_close(
                    layout.active_queries, layout.visibility[:, 0].any(-1)
                )

    def test_sequence_layouts_carry_active_queries(self):
        config = draft_config(target_bindings=(), target_logit_transform=None)
        positions = torch.arange(5)[None]
        visibility = torch.ones(1, 5, 5, dtype=torch.bool).tril()
        layout = sequence_inputs(config, positions, visibility)[0].attention_layout
        torch.testing.assert_close(layout.active_queries, visibility.any(-1))


if __name__ == "__main__":
    unittest.main()
