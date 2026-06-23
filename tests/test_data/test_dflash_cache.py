import unittest

import torch

from specforge.data.dflash_cache import (
    DEFAULT_DFLASH_CAPTURE_LAYERS,
    cache_file_path,
    layer_stack_from_concatenated,
    parse_layer_ids,
)


class TestDFlashCache(unittest.TestCase):
    def test_default_capture_layers_has_twelve_unique_layers(self):
        self.assertEqual(len(DEFAULT_DFLASH_CAPTURE_LAYERS), 12)
        self.assertEqual(len(set(DEFAULT_DFLASH_CAPTURE_LAYERS)), 12)

    def test_parse_layer_ids(self):
        self.assertEqual(parse_layer_ids("1, 7,12"), [1, 7, 12])
        with self.assertRaises(ValueError):
            parse_layer_ids("1,7,7")

    def test_layer_stack_from_concatenated(self):
        hidden = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 20)
        stacked = layer_stack_from_concatenated(hidden, num_layers=4, hidden_size=5)
        self.assertEqual(stacked.shape, (2, 3, 4, 5))
        self.assertTrue(torch.equal(stacked.reshape(2, 3, 20), hidden))

    def test_cache_file_path_groups_samples(self):
        path = cache_file_path("/cache", sample_index=2001, group_size=2000)
        self.assertEqual(str(path), "/cache/group_000001/sample_000000002001.ckpt")


if __name__ == "__main__":
    unittest.main()
