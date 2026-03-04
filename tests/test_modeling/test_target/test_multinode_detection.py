"""Tests for multi-node topology and IPC optimization detection.

The should_disable_ipc_optimizations gate controls whether CUDA IPC-based
optimizations (custom_all_reduce, FlashInfer allreduce fusion) are disabled.
These optimizations use IPC handles that cannot cross node boundaries
unless MNNVL fabric memory (via IMEX channels) is available.
"""

import os
import unittest
from unittest.mock import patch

from specforge.modeling.target.eagle3_target_model import (
    _has_mnnvl_ipc,
    _is_multi_node,
    should_disable_ipc_optimizations,
)


class TestIsMultiNode(unittest.TestCase):
    """Test the _is_multi_node topology detection."""

    # --- Single-node topologies ---

    @patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "4"})
    def test_single_node_tp4(self):
        """1 node, 4 GPUs, TP=4."""
        self.assertFalse(_is_multi_node(world_size=4, tp_size=4))

    @patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "8"})
    def test_single_node_tp4_dp2(self):
        """1 node, 8 GPUs, TP=4, DP=2: single-node DP."""
        self.assertFalse(_is_multi_node(world_size=8, tp_size=4))

    @patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "1"})
    def test_single_node_tp1(self):
        """1 node, 1 GPU."""
        self.assertFalse(_is_multi_node(world_size=1, tp_size=1))

    # --- Multi-node topologies ---

    @patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "4"})
    def test_multi_node_tp4_dp2(self):
        """2 nodes, 4 GPUs each, TP=4, DP=2."""
        self.assertTrue(_is_multi_node(world_size=8, tp_size=4))

    @patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "4"})
    def test_multi_node_tp4_dp4(self):
        """4 nodes, 4 GPUs each, TP=4, DP=4."""
        self.assertTrue(_is_multi_node(world_size=16, tp_size=4))

    @patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "4"})
    def test_multi_node_tp8_across_nodes(self):
        """2 nodes, 4 GPUs each, TP=8 spanning nodes."""
        self.assertTrue(_is_multi_node(world_size=8, tp_size=8))

    # --- Fallback (no LOCAL_WORLD_SIZE) ---

    @patch.dict(os.environ, {}, clear=False)
    def test_fallback_single_node(self):
        os.environ.pop("LOCAL_WORLD_SIZE", None)
        self.assertFalse(_is_multi_node(world_size=4, tp_size=4))

    @patch.dict(os.environ, {}, clear=False)
    def test_fallback_multi_node(self):
        os.environ.pop("LOCAL_WORLD_SIZE", None)
        self.assertTrue(_is_multi_node(world_size=8, tp_size=4))

    @patch.dict(os.environ, {}, clear=False)
    def test_fallback_cross_node_tp_false_negative(self):
        """Fallback cannot detect cross-node TP where world_size == tp_size."""
        os.environ.pop("LOCAL_WORLD_SIZE", None)
        self.assertFalse(_is_multi_node(world_size=8, tp_size=8))


class TestShouldDisableIpcOptimizations(unittest.TestCase):
    """Test the combined multi-node + MNNVL gate."""

    # --- Single-node: never disable ---

    @patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "4"})
    @patch("specforge.modeling.target.eagle3_target_model._has_mnnvl_ipc", return_value=False)
    def test_single_node_no_mnnvl(self, _):
        """Single-node without MNNVL: keep optimizations."""
        self.assertFalse(should_disable_ipc_optimizations(world_size=4, tp_size=4))

    @patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "8"})
    @patch("specforge.modeling.target.eagle3_target_model._has_mnnvl_ipc", return_value=False)
    def test_single_node_dp_no_mnnvl(self, _):
        """Single-node DP without MNNVL: keep optimizations."""
        self.assertFalse(should_disable_ipc_optimizations(world_size=8, tp_size=4))

    # --- Multi-node without MNNVL: disable ---

    @patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "4"})
    @patch("specforge.modeling.target.eagle3_target_model._has_mnnvl_ipc", return_value=False)
    def test_multi_node_no_mnnvl(self, _):
        """Multi-node without MNNVL: disable IPC optimizations."""
        self.assertTrue(should_disable_ipc_optimizations(world_size=8, tp_size=4))

    # --- Multi-node with MNNVL (GB200 NVL72): keep ---

    @patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "4"})
    @patch("specforge.modeling.target.eagle3_target_model._has_mnnvl_ipc", return_value=True)
    def test_multi_node_with_mnnvl(self, _):
        """Multi-node with MNNVL (e.g. GB200 NVL72): keep IPC optimizations."""
        self.assertFalse(should_disable_ipc_optimizations(world_size=8, tp_size=4))

    @patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "4"})
    @patch("specforge.modeling.target.eagle3_target_model._has_mnnvl_ipc", return_value=True)
    def test_multi_node_cross_node_tp_with_mnnvl(self, _):
        """Cross-node TP with MNNVL: keep IPC optimizations."""
        self.assertFalse(should_disable_ipc_optimizations(world_size=8, tp_size=8))


class TestHasMnnvlIpc(unittest.TestCase):
    """Test IMEX channel detection."""

    @patch("os.path.isdir", return_value=True)
    def test_imex_present(self, _):
        self.assertTrue(_has_mnnvl_ipc())

    @patch("os.path.isdir", return_value=False)
    def test_imex_absent(self, _):
        self.assertFalse(_has_mnnvl_ipc())


if __name__ == "__main__":
    unittest.main()
