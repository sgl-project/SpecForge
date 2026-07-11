import json
import os
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import MagicMock, patch

import torch

from scripts.train_peagle import (
    _rank_training_state_path,
    _resolve_resume_training_state_path,
    build_draft_model,
)

TINY_CHECKPOINT_CONFIG = {
    "architectures": ["PEagleDraftModel"],
    "model_type": "llama",
    "hidden_act": "silu",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "max_position_embeddings": 64,
    "vocab_size": 32,
    "draft_vocab_size": 32,
    "pad_token_id": 0,
    "rms_norm_eps": 1e-5,
    "tie_word_embeddings": False,
}

TINY_PEAGLE_CONFIG = {
    "num_depths": 5,
    "down_sample_ratio": 0.7,
    "down_sample_ratio_min": 0.3,
    "mask_token_id": 31,
    "num_draft_layers": 2,
    "norm_before_residual": True,
}


class TestTrainPEagleResume(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.workdir.cleanup)
        self.checkpoint_dir = os.path.join(
            self.workdir.name,
            "epoch_0_step_1",
        )
        os.makedirs(self.checkpoint_dir)
        with open(
            os.path.join(self.checkpoint_dir, "config.json"),
            "w",
            encoding="utf-8",
        ) as output:
            json.dump(TINY_CHECKPOINT_CONFIG, output)
        with open(
            os.path.join(self.checkpoint_dir, "peagle_config.json"),
            "w",
            encoding="utf-8",
        ) as output:
            json.dump(TINY_PEAGLE_CONFIG, output)

    def _args(self, **overrides):
        values = {
            "resume": True,
            "ckpt_dir": None,
            "output_dir": self.workdir.name,
            "num_draft_layers": None,
            "norm_before_residual": None,
            "num_depths": None,
            "down_sample_ratio": None,
            "down_sample_ratio_min": None,
            "mask_token_id": None,
            "draft_model_config": None,
            "target_model_path": "unused-target",
            "model_download_dir": None,
            "embedding_key": "model.embed_tokens.weight",
        }
        values.update(overrides)
        return Namespace(**values)

    @patch("scripts.train_peagle.print_on_rank0")
    @patch("safetensors.torch.load_file")
    @patch("scripts.train_peagle.torch.load")
    @patch("scripts.train_peagle.PEagleDraftModel")
    def test_resume_restores_model_and_cod_semantics(
        self,
        model_cls,
        torch_load,
        load_file,
        _print_on_rank0,
    ):
        for filename in ("model.safetensors", "training_state.pt"):
            open(os.path.join(self.checkpoint_dir, filename), "wb").close()
        load_file.return_value = {"embed_tokens.weight": torch.zeros(1)}
        torch_load.return_value = {"epoch": 0, "global_step": 1}
        model = MagicMock()
        model.to.return_value = model
        model.norm_before_residual = True
        model_cls.return_value = model
        args = self._args()

        config, returned_model, ckpt_info, resume_state = build_draft_model(args)

        self.assertIs(returned_model, model)
        self.assertEqual(ckpt_info, (0, 1))
        self.assertEqual(resume_state["global_step"], 1)
        self.assertEqual(config.num_hidden_layers, 2)
        self.assertTrue(config.norm_before_residual)
        self.assertEqual(args.num_draft_layers, 2)
        self.assertTrue(args.norm_before_residual)
        self.assertEqual(args.num_depths, 5)
        self.assertEqual(args.down_sample_ratio, 0.7)
        self.assertEqual(args.down_sample_ratio_min, 0.3)
        self.assertEqual(args.mask_token_id, 31)
        model_cls.assert_called_once_with(
            config=config,
            norm_before_residual=None,
        )

    def test_resume_rejects_layer_count_change(self):
        args = self._args(num_draft_layers=4)

        with self.assertRaisesRegex(ValueError, "layer count"):
            build_draft_model(args)

    @patch("scripts.train_peagle.dist.get_world_size", return_value=2)
    @patch("scripts.train_peagle.dist.get_rank", return_value=1)
    @patch("scripts.train_peagle.dist.is_initialized", return_value=True)
    def test_distributed_resume_uses_rank_local_optimizer_state(
        self,
        _is_initialized,
        _get_rank,
        _get_world_size,
    ):
        rank_path = _rank_training_state_path(self.checkpoint_dir, 1)
        open(rank_path, "wb").close()
        open(os.path.join(self.checkpoint_dir, "training_state.pt"), "wb").close()

        self.assertEqual(
            _resolve_resume_training_state_path(self.checkpoint_dir),
            rank_path,
        )

    @patch("scripts.train_peagle.dist.get_world_size", return_value=2)
    @patch("scripts.train_peagle.dist.get_rank", return_value=1)
    @patch("scripts.train_peagle.dist.is_initialized", return_value=True)
    def test_distributed_resume_rejects_legacy_rank_zero_state(
        self,
        _is_initialized,
        _get_rank,
        _get_world_size,
    ):
        open(os.path.join(self.checkpoint_dir, "training_state.pt"), "wb").close()

        with self.assertRaisesRegex(ValueError, "legacy rank-0"):
            _resolve_resume_training_state_path(self.checkpoint_dir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
