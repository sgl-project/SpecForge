# coding=utf-8
"""Gate: the typed config builds the unified ``TrainingRun`` lifecycle.

A tiny offline fixture world + a YAML config through package-level assembly
must produce the same ``TrainingRun`` the CLI executes, and it must train.

GPU-only. Run on the H200 box via rcli.
"""

import os
import tempfile
import unittest
from unittest import mock

import torch

CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, "cli config build requires CUDA")
class TestCliConfigBuild(unittest.TestCase):
    def test_config_build_matches_programmatic_and_trains(self):
        torch.manual_seed(0)
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29582")

        import yaml

        from specforge.config import load_config
        from specforge.training.assembly import build_training_run
        from specforge.training.controller import TrainerController

        workdir = tempfile.mkdtemp(prefix="cli_cfg_")
        cfg_path = fx.write_draft_config(os.path.join(workdir, "draft.json"))
        target_dir = fx.write_target_head_dir(os.path.join(workdir, "target"))
        vocab_path = fx.write_vocab_mapping(os.path.join(workdir, "vm.pt"))
        feat_dir = fx.write_offline_files(os.path.join(workdir, "features"), n=4)

        run_config = {
            "model": {
                "target_model_path": target_dir,
                "draft_model_config": cfg_path,
                "vocab_mapping_path": vocab_path,
                # fixture target dir holds only the lm_head, not an embedding
                "load_target_embedding": False,
            },
            "data": {"hidden_states_path": feat_dir, "max_length": 512},
            "training": {
                "batch_size": 2,
                "accumulation_steps": 1,
                "ttt_length": 3,
                "max_steps": 4,
                "max_checkpoints": 2,
                "log_interval": 1,
            },
            "run_id": "cli-gate",
            "output_dir": os.path.join(workdir, "out"),
        }
        yaml_path = os.path.join(workdir, "run.yaml")
        with open(yaml_path, "w") as f:
            yaml.safe_dump(run_config, f)

        cfg = load_config(yaml_path, ["training.max_steps=2"])  # override applies
        run = build_training_run(cfg)
        trainer, loader = run.trainer, run.loader

        # package-level assembly is the single wiring the CLI executes
        self.assertIsInstance(trainer, TrainerController)
        self.assertEqual(trainer.run_id, "cli-gate")
        self.assertEqual(trainer.max_steps, 2)
        self.assertEqual(trainer._checkpoint_manager().max_checkpoints, 2)
        self.assertEqual(trainer.output_dir, run_config["output_dir"])
        self.assertEqual(trainer.core.accumulation_steps, 1)
        self.assertEqual(trainer.core.strategy.name, "eagle3")
        self.assertEqual(loader.batch_size, 2)

        # and it actually trains to the configured step cap
        step = run.run()
        self.assertEqual(step, 2)


class TestCliDispatch(unittest.TestCase):
    def test_train_command_dispatches_one_validated_config(self):
        from specforge.cli import main
        from specforge.config import Config

        cfg = Config.model_validate(
            {
                "model": {"target_model_path": "t", "draft_model_config": "d"},
                "data": {"prompts_path": "/prompts.jsonl"},
                "training": {"strategy": "dflash"},
            }
        )
        with (
            mock.patch("specforge.cli.load_config", return_value=cfg) as load,
            mock.patch("specforge.cli._train", return_value=3) as train,
        ):
            self.assertEqual(main(["train", "--config", "run.yaml"]), 0)
        load.assert_called_once_with("run.yaml", [])
        train.assert_called_once_with(cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
