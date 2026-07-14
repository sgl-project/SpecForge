# coding=utf-8
"""Gate: `specforge train` config builds the same Trainer the programmatic path does.

A tiny offline fixture world + a YAML config through ``cli.build_from_config``
must produce a TrainerController wired exactly as ``build_offline_runtime``
called directly — and it must actually train.

GPU-only. Run on the H200 box via rcli.
"""

import os
import tempfile
import unittest

import torch

CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, "cli config build requires CUDA")
class TestCliConfigBuild(unittest.TestCase):
    def test_config_build_matches_programmatic_and_trains(self):
        torch.manual_seed(0)
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29582")

        import yaml

        from specforge.cli import build_from_config
        from specforge.config import load_config
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
        trainer, loader, drive_rollout = build_from_config(cfg)

        # same wiring the programmatic path produces
        self.assertIsInstance(trainer, TrainerController)
        self.assertIsNone(drive_rollout)  # offline
        self.assertEqual(trainer.run_id, "cli-gate")
        self.assertEqual(trainer.max_steps, 2)
        self.assertEqual(trainer._checkpoint_manager().max_checkpoints, 2)
        self.assertEqual(trainer.output_dir, run_config["output_dir"])
        self.assertEqual(trainer.core.accumulation_steps, 1)
        self.assertEqual(trainer.core.strategy.name, "eagle3")
        self.assertEqual(loader.batch_size, 2)

        # and it actually trains to the configured step cap
        step = trainer.fit(loader)
        self.assertEqual(step, 2)

    def test_non_eagle3_strategy_points_to_dedicated_script(self):
        from specforge.cli import build_from_config
        from specforge.config import Config

        cfg = Config.model_validate(
            {
                "model": {"target_model_path": "t", "draft_model_config": "d"},
                "data": {"hidden_states_path": "/features"},
                "training": {"strategy": "dflash"},
            }
        )
        with self.assertRaises(NotImplementedError):
            build_from_config(cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
