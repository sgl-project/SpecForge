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
        self._check_backend("fsdp")

    def test_config_selects_fsdp2_and_trains(self):
        self._check_backend("fsdp2")

    def _check_backend(self, training_backend):
        torch.manual_seed(0)
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29582")

        import yaml

        from specforge.application import build_application_run, resolve_run
        from specforge.config import load_config
        from specforge.training import Trainer

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
                "save_interval": 1,
                "log_interval": 1,
            },
            "run_id": "cli-gate",
            "output_dir": os.path.join(workdir, "out"),
        }
        yaml_path = os.path.join(workdir, "run.yaml")
        with open(yaml_path, "w") as f:
            yaml.safe_dump(run_config, f)

        cfg = load_config(
            yaml_path,
            [
                "training.max_steps=2",
                f"training.backend={training_backend}",
                "training.fsdp_sharding=FULL_SHARD",
            ],
        )
        construction_rng = torch.get_rng_state()
        run = build_application_run(resolve_run(cfg))
        trainer = run.trainer

        # package-level assembly is the single wiring the CLI executes
        self.assertIsInstance(trainer, Trainer)
        self.assertEqual(trainer.backend.name, training_backend)
        self.assertEqual(
            trainer.backend.parallel_config.sharding_strategy, "FULL_SHARD"
        )
        self.assertEqual(trainer.run_id, "cli-gate")
        self.assertEqual(trainer.max_steps, 2)
        self.assertEqual(trainer.max_checkpoints, 2)
        self.assertEqual(trainer.output_dir, run_config["output_dir"])
        self.assertEqual(trainer.batch_size, 2)
        self.assertEqual(trainer.core.accumulation_steps, 1)
        self.assertEqual(trainer.core.strategy.name, "eagle3")

        # and it actually trains to the configured step cap
        step = run.run()
        self.assertEqual(step, 2)
        self.assertTrue(
            os.path.islink(os.path.join(run_config["output_dir"], "cli-gate-latest"))
        )

        # Resume through the public config/Trainer/CheckpointManager seam, not
        # just backend.load_state_dict; require the same final draft weights.
        from specforge.training.checkpoint import CheckpointManager

        checkpoint = os.path.join(cfg.output_dir, "cli-gate-step1")
        final = CheckpointManager.read_resume_state(
            os.path.join(cfg.output_dir, "cli-gate-latest")
        )
        self.assertEqual(final["backend"]["metadata"]["backend"], training_backend)
        resumed_cfg = cfg.model_copy(
            update={
                "training": cfg.training.model_copy(update={"resume_from": checkpoint})
            }
        )
        torch.set_rng_state(construction_rng)
        resumed = build_application_run(resolve_run(resumed_cfg))
        self.assertEqual(resumed.trainer.global_step, 1)
        self.assertEqual(resumed.run(), 2)
        actual = CheckpointManager.read_resume_state(
            os.path.join(cfg.output_dir, "cli-gate-latest")
        )
        for key, expected in final["draft_state_dict"].items():
            torch.testing.assert_close(
                actual["draft_state_dict"][key], expected, rtol=0, atol=0
            )


class TestCliDispatch(unittest.TestCase):
    def test_train_command_dispatches_one_resolved_run(self):
        from specforge.application import ResolvedRun
        from specforge.cli import main
        from specforge.config import Config

        cfg = Config.model_validate(
            {
                "model": {"target_model_path": "t", "draft_model_config": "d"},
                "data": {"hidden_states_path": "/features"},
                "training": {"strategy": "dflash"},
            }
        )
        with (
            mock.patch("specforge.cli.load_config", return_value=cfg) as load,
            mock.patch("specforge.cli._train", return_value=3) as train,
        ):
            self.assertEqual(main(["train", "--config", "run.yaml"]), 0)
        load.assert_called_once_with("run.yaml", [])
        train.assert_called_once()
        resolved = train.call_args.args[0]
        self.assertIsInstance(resolved, ResolvedRun)
        self.assertEqual(resolved.config, cfg)
        self.assertEqual(resolved.algorithm.name, "dflash")


if __name__ == "__main__":
    unittest.main(verbosity=2)
