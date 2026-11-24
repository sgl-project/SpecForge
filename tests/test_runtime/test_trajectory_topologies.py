"""Placement and CLI configuration of the accelerator reference trajectories."""

import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import yaml

from specforge.application.composition import resolve_run
from specforge.config import load_config
from specforge.distributed import Topology
from specforge.launch import _shard_offline_refs
from specforge.training.canonical_data import shard_sequence
from tests.test_runtime import trajectories


class TrajectoryTopologiesTest(unittest.TestCase):
    def test_all_bundled_trainer_world_sizes_have_a_default_case(self):
        sizes = set()
        recipes = Path(__file__).resolve().parents[2] / "examples/configs"
        for path in recipes.rglob("*.yaml"):
            trainer = yaml.safe_load(path.read_text())["deployment"].get("trainer", {})
            sizes.add(trainer.get("nnodes", 1) * trainer.get("nproc_per_node", 1))
        self.assertLessEqual(
            sizes, {case.world_size for case in trajectories.every_case()}
        )

    def test_accelerator_placement_and_explicit_ranks(self):
        with (
            patch.object(trajectories, "get_device_type", return_value="cuda"),
            patch("torch.get_device_module") as devices,
            patch.object(trajectories, "_usp_available", return_value=False) as usp,
        ):
            devices.return_value.device_count.return_value = 4
            placed = trajectories.cases()
            self.assertIn(3, {case.world_size for case in placed})
            self.assertTrue(all(case.sp_ulysses_size == 1 for case in placed))
            usp.return_value = True
            self.assertTrue(
                any(case.sp_ulysses_size == 2 for case in trajectories.cases())
            )
            with TemporaryDirectory() as temp, redirect_stderr(io.StringIO()) as error:
                output = Path(temp) / "wide.pt"
                with self.assertRaises(SystemExit) as failure:
                    trajectories.main(["record", str(output), "--ranks", "8"])
                self.assertEqual(failure.exception.code, 2)
                self.assertIn("4 visible cuda devices", error.getvalue())
                self.assertFalse(output.exists())
        with patch.object(trajectories, "get_device_type", return_value="cpu"):
            cpu = trajectories.cases([3])
            # FSDP1 refuses to compute on the host; FSDP2 shards there.
            self.assertTrue(
                all(case.sharding == "NO_SHARD" for case in cpu if case.runtime == "fsdp")
            )
            self.assertIn(
                ("fsdp2", "SHARD_GRAD_OP"), {(case.runtime, case.sharding) for case in cpu}
            )
            self.assertTrue(all(case.sp_ulysses_size == 1 for case in cpu))
            self.assertIn(3, {case.world_size for case in cpu})
        with self.assertRaisesRegex(ValueError, "positive world sizes"):
            trajectories.every_case([0])

    def test_usp_case_drives_two_data_replicas_and_the_lk_loss(self):
        (case,) = (
            case for case in trajectories.every_case() if case.sp_ulysses_size > 1
        )
        self.assertEqual(case.world_size // case.sp_ulysses_size, 2)
        self.assertEqual(
            (case.window_size, case.batches_per_epoch, case.updates), (4, 6, 3)
        )

        def inspect_rank_inputs(worker, args, nprocs, join):
            _, runs = args
            config, overrides, _, _ = runs[0]
            cfg = load_config(config, overrides)
            self.assertEqual(nprocs, cfg.deployment.trainer.nproc_per_node)
            self.assertEqual(resolve_run(cfg).config.training.attention_backend, "usp")
            self.assertEqual(cfg.training.sp_ulysses_size, 2)
            self.assertEqual(cfg.training.accumulation_steps, 2)
            self.assertEqual(cfg.training.loss["terms"]["token"]["kind"], "lk_lambda")
            self.assertEqual(cfg.model.torch_dtype, "bfloat16")
            files = sorted(Path(cfg.data.hidden_states_path).glob("*.ckpt"))
            partitions = [
                _shard_offline_refs(
                    files,
                    topology=Topology(
                        world_size=case.world_size,
                        sp_ulysses_size=cfg.training.sp_ulysses_size,
                        draft_dp_rank=rank,
                    ),
                    use_usp_preprocess=True,
                    seed=cfg.training.seed,
                )
                for rank in range(2)
            ]
            self.assertTrue(set(partitions[0]).isdisjoint(partitions[1]))
            for owned in partitions:
                self.assertEqual(len(owned), case.batches_per_epoch)
                # Some supervised positions belong to each side of the same
                # sequence, so the LK reduction really needs its SP peer.
                for path in owned:
                    record = torch.load(path)
                    sample = {
                        key: record[key][None] for key in ("input_ids", "loss_mask")
                    }
                    overlap = cfg.training.objective_options["ttt_length"]
                    supervised = []
                    for rank in range(2):
                        shard = shard_sequence(
                            sample, rank=rank, size=2, overlap=overlap
                        )
                        length = shard["input_ids"].shape[1] - overlap
                        supervised.append(shard["loss_mask"][:, 1 : length + 1].any())
                    if all(supervised):
                        break
                else:
                    self.fail("no sequence has supervision on both ranks")

        with (
            patch.object(
                trajectories.multiprocessing, "spawn", side_effect=inspect_rank_inputs
            ),
            patch.object(trajectories, "_read_case", return_value=[]),
        ):
            trajectories.record([case])


if __name__ == "__main__":
    unittest.main()
