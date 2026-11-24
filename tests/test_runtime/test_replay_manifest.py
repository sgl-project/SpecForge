"""A replay must be bound to a complete checkpoint and the same acknowledged IDs."""

import hashlib
import json
import sqlite3
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from specforge.config import Config
from specforge.training.replay import load_replay_exclusions, prepare_replay_snapshot


def _validate_on_two_ranks(rank, rendezvous, manifest, config):
    from specforge.training.disaggregated import _load_online_replay

    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        cfg = Config.model_validate(config)
        if rank != 0:
            cfg.deployment.disaggregated.consumer_state_dir = (
                "/not-mounted-on-this-rank"
            )
        _load_online_replay(manifest, cfg)
        if rank == 0:
            with sqlite3.connect(
                Path(cfg.deployment.disaggregated.consumer_state_dir)
                / "consumer.sqlite"
            ) as db:
                db.execute("UPDATE marker SET v='3' WHERE k='global_step'")
        try:
            _load_online_replay(manifest, cfg)
        except ValueError as exc:
            assert "aligned" in str(exc)
        else:
            raise AssertionError("ledger mismatch was not propagated to every rank")
    finally:
        dist.destroy_process_group()


class TestReplayManifest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.environment = patch.dict("os.environ")
        self.environment.start()
        self.addCleanup(self.environment.stop)
        import os

        os.environ.pop("DISAGG_DB", None)
        self.index = self.root / "index.json"
        self.index.write_text(json.dumps({"records": 20, "source_sha256": "a" * 64}))
        self.cfg = Config.model_validate(
            {
                "run_id": "training-run",
                "model": {
                    "target_model_path": "target/model",
                    "target_backend": "sglang",
                },
                "data": {
                    "prompts_path": "prompts.jsonl",
                    "prompts_index_path": str(self.index),
                },
                "training": {
                    "role": "consumer",
                    "batch_size": 2,
                    "accumulation_steps": 2,
                    "num_epochs": 2,
                    "prompt_epoch_offset": 1,
                    "seed": 17,
                    "total_steps": 10,
                },
                "deployment": {
                    "mode": "disaggregated",
                    "trainer": {"nnodes": 1, "nproc_per_node": 1},
                    "disaggregated": {
                        "control_dir": str(self.root / "control"),
                        "consumer_state_dir": str(self.root / "original"),
                        "backend": "mooncake",
                    },
                },
            }
        )
        self.source = self.root / "original" / "consumer.sqlite"
        self.source.parent.mkdir()
        self.ids = {f"training-run:epoch0001-prompt{i:012d}" for i in range(8)}
        with sqlite3.connect(self.source) as db:
            db.execute("CREATE TABLE marker (k TEXT PRIMARY KEY, v TEXT)")
            db.executemany(
                "INSERT INTO marker VALUES (?, ?)",
                [("global_step", "2"), ("optimizer_durable", "true")],
            )
            db.execute("CREATE TABLE acked (sample_id TEXT PRIMARY KEY)")
            db.executemany("INSERT INTO acked VALUES (?)", [(i,) for i in self.ids])
            db.execute(
                "CREATE TABLE committed (sample_id TEXT PRIMARY KEY, ref_json TEXT NOT NULL)"
            )
            db.execute("INSERT INTO committed VALUES ('pending-ref', '{}')")
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        self.state = {
            "run_id": "training-run",
            "world_size": 1,
            "batch_size": 2,
            "accumulation_steps": 2,
            "global_step": 2,
            "draft_state_dict": {"weight": torch.tensor([1.0])},
            "replicated_optimizer_state": {"state": {"step": 2}},
        }
        torch.save(self.state, self.checkpoint / "training_state.pt")
        torch.save(
            {"rng": {"torch": torch.get_rng_state()}, "optimizer": None},
            self.checkpoint / "training_state_rank0.pt",
        )
        self.destination = self.root / "fresh"

    def prepare(self):
        manifest = prepare_replay_snapshot(self.cfg, self.checkpoint, self.destination)
        self.cfg.deployment.disaggregated.consumer_state_dir = str(self.destination)
        self.cfg.training.resume_from = str(self.checkpoint)
        return manifest

    def test_snapshot_preserves_source_and_replays_exact_ack_set(self):
        manifest = self.prepare()
        self.assertEqual(load_replay_exclusions(manifest, self.cfg, 20), self.ids)
        self.cfg.training.role = "producer"
        self.cfg.training.resume_from = None
        self.assertEqual(load_replay_exclusions(manifest, self.cfg, 20), self.ids)
        with sqlite3.connect(self.source) as db:
            self.assertEqual(
                db.execute("SELECT count(*) FROM committed").fetchone()[0], 1
            )
        with sqlite3.connect(self.destination / "consumer.sqlite") as db:
            self.assertEqual(
                db.execute("SELECT count(*) FROM committed").fetchone()[0], 0
            )
        self.assertEqual(json.loads(manifest.read_text())["prompt_seed"], 17)
        with self.assertRaises(FileExistsError):
            prepare_replay_snapshot(self.cfg, self.checkpoint, self.destination)

    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(), "requires Gloo"
    )
    def test_only_rank_zero_reads_ledger_and_all_ranks_receive_failure(self):
        manifest = self.prepare()
        mp.spawn(
            _validate_on_two_ranks,
            args=(str(self.root / "gloo"), str(manifest), self.cfg.model_dump()),
            nprocs=2,
            join=True,
        )

    def test_missing_optimizer_state_is_rejected(self):
        del self.state["replicated_optimizer_state"]
        torch.save(self.state, self.checkpoint / "training_state.pt")
        with self.assertRaisesRegex(ValueError, "optimizer or RNG"):
            self.prepare()

    def test_ledger_ahead_of_checkpoint_is_rejected_without_creating_destination(self):
        with sqlite3.connect(self.source) as db:
            db.execute("UPDATE marker SET v='3' WHERE k='global_step'")
        with self.assertRaisesRegex(ValueError, "aligned"):
            self.prepare()
        self.assertFalse(self.destination.exists())

    def test_checkpoint_config_mismatch_and_missing_shard_are_rejected(self):
        self.state["run_id"] = "another-run"
        torch.save(self.state, self.checkpoint / "training_state.pt")
        with self.assertRaisesRegex(ValueError, "run_id"):
            self.prepare()
        self.state["run_id"] = "training-run"
        torch.save(self.state, self.checkpoint / "training_state.pt")
        (self.checkpoint / "training_state_rank0.pt").unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()
        self.assertFalse(self.destination.exists())

    def test_run_batch_seed_and_corpus_drift_are_rejected(self):
        manifest = self.prepare()
        for section, field, value, message in (
            (self.cfg, "run_id", "another-run", "run_id"),
            (self.cfg.training, "batch_size", 3, "global_batch"),
            (self.cfg.training, "seed", 99, "prompt_seed"),
            (self.cfg.training, "prompt_epoch_offset", 2, "prompt_epoch_offset"),
        ):
            with self.subTest(field=field):
                original = getattr(section, field)
                setattr(section, field, value)
                with self.assertRaisesRegex(ValueError, message):
                    load_replay_exclusions(manifest, self.cfg)
                setattr(section, field, original)
        self.index.write_text(json.dumps({"records": 20, "source_sha256": "b" * 64}))
        with self.assertRaisesRegex(ValueError, "corpus_sha256"):
            load_replay_exclusions(manifest, self.cfg)

    def test_fresh_weights_cannot_be_used_with_an_acknowledged_replay(self):
        manifest = self.prepare()
        self.cfg.training.resume_from = None
        with self.assertRaisesRegex(ValueError, "training.resume_from"):
            load_replay_exclusions(manifest, self.cfg)

    def test_different_checkpoint_bytes_are_rejected(self):
        manifest = self.prepare()
        torch.save({"rng": {}}, self.checkpoint / "training_state_rank0.pt")
        with self.assertRaisesRegex(ValueError, "checkpoint digest"):
            load_replay_exclusions(manifest, self.cfg)

    def test_wrong_ack_set_even_with_same_count_is_rejected(self):
        manifest = self.prepare()
        with sqlite3.connect(self.destination / "consumer.sqlite") as db:
            db.execute(
                "UPDATE acked SET sample_id=? WHERE sample_id=?",
                ("training-run:epoch0001-prompt000000000010", min(self.ids)),
            )
        with self.assertRaisesRegex(ValueError, "exact acknowledged"):
            load_replay_exclusions(manifest, self.cfg)

    def test_tampered_or_duplicate_ack_file_is_rejected(self):
        manifest = self.prepare()
        ack_path = self.destination / "acked-ids.txt"
        ack_path.write_text(ack_path.read_text() + min(self.ids) + "\n")
        with self.assertRaisesRegex(ValueError, "ID digest"):
            load_replay_exclusions(manifest, self.cfg)
        payload = json.loads(manifest.read_text())
        payload["acked_ids_sha256"] = hashlib.sha256(ack_path.read_bytes()).hexdigest()
        manifest.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            load_replay_exclusions(manifest, self.cfg)

    def test_stale_committed_refs_are_rejected(self):
        manifest = self.prepare()
        with sqlite3.connect(self.destination / "consumer.sqlite") as db:
            db.execute("INSERT INTO committed VALUES ('stale-ref', '{}')")
        with self.assertRaisesRegex(ValueError, "fresh ledger"):
            load_replay_exclusions(manifest, self.cfg)

    def test_ack_count_and_legacy_single_epoch_plan_are_rejected(self):
        with sqlite3.connect(self.source) as db:
            db.execute("DELETE FROM acked WHERE sample_id=?", (min(self.ids),))
        with self.assertRaisesRegex(ValueError, "sample count"):
            self.prepare()
        with sqlite3.connect(self.source) as db:
            db.execute("INSERT INTO acked VALUES (?)", (min(self.ids),))
        self.cfg.training.num_epochs = 1
        self.cfg.training.prompt_epoch_offset = 0
        with self.assertRaisesRegex(ValueError, "epoch-tagged"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
