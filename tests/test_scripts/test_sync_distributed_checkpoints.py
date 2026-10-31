"""Dependency-light tests for the non-shared-filesystem checkpoint relay."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "disagg" / "sync_distributed_checkpoints.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_relay_example", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RELAY_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELAY_MODULE)


class DistributedCheckpointRelayTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="checkpoint_relay_")
        self.root = Path(self._tmp.name)
        self.run_id = "relay-test"
        self.relays = []

    def tearDown(self):
        for relay in self.relays:
            relay._httpd.server_close()
        self._tmp.cleanup()

    def _checkpoint(self, root: Path, step: int, ranks: range) -> Path:
        checkpoint = root / "output" / f"{self.run_id}-step{step}"
        checkpoint.mkdir(parents=True)
        if 0 in ranks:
            (checkpoint / "training_state.pt").write_bytes(
                f"shared-step-{step}".encode()
            )
        for rank in ranks:
            (checkpoint / f"training_state_rank{rank}.pt").write_bytes(
                f"rank-{rank}-step-{step}".encode()
            )
        return checkpoint

    def _relay(
        self,
        root: Path,
        *,
        local_ranks: range,
        peer_ranks: range,
        max_archives: int = 2,
    ):
        relay = RELAY_MODULE.CheckpointRelay(
            SimpleNamespace(
                run_root=str(root),
                run_id=self.run_id,
                local_ranks=tuple(local_ranks),
                peer_ranks=tuple(peer_ranks),
                peer_url="file:///not-configured",
                poll_s=0.01,
                max_archives=max_archives,
                serve_host="127.0.0.1",
                serve_port=0,
            )
        )
        self.relays.append(relay)
        return relay

    def test_two_nodes_assemble_complete_checkpoints_and_bound_archives(self):
        node0 = self.root / "node0"
        node1 = self.root / "node1"
        for step in (1, 2, 3):
            self._checkpoint(node0, step, range(0, 2))
            self._checkpoint(node1, step, range(2, 4))

        relay0 = self._relay(node0, local_ranks=range(0, 2), peer_ranks=range(2, 4))
        relay1 = self._relay(node1, local_ranks=range(2, 4), peer_ranks=range(0, 2))
        relay0.peer_url = relay1.relay_dir.as_uri()
        relay1.peer_url = relay0.relay_dir.as_uri()

        relay0._publish_local()
        relay1._publish_local()
        relay0._pull_peer()
        relay1._pull_peer()

        for root in (node0, node1):
            for step in (2, 3):
                checkpoint = root / "output" / f"{self.run_id}-step{step}"
                expected = {"training_state.pt"}
                expected.update(f"training_state_rank{rank}.pt" for rank in range(4))
                self.assertTrue(
                    expected.issubset(path.name for path in checkpoint.iterdir())
                )

        for relay in (relay0, relay1):
            manifest = json.loads(
                (relay.relay_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual([entry["step"] for entry in manifest["entries"]], [2, 3])
            local_archives = [
                path
                for path in relay.relay_dir.glob("*.tar")
                if not path.name.startswith("peer-")
            ]
            peer_archives = list(relay.relay_dir.glob("peer-*.tar"))
            self.assertEqual(len(local_archives), 2)
            self.assertEqual(len(peer_archives), 2)

        self._checkpoint(node0, 4, range(0, 2))
        self._checkpoint(node1, 4, range(2, 4))
        relay0._publish_local()
        relay1._publish_local()
        relay0._pull_peer()
        relay1._pull_peer()

        for relay in (relay0, relay1):
            names = {path.name for path in relay.relay_dir.glob("*.tar")}
            self.assertFalse(any("step2-" in name for name in names))
            self.assertTrue(any("step3-" in name for name in names))
            self.assertTrue(any("step4-" in name for name in names))

        # A transiently absent or already-pruned output tree must not erase the
        # relay's bounded recovery copies.
        shutil.rmtree(node0 / "output")
        shutil.rmtree(node1 / "output")
        relay0._publish_local()
        relay1._publish_local()
        for relay in (relay0, relay1):
            manifest = json.loads(
                (relay.relay_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual([entry["step"] for entry in manifest["entries"]], [3, 4])

    def test_rank_ranges_and_archive_retention_are_validated(self):
        self.assertTrue(os.access(SCRIPT, os.X_OK))
        self.assertEqual(RELAY_MODULE._rank_range("8-15"), tuple(range(8, 16)))
        with self.assertRaisesRegex(Exception, "rank range"):
            RELAY_MODULE._rank_range("15-8")
        self.assertEqual(RELAY_MODULE._positive_int("3"), 3)
        with self.assertRaisesRegex(Exception, "at least 1"):
            RELAY_MODULE._positive_int("0")

    def test_peer_archive_name_cannot_escape_or_disagree_with_step(self):
        relay = self._relay(
            self.root / "node0",
            local_ranks=range(0, 2),
            peer_ranks=range(2, 4),
        )
        base_entry = {
            "step": 3,
            "sha256": "0" * 64,
            "files": [
                "training_state_rank2.pt",
                "training_state_rank3.pt",
            ],
        }
        with self.assertRaisesRegex(ValueError, "unexpected peer archive"):
            relay._install_peer_archive(
                {**base_entry, "archive": "../outside-step3-ranks2-3.tar"}
            )
        with self.assertRaisesRegex(ValueError, "unexpected peer archive"):
            relay._install_peer_archive(
                {
                    **base_entry,
                    "archive": f"{self.run_id}-step4-ranks2-3.tar",
                }
            )
        with self.assertRaisesRegex(ValueError, "unexpected peer archive name"):
            relay._download("../outside.tar", "0" * 64)


if __name__ == "__main__":
    unittest.main()
