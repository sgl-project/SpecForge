# coding=utf-8
"""Executable contract tests for the SGLang spec-capture patch artifact."""

import hashlib
import importlib.util
import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
_PATCH = _ROOT / "patches" / "sglang" / "v0.5.14" / "spec-capture.patch"
_MIGRATION_PATCH = (
    _ROOT / "patches" / "sglang" / "v0.5.14" / "spec-capture-v1-to-v2.patch"
)
_INSTALLER = _ROOT / "scripts" / "apply_sglang_spec_capture_patch.sh"


def _extract_sink() -> Path:
    lines = _PATCH.read_text().splitlines()
    target = "+++ b/python/sglang/srt/spec_capture_sink.py"
    start = lines.index(target) + 1
    while start < len(lines) and not lines[start].startswith("@@"):
        start += 1
    hunk_header = lines[start]
    declared_lines = int(hunk_header.split("+1,", 1)[1].split(" ", 1)[0])
    start += 1
    source = []
    for line in lines[start:]:
        if line.startswith("diff --git "):
            break
        if line.startswith("+"):
            source.append(line[1:])
        elif line == "\\ No newline at end of file":
            continue
        else:
            raise AssertionError(f"unexpected non-addition in new sink hunk: {line}")
    if len(source) != declared_lines:
        raise AssertionError(
            f"sink hunk declares {declared_lines} lines but contains {len(source)}"
        )
    root = Path(tempfile.mkdtemp(prefix="sglang-capture-patch-"))
    path = root / "spec_capture_sink.py"
    path.write_text("\n".join(source) + "\n")
    return path


def _load_sink_module():
    path = _extract_sink()
    spec = importlib.util.spec_from_file_location("_patched_spec_capture_sink", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSGLangCapturePatch(unittest.TestCase):
    def test_patch_is_well_formed(self):
        subprocess.run(
            ["git", "apply", "--numstat", str(_PATCH)],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_patch_captures_final_pre_norm_hidden_state(self):
        text = _PATCH.read_text(encoding="utf-8")
        self.assertIn("diff --git a/python/sglang/srt/models/qwen2.py", text)
        self.assertIn("if self.end_layer in self.layers_to_capture:", text)
        self.assertLess(
            text.index("if self.end_layer in self.layers_to_capture:"),
            text.index(
                "if hidden_states.shape[0] != 0:", text.index("models/qwen2.py")
            ),
        )

    def test_patch_copies_only_requested_capture_artifacts(self):
        text = _PATCH.read_text(encoding="utf-8")
        start = text.index("+    def _append_spec_capture_states(")
        end = text.index("+    def _sink_spec_capture(", start)
        body = text[start:end]

        self.assertIn(
            '+        if "aux" in requested_artifacts:\n'
            "+            req.spec_capture_aux.append(\n"
            "+                logits_output.hidden_states[start:end].cpu().clone()",
            body,
        )
        self.assertIn(
            "+        if (\n"
            '+            "last_hidden" in requested_artifacts\n'
            "+            and logits_output.last_hidden_states is not None\n"
            "+        ):\n"
            "+            req.spec_capture_last_hidden.append(\n"
            "+                logits_output.last_hidden_states[start:end].cpu().clone()",
            body,
        )

    def test_previous_patch_migration_round_trip(self):
        legacy = (
            '        """Accumulate captured rows as CPU tensors for the Mooncake sink.\n'
            "\n"
            "        Same offset arithmetic as ``_append_prefill_hidden_states`` but keeps\n"
            "        tensor slices (aux concat in ``hidden_states``, post-norm last in\n"
            "        ``last_hidden_states``) rather than the JSON-able response payload.\n"
            '        """\n'
            "        start = hidden_state_offset\n"
            "        end = start + len(req.origin_input_ids)\n"
            "        req.spec_capture_aux.append(\n"
            "            logits_output.hidden_states[start:end].cpu().clone()\n"
            "        )\n"
            "        if logits_output.last_hidden_states is not None:\n"
            "            req.spec_capture_last_hidden.append(\n"
            "                logits_output.last_hidden_states[start:end].cpu().clone()\n"
            "            )\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = (
                root
                / "sglang/srt/managers/scheduler_components/batch_result_processor.py"
            )
            target.parent.mkdir(parents=True)
            target.write_text(legacy, encoding="utf-8")
            command = ["patch", "--batch", "-p2", "-d", str(root)]

            subprocess.run(
                [*command, "--forward"],
                input=_MIGRATION_PATCH.read_bytes(),
                check=True,
                capture_output=True,
            )
            current = target.read_text(encoding="utf-8")
            self.assertIn('if "aux" in requested_artifacts:', current)
            self.assertIn('"last_hidden" in requested_artifacts', current)

            subprocess.run(
                [*command, "--reverse"],
                input=_MIGRATION_PATCH.read_bytes(),
                check=True,
                capture_output=True,
            )
            self.assertEqual(target.read_text(encoding="utf-8"), legacy)

    def test_installer_pins_patch_and_migration_digests(self):
        installer = _INSTALLER.read_text(encoding="utf-8")
        patch_digest = hashlib.sha256(_PATCH.read_bytes()).hexdigest()
        migration_digest = hashlib.sha256(_MIGRATION_PATCH.read_bytes()).hexdigest()
        self.assertIn(f'EXPECTED_PATCH_SHA256="{patch_digest}"', installer)
        self.assertIn(f'EXPECTED_MIGRATION_SHA256="{migration_digest}"', installer)

    def test_remove_exact_uses_force_without_granting_a_read_lease(self):
        module = _load_sink_module()
        root = Path(tempfile.mkdtemp(prefix="capture-remove-"))
        sink = module.SpecCaptureSink(
            store_id="run0",
            auth_token="secret",
            max_sample_bytes=1 << 20,
            inventory_db_path=str(root / "inventory.db"),
            lifecycle_db_path=str(root / "lifecycle.db"),
            aux_layer_ids=[1, 2],
        )

        class Store:
            status = -704

            def __init__(self):
                self.calls = []

            def remove(self, key, force=False):
                self.calls.append((key, force))
                return self.status

            def is_exist(self, key):
                raise AssertionError("remove cleanup must not grant a read lease")

        store = Store()
        sink._connect = lambda: store
        sink._remove_exact("run0/s0/g1/hidden_states", force=True)
        self.assertEqual(store.calls, [("run0/s0/g1/hidden_states", True)])

        store.status = -706
        with self.assertRaisesRegex(RuntimeError, "status -706"):
            sink._remove_exact("run0/s0/g1/hidden_states", force=False)
        self.assertEqual(store.calls[-1], ("run0/s0/g1/hidden_states", False))

    def _sink(self, *, max_bytes=1 << 20, inventory_path=None):
        module = _load_sink_module()
        inventory_path = inventory_path or str(
            Path(tempfile.mkdtemp(prefix="capture-inventory-")) / "inventory.db"
        )
        sink = module.SpecCaptureSink(
            store_id="run0",
            auth_token="secret",
            max_sample_bytes=max_bytes,
            inventory_db_path=inventory_path,
            lifecycle_db_path=inventory_path + ".lifecycle",
            aux_layer_ids=[1, 2],
        )
        writes = []
        sink._remove_exact = lambda key, *, force: None
        sink._put_tensor = lambda key, tensor, **_kwargs: writes.append(
            (key, tensor.clone())
        )
        return sink, writes

    @staticmethod
    def _request(**overrides):
        request = {
            "auth_token": "secret",
            "store_id": "run0",
            "sample_id": "run0:s0",
            "gen": 1,
            "replace": False,
            "features": {"aux": "hidden_states"},
            "passthrough": [
                {
                    "name": "input_ids",
                    "data": [1, 2, 3],
                    "shape": [1, 3],
                    "dtype": "int64",
                }
            ],
        }
        request.update(overrides)
        return request

    def test_valid_authenticated_request_uses_server_namespace(self):
        sink, writes = self._sink()
        result = sink.put_sample(
            self._request(),
            aux=torch.ones(3, 4, dtype=torch.bfloat16),
            last_hidden=None,
        )
        self.assertEqual(result["store_id"], "run0")
        self.assertEqual(
            [key for key, _ in writes],
            ["run0/run0:s0/g1/hidden_states", "run0/run0:s0/g1/input_ids"],
        )

    def test_lifecycle_is_planned_before_first_write_and_then_resident(self):
        sink, writes = self._sink()
        states_at_write = []

        def record_write(key, tensor, **_kwargs):
            row = sink._lifecycle.execute(
                "SELECT state, feature_names_json, estimated_bytes FROM "
                "mooncake_objects WHERE store_id=? AND sample_id=? AND generation=?",
                ("run0", "run0:s0", 1),
            ).fetchone()
            states_at_write.append(row)
            writes.append((key, tensor.clone()))

        sink._put_tensor = record_write
        sink.put_sample(
            self._request(),
            aux=torch.ones(3, 4, dtype=torch.bfloat16),
            last_hidden=None,
        )

        self.assertEqual([row[0] for row in states_at_write], ["planned", "planned"])
        self.assertEqual(states_at_write[0][1], '["hidden_states","input_ids"]')
        self.assertEqual(states_at_write[0][2], 48)
        state = sink._lifecycle.execute(
            "SELECT state FROM mooncake_objects WHERE store_id=? AND sample_id=? "
            "AND generation=?",
            ("run0", "run0:s0", 1),
        ).fetchone()[0]
        self.assertEqual(state, "resident")

    def test_lifecycle_bytes_exclude_unrequested_artifacts(self):
        sink, writes = self._sink()
        sink.put_sample(
            self._request(),
            aux=torch.ones(3, 4, dtype=torch.bfloat16),
            # SGLang produces this tensor for DFlash capture too, but the request
            # does not persist it and the owner must not account for it.
            last_hidden=torch.ones(3, 8, dtype=torch.bfloat16),
        )

        row = sink._lifecycle.execute(
            "SELECT feature_names_json, estimated_bytes FROM mooncake_objects "
            "WHERE store_id=? AND sample_id=? AND generation=?",
            ("run0", "run0:s0", 1),
        ).fetchone()
        self.assertEqual(row, ('["hidden_states","input_ids"]', 48))
        self.assertEqual(
            [key for key, _ in writes],
            ["run0/run0:s0/g1/hidden_states", "run0/run0:s0/g1/input_ids"],
        )

    def test_failed_write_cleans_planned_lifecycle(self):
        sink, _ = self._sink()
        removed = []
        calls = 0

        def fail_second_write(key, tensor, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected put failure")

        sink._put_tensor = fail_second_write
        sink._remove_exact = lambda key, *, force: removed.append((key, force))

        with self.assertRaisesRegex(RuntimeError, "injected put failure"):
            sink.put_sample(
                self._request(),
                aux=torch.ones(3, 4, dtype=torch.bfloat16),
                last_hidden=None,
            )

        self.assertEqual(
            removed,
            [
                ("run0/run0:s0/g1/hidden_states", True),
                ("run0/run0:s0/g1/input_ids", True),
            ],
        )
        state = sink._lifecycle.execute(
            "SELECT state FROM mooncake_objects WHERE store_id=? AND sample_id=? "
            "AND generation=?",
            ("run0", "run0:s0", 1),
        ).fetchone()[0]
        self.assertEqual(state, "cleaned")

    def test_failed_cleanup_leaves_planned_row_for_owner_takeover(self):
        sink, _ = self._sink()

        def fail_write(key, tensor, **_kwargs):
            raise RuntimeError("injected put failure")

        def fail_remove(key, *, force):
            raise RuntimeError("injected remove failure")

        sink._put_tensor = fail_write
        sink._remove_exact = fail_remove

        with self.assertRaisesRegex(RuntimeError, "injected remove failure"):
            sink.put_sample(
                self._request(),
                aux=torch.ones(3, 4, dtype=torch.bfloat16),
                last_hidden=None,
            )

        with sqlite3.connect(sink.lifecycle_db_path) as lifecycle:
            state = lifecycle.execute(
                "SELECT state FROM mooncake_objects WHERE store_id=? AND "
                "sample_id=? AND generation=?",
                ("run0", "run0:s0", 1),
            ).fetchone()[0]
        self.assertEqual(state, "planned")

    def test_capability_namespace_schema_and_quota_are_enforced(self):
        cases = (
            ({"auth_token": "wrong"}, "capability"),
            ({"store_id": "attacker"}, "server-owned namespace"),
            ({"sample_id": "other:s0"}, "prefixed"),
            ({"features": {"aux": "../../escape"}}, "not allowed"),
            (
                {
                    "passthrough": [
                        {
                            "name": "input_ids",
                            "data": [1],
                            "shape": [1, 2],
                            "dtype": "int64",
                        }
                    ]
                },
                "requires 2 values",
            ),
        )
        for override, error in cases:
            with self.subTest(override=override):
                sink, writes = self._sink()
                with self.assertRaisesRegex((ValueError, PermissionError), error):
                    sink.put_sample(
                        self._request(**override),
                        aux=torch.ones(3, 4, dtype=torch.bfloat16),
                        last_hidden=None,
                    )
                self.assertEqual(writes, [])

        sink, writes = self._sink(max_bytes=1)
        with self.assertRaisesRegex(ValueError, "above the"):
            sink.put_sample(
                self._request(),
                aux=torch.ones(3, 4, dtype=torch.bfloat16),
                last_hidden=None,
            )
        self.assertEqual(writes, [])

    def test_response_loss_retry_is_idempotent_and_reclaims_prior_generation(self):
        inventory = str(
            Path(tempfile.mkdtemp(prefix="capture-response-loss-")) / "inventory.db"
        )
        first, first_writes = self._sink(inventory_path=inventory)
        result = first.put_sample(
            self._request(gen=1),
            aux=torch.ones(3, 4, dtype=torch.bfloat16),
            last_hidden=None,
        )
        write_count = len(first_writes)

        repeated = first.put_sample(
            self._request(gen=1),
            aux=torch.zeros(3, 4, dtype=torch.bfloat16),
            last_hidden=None,
        )
        self.assertEqual(repeated, result)
        self.assertEqual(len(first_writes), write_count)

        restarted, second_writes = self._sink(inventory_path=inventory)
        removed = []
        restarted._remove_exact = lambda key, *, force: removed.append((key, force))
        retried = restarted.put_sample(
            self._request(gen=2),
            aux=torch.ones(3, 4, dtype=torch.bfloat16),
            last_hidden=None,
        )
        self.assertEqual(retried["gen"], 2)
        self.assertEqual(
            removed,
            [
                ("run0/run0:s0/g1/hidden_states", False),
                ("run0/run0:s0/g1/input_ids", False),
            ],
        )
        self.assertTrue(all("/g2/" in key for key, _ in second_writes))

    def test_replacement_journal_recovers_at_every_prior_key_delete(self):
        for crash_after in (0, 1, 2):
            with self.subTest(crash_after=crash_after):
                inventory = str(
                    Path(tempfile.mkdtemp(prefix="capture-replacement-journal-"))
                    / "inventory.db"
                )
                keys = set()

                def attach(sink):
                    sink._put_tensor = lambda key, tensor, **_kwargs: keys.add(key)
                    sink._remove_exact = lambda key, *, force: keys.discard(key)

                first, _ = self._sink(inventory_path=inventory)
                attach(first)
                first.put_sample(
                    self._request(gen=1),
                    aux=torch.ones(3, 4, dtype=torch.bfloat16),
                    last_hidden=None,
                )
                old_keys = {
                    "run0/run0:s0/g1/hidden_states",
                    "run0/run0:s0/g1/input_ids",
                }
                self.assertEqual(keys, old_keys)

                interrupted, _ = self._sink(inventory_path=inventory)
                remove_calls = 0

                def crash_during_remove(key, *, force):
                    nonlocal remove_calls
                    self.assertFalse(force)
                    remove_calls += 1
                    if crash_after == 0 and remove_calls == 1:
                        raise RuntimeError("injected replacement crash")
                    keys.discard(key)
                    if remove_calls == crash_after:
                        raise RuntimeError("injected replacement crash")

                interrupted._put_tensor = (
                    lambda key, tensor, **_kwargs: keys.add(key)
                )
                interrupted._remove_exact = crash_during_remove
                with self.assertRaisesRegex(RuntimeError, "replacement crash"):
                    interrupted.put_sample(
                        self._request(gen=2),
                        aux=torch.ones(3, 4, dtype=torch.bfloat16),
                        last_hidden=None,
                    )

                row = interrupted._inventory.execute(
                    "SELECT generation, state, prior_generation, prior_keys_json "
                    "FROM captures WHERE sample_id=?",
                    ("run0:s0",),
                ).fetchone()
                self.assertEqual(row[:3], (2, "replacing", 1))
                self.assertEqual(set(json.loads(row[3])), old_keys)

                delayed, delayed_writes = self._sink(inventory_path=inventory)
                with self.assertRaisesRegex(ValueError, "stale"):
                    delayed.put_sample(
                        self._request(gen=1),
                        aux=torch.ones(3, 4, dtype=torch.bfloat16),
                        last_hidden=None,
                    )
                self.assertEqual(delayed_writes, [])

                resumed, _ = self._sink(inventory_path=inventory)
                attach(resumed)
                result = resumed.put_sample(
                    self._request(gen=2),
                    aux=torch.ones(3, 4, dtype=torch.bfloat16),
                    last_hidden=None,
                )
                self.assertEqual(result["gen"], 2)
                self.assertEqual(
                    keys,
                    {
                        "run0/run0:s0/g2/hidden_states",
                        "run0/run0:s0/g2/input_ids",
                    },
                )
                final = resumed._inventory.execute(
                    "SELECT generation, state, prior_generation, prior_keys_json "
                    "FROM captures WHERE sample_id=?",
                    ("run0:s0",),
                ).fetchone()
                self.assertEqual(final, (2, "committed", None, None))
                states = dict(
                    resumed._lifecycle.execute(
                        "SELECT generation, state FROM mooncake_objects WHERE "
                        "store_id=? AND sample_id=? ORDER BY generation",
                        ("run0", "run0:s0"),
                    ).fetchall()
                )
                self.assertEqual(states, {1: "cleaned", 2: "resident"})

    def test_inventory_schema_migrates_existing_capture_database(self):
        inventory = str(
            Path(tempfile.mkdtemp(prefix="capture-inventory-migration-"))
            / "inventory.db"
        )
        with sqlite3.connect(inventory) as connection:
            connection.execute(
                "CREATE TABLE captures (sample_id TEXT PRIMARY KEY, "
                "generation INTEGER NOT NULL, keys_json TEXT NOT NULL, "
                "result_json TEXT, state TEXT NOT NULL)"
            )
        sink, _ = self._sink(inventory_path=inventory)
        columns = {
            row[1] for row in sink._inventory.execute("PRAGMA table_info(captures)")
        }
        self.assertIn("prior_generation", columns)
        self.assertIn("prior_keys_json", columns)


if __name__ == "__main__":
    unittest.main()
