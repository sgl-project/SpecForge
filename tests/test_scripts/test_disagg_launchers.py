"""Dependency-light contracts for the thin disaggregated examples."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ONLINE = ROOT / "examples" / "disagg" / "run_online.sh"
OFFLINE = ROOT / "examples" / "disagg" / "run_offline.sh"
OFFLINE_TWO_NODE = ROOT / "examples" / "disagg" / "run_offline_2node.sh"
TWO_NODE = ROOT / "examples" / "disagg" / "run_qwen3_8b_dflash_disagg_2node.sh"
INKLING_TWO_NODE = ROOT / "examples" / "disagg" / "run_inkling_dspark_disagg_2node.sh"
KIMI_K3_CAPTURE_PATCH = (
    ROOT / "patches" / "sglang" / "kimi-k3-f8493a4" / "spec-capture.patch"
)


class DisaggregatedWrapperTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="disagg_wrapper_")
        self.root = Path(self._tmp.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.capture = self.root / "args.txt"
        self.config = self.root / "run config.yaml"
        self.config.write_text("run_id: wrapper-test\n", encoding="utf-8")
        executable = self.bin_dir / "specforge"
        executable.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'printf "%s\\n" "$@" > "$CAPTURE_PATH"\n',
            encoding="utf-8",
        )
        executable.chmod(0o755)

    def tearDown(self):
        self._tmp.cleanup()

    def _env(self, *, include_config=True):
        env = os.environ.copy()
        env["PATH"] = f"{self.bin_dir}{os.pathsep}{env.get('PATH', '')}"
        env["CAPTURE_PATH"] = str(self.capture)
        if include_config:
            env["CONFIG"] = str(self.config)
        else:
            env.pop("CONFIG", None)
        return env

    def _run(self, wrapper, *args, include_config=True):
        return subprocess.run(
            [str(wrapper), *args],
            cwd=ROOT,
            env=self._env(include_config=include_config),
            text=True,
            capture_output=True,
            check=False,
        )

    def test_wrappers_are_executable_and_syntax_valid(self):
        for wrapper in (ONLINE, OFFLINE):
            with self.subTest(wrapper=wrapper.name):
                self.assertTrue(os.access(wrapper, os.X_OK))
                result = subprocess.run(
                    ["bash", "-n", str(wrapper)],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_wrappers_only_delegate_to_the_unified_cli(self):
        for wrapper in (ONLINE, OFFLINE):
            with self.subTest(wrapper=wrapper.name):
                result = self._run(
                    wrapper,
                    "--role",
                    "consumer",
                    "--plan",
                    "training.batch_size=2",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    self.capture.read_text(encoding="utf-8").splitlines(),
                    [
                        "train",
                        "--config",
                        str(self.config),
                        "--role",
                        "consumer",
                        "--plan",
                        "training.batch_size=2",
                    ],
                )

    def test_wrappers_validate_only_the_config_boundary(self):
        for wrapper in (ONLINE, OFFLINE):
            with self.subTest(wrapper=wrapper.name):
                result = self._run(wrapper, include_config=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn("set CONFIG", result.stderr)

    def test_topology_and_transport_logic_are_not_duplicated_in_shell(self):
        forbidden = (
            "torchrun",
            "NPROC_PER_NODE",
            "NNODES",
            "NODE_RANK",
            "DISAGG_DB",
            "DISAGG_REF_CHANNEL",
            "DISAGG_MANIFEST",
            "MOONCAKE_",
        )
        for wrapper in (ONLINE, OFFLINE):
            source = wrapper.read_text(encoding="utf-8")
            for token in forbidden:
                with self.subTest(wrapper=wrapper.name, token=token):
                    self.assertNotIn(token, source)

    def test_help_describes_auto_and_explicit_roles(self):
        for wrapper in (ONLINE, OFFLINE):
            result = self._run(wrapper, "--help")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--role", result.stdout)
            self.assertIn("producer and consumer", result.stdout)

    def test_kimi_k3_capture_patch_only_copies_features_on_the_writer_rank(self):
        source = KIMI_K3_CAPTURE_PATCH.read_text(encoding="utf-8")
        self.assertIn("self.output_streamer.ps.attn_tp_rank != 0", source)
        self.assertNotIn(".cpu().clone()", source)
        self.assertIn('getattr(logits_output, "_spec_capture_aux_cpu", None)', source)
        self.assertIn("logits_output.hidden_states.cpu()", source)
        self.assertIn('"aux" in features', source)
        self.assertIn('"last_hidden" in features', source)
        self.assertIn("_should_copy_hidden_states_to_cpu", source)
        self.assertIn("self.ps.attn_tp_rank == 0", source)
        self.assertIn("self.logits_output.last_hidden_states = _async_d2h(", source)
        self.assertIn("len(chunks) == 1", source)
        self.assertIn("ThreadPoolExecutor(", source)
        self.assertIn('getattr(store, "batch_put_from", None)', source)
        self.assertIn("SGLANG_SPEC_CAPTURE_MAX_PENDING_BATCHES", source)
        self.assertIn("req.finished() and req.spec_capture_result is None", source)

    def test_two_node_wrapper_keeps_training_on_the_unified_cli(self):
        self.assertTrue(os.access(TWO_NODE, os.X_OK))
        syntax = subprocess.run(
            ["bash", "-n", str(TWO_NODE)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

        shared_root = self.root / "shared-attempt"
        consumer_state = self.root / "node-local-consumer-state"
        base_env = self._env()
        base_env.update(
            {
                "NUM_NODES": "2",
                "HEAD_IP": "10.0.0.1",
                "DISAGG_STORE_ID": "two-node-test",
                "DISAGG_RUN_ROOT": str(shared_root),
                "DISAGG_CONSUMER_STATE_DIR": str(consumer_state),
                "DRY_RUN": "1",
            }
        )
        outputs = {}
        for rank in ("0", "1"):
            with self.subTest(rank=rank):
                env = {**base_env, "NODE_RANK": rank}
                result = subprocess.run(
                    [str(TWO_NODE), "training.max_steps=1"],
                    cwd=ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                outputs[rank] = result.stdout

        self.assertIn("mooncake_master", outputs["0"])
        self.assertIn("sglang.launch_server", outputs["0"])
        self.assertIn("specforge train", outputs["0"])
        self.assertIn("--role producer", outputs["0"])
        self.assertIn("specforge train", outputs["1"])
        self.assertIn("--role consumer", outputs["1"])
        self.assertIn(
            "deployment.disaggregated.consumer_state_dir=" + str(consumer_state),
            outputs["1"],
        )
        self.assertNotIn(
            "deployment.disaggregated.consumer_state_dir=" + str(shared_root),
            outputs["1"],
        )
        self.assertNotIn("run_disagg_dflash.py", "".join(outputs.values()))
        self.assertNotIn("torchrun", "".join(outputs.values()))
        self.assertFalse(shared_root.exists())

    def test_inkling_two_node_wrapper_pins_the_validated_server_contract(self):
        self.assertTrue(os.access(INKLING_TWO_NODE, os.X_OK))
        syntax = subprocess.run(
            ["bash", "-n", str(INKLING_TWO_NODE)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

        env = self._env()
        env.update(
            {
                "NODE_RANK": "0",
                "NUM_NODES": "2",
                "HEAD_IP": "10.0.0.1",
                "DISAGG_STORE_ID": "inkling-two-node-test",
                "DISAGG_RUN_ROOT": str(self.root / "inkling-shared-attempt"),
                "DRY_RUN": "1",
            }
        )
        result = subprocess.run(
            [str(INKLING_TWO_NODE), "training.max_steps=1"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        output = result.stdout
        for expected in (
            "thinkingmachines/Inkling",
            "--tp-size 4",
            "--spec-capture-aux-layer-ids 5 17 35 47 59",
            "--attention-backend fa4",
            "--quantization modelopt_fp4",
            "--mamba-radix-cache-strategy extra_buffer",
            "training.accumulation_steps=128",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, output)
        self.assertNotIn("--disable-radix-cache", output)

        source = INKLING_TWO_NODE.read_text(encoding="utf-8")
        self.assertIn("SGLANG_ENABLE_UNIFIED_RADIX_TREE", source)
        self.assertIn("SGLANG_OPT_USE_INKLING_CUSTOM_AR", source)

    def test_offline_two_node_wrapper_dispatches_roles_to_the_unified_cli(self):
        self.assertTrue(os.access(OFFLINE_TWO_NODE, os.X_OK))
        syntax = subprocess.run(
            ["bash", "-n", str(OFFLINE_TWO_NODE)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

        for rank, role in (("0", "producer"), ("1", "consumer")):
            with self.subTest(rank=rank):
                env = self._env()
                env.update({"RCLI_NODE_RANK": rank, "RCLI_NUM_NODES": "2"})
                result = subprocess.run(
                    [str(OFFLINE_TWO_NODE), "training.max_steps=1"],
                    cwd=ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    self.capture.read_text(encoding="utf-8").splitlines(),
                    [
                        "train",
                        "--config",
                        str(self.config),
                        "--role",
                        role,
                        "training.max_steps=1",
                    ],
                )

        source = OFFLINE_TWO_NODE.read_text(encoding="utf-8")
        for token in ("torchrun", "MOONCAKE_", "DISAGG_DB", "DISAGG_MANIFEST"):
            with self.subTest(token=token):
                self.assertNotIn(token, source)

    def test_offline_two_node_wrapper_rejects_missing_or_invalid_rank(self):
        for rank in ("", "2"):
            with self.subTest(rank=rank or "missing"):
                env = self._env()
                env.update(
                    {
                        "RCLI_NODE_RANK": rank,
                        "NODE_RANK": "",
                        "RCLI_NUM_NODES": "2",
                    }
                )
                result = subprocess.run(
                    [str(OFFLINE_TWO_NODE)],
                    cwd=ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("must be 0 or 1", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
