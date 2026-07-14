"""Dependency-light contracts for the real overfit/serving gate orchestration."""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATE_DIR = ROOT / "scripts" / "gates"
COMMON = GATE_DIR / "_e2e_common.sh"
OVERFIT = GATE_DIR / "run_disaggregated_overfit_gate.sh"
SERVING = GATE_DIR / "run_dflash_serving_gate.sh"
NORMALIZE = GATE_DIR / "normalize_dflash_export.py"
GATE_README = GATE_DIR / "README.md"
TRAINING_GUIDE = ROOT / "docs" / "basic_usage" / "training.md"
CUSTOMIZATION_GUIDE = ROOT / "docs" / "advanced_features" / "customization.md"
CONFIG_README = ROOT / "examples" / "configs" / "README.md"


def command_line(output: str, marker: str, suffix: str = "") -> list[str]:
    line = next(line for line in output.splitlines() if marker in line)
    self_contained = line.removeprefix("+ ")
    if suffix:
        self_contained = self_contained.split(suffix, 1)[0]
    return shlex.split(self_contained)


class TestGateOrchestration(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="specforge gate ")
        self.root = Path(self._tmp.name)
        self.config = self.root / "run config.yaml"
        self.config.write_text("run_id: dry-run\n", encoding="utf-8")
        self.draft_config = self.root / "domino config.json"
        self.draft_config.write_text(
            json.dumps(
                {
                    "architectures": ["DominoDraftModel"],
                    "auto_map": {"AutoModel": "domino.DominoDraftModel"},
                    "block_size": 16,
                    "dflash_config": {
                        "mask_token_id": 123,
                        "target_layer_ids": [1, 9, 17, 25, 33],
                        "projector_type": "domino",
                        "gru_hidden_dim": 1024,
                    },
                }
            ),
            encoding="utf-8",
        )
        self.source = self.root / "source data.jsonl"
        self.source.write_text("{}\n", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def clean_env(self) -> dict[str, str]:
        prefixes = (
            "CAPTURE_",
            "CONSUMER_",
            "DISAGG_",
            "GATE_",
            "MOONCAKE_",
            "SERVING_",
        )
        explicit = {
            "BLOCK_SIZE",
            "CHECKPOINT_PATH",
            "CONFIG",
            "DRAFT_CONFIG_PATH",
            "MAX_STEPS",
            "NNODES",
            "NPROC_PER_NODE",
            "PROMPT_ARTIFACT_PATH",
            "RUN_ID",
            "RUN_SERVING_GATE",
            "RUN_TAG",
            "SOURCE_DATA_PATH",
            "SPECFORGE_BIN",
            "START_CAPTURE_SERVER",
            "START_MOONCAKE",
            "TARGET_MODEL_PATH",
            "WORK_DIR",
        }
        return {
            key: value
            for key, value in os.environ.items()
            if key not in explicit and not key.startswith(prefixes)
        }

    def run_script(
        self, script: Path, env: dict[str, str]
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(script)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_shells_are_syntax_valid_and_use_the_canonical_boundaries(self):
        for script in (COMMON, OVERFIT, SERVING):
            result = subprocess.run(
                ["bash", "-n", str(script)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

        overfit_source = OVERFIT.read_text(encoding="utf-8")
        serving_source = SERVING.read_text(encoding="utf-8")
        common_source = COMMON.read_text(encoding="utf-8")
        self.assertIn("examples/disagg/run_online.sh", overfit_source)
        self.assertNotIn("train_domino.py", overfit_source)
        self.assertNotIn("train_dflash.py", overfit_source)
        self.assertNotIn("train --config", overfit_source)
        self.assertIn("export --to hf", serving_source)
        self.assertIn("run_dflash_chat_serving_gate.py", serving_source)
        self.assertIn("trap gate_cleanup EXIT", common_source)
        self.assertIn("setsid", common_source)
        self.assertIn('kill "-$signal" -- "-$pid"', common_source)
        self.assertIn("_gate_signal_process TERM", common_source)
        self.assertIn("_gate_signal_process KILL", common_source)
        self.assertIn(
            "CHECKPOINT_PATH=$CONSUMER_OUTPUT_DIR/$RUN_ID-step$MAX_STEPS",
            overfit_source,
        )
        self.assertIn(
            'gate_require_file "$CHECKPOINT_PATH/training_state.pt"',
            overfit_source,
        )

    def test_cleanup_terminates_and_reaps_an_owned_background_process(self):
        log_path = self.root / "background process.log"
        script = r"""
source "$1"
gate_install_cleanup_traps
gate_start_service sleeper "$3" "$2" -c 'import time; time.sleep(30)'
pid=$GATE_LAST_PID
gate_stop_services
if kill -0 "$pid" 2>/dev/null; then
    printf 'background process still alive: %s\n' "$pid" >&2
    exit 1
fi
"""
        result = subprocess.run(
            [
                "bash",
                "-c",
                script,
                "gate-cleanup-test",
                str(COMMON),
                sys.executable,
                str(log_path),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            # A missed setsid startup signal blocks on the 30-second sleeper;
            # keep enough scheduling headroom for slower macOS CI while still
            # failing well before that broken path could finish naturally.
            timeout=8,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_overfit_dry_run_constructs_the_full_unified_command_plan(self):
        work_dir = self.root / "overfit work"
        env = self.clean_env()
        env.update(
            {
                "CONFIG": str(self.config),
                "TARGET_MODEL_PATH": "org/target-model",
                "DRAFT_CONFIG_PATH": str(self.draft_config),
                "SOURCE_DATA_PATH": str(self.source),
                "WORK_DIR": str(work_dir),
                "RUN_ID": "dry-overfit",
                "MAX_STEPS": "10",
                "NPROC_PER_NODE": "2",
                "CAPTURE_GPUS": "0,1",
                "CAPTURE_TP": "2",
                "CONSUMER_GPUS": "2,3",
                "GATE_DRY_RUN": "1",
                "RUN_SERVING_GATE": "true",
                "PYTHON": sys.executable,
            }
        )

        result = self.run_script(OVERFIT, env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(work_dir.exists())
        self.assertNotIn("train_domino.py", result.stdout)
        self.assertNotIn("train_dflash.py", result.stdout)

        select = command_line(result.stdout, "select_overfit_sample.py")
        self.assertIn(str(self.source), select)
        self.assertIn(str(work_dir / "single_sample.jsonl"), select)
        self.assertIn("--require-untruncated", select)

        capture = command_line(result.stdout, "sglang.launch_server", " > ")
        self.assertIn("--enable-spec-capture", capture)
        self.assertEqual(capture[capture.index("--tp-size") + 1], "2")
        layer_index = capture.index("--spec-capture-aux-layer-ids")
        self.assertEqual(
            capture[layer_index + 1 : layer_index + 6], ["1", "9", "17", "25", "33"]
        )

        producer = command_line(result.stdout, "run_online.sh --role producer", " > ")
        consumer = command_line(
            result.stdout, "run_online.sh --role consumer", " 2>&1 | tee "
        )
        for command in (producer, consumer):
            self.assertIn("training.num_epochs=20", command)
            self.assertIn("training.max_steps=10", command)
            self.assertIn("training.batch_size=1", command)
            self.assertIn("training.accumulation_steps=1", command)
            self.assertIn("tracking.report_to=none", command)
            self.assertIn("deployment.trainer.nproc_per_node=2", command)
            self.assertIn(
                f"data.train_data_path={work_dir / 'single_sample.jsonl'}", command
            )
        self.assertNotIn("NPROC_PER_NODE=2", consumer)

        check = command_line(result.stdout, "check_overfit_metrics.py")
        self.assertEqual(check[check.index("--expected-step") + 1], "10")
        serving = command_line(result.stdout, "run_dflash_serving_gate.sh")
        self.assertIn("bash", serving)
        self.assertIn(
            f"CHECKPOINT_PATH={work_dir / 'consumer' / 'dry-overfit-step10'}",
            serving,
        )

    def test_serving_dry_run_constructs_export_server_and_strict_check(self):
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        prompt = self.root / "prompt.json"
        prompt.write_text("{}\n", encoding="utf-8")
        work_dir = self.root / "serving work"
        env = self.clean_env()
        env.update(
            {
                "CHECKPOINT_PATH": str(checkpoint),
                "TARGET_MODEL_PATH": "org/target-model",
                "DRAFT_CONFIG_PATH": str(self.draft_config),
                "PROMPT_ARTIFACT_PATH": str(prompt),
                "WORK_DIR": str(work_dir),
                "SERVING_GPUS": "4,5",
                "SERVING_TP": "2",
                "SERVING_PORT": "31001",
                "SERVED_MODEL": "gate-target",
                "GATE_DRY_RUN": "true",
                "PYTHON": sys.executable,
                "SGLANG_PYTHON": sys.executable,
                "SPECFORGE_BIN": "specforge",
            }
        )

        result = self.run_script(SERVING, env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(work_dir.exists())

        export = command_line(result.stdout, "export --to hf")
        self.assertEqual(export[:4], ["specforge", "export", "--to", "hf"])
        self.assertEqual(export[export.index("--checkpoint") + 1], str(checkpoint))
        self.assertEqual(
            export[export.index("--output-dir") + 1], str(work_dir / "draft_hf")
        )

        normalize = command_line(result.stdout, "normalize_dflash_export.py")
        self.assertEqual(normalize[normalize.index("--block-size") + 1], "16")

        server = command_line(result.stdout, "sglang.launch_server", " > ")
        self.assertIn("CUDA_VISIBLE_DEVICES=4,5", server)
        self.assertEqual(server[server.index("--tp-size") + 1], "2")
        self.assertEqual(server[server.index("--port") + 1], "31001")
        self.assertEqual(server[server.index("--speculative-algorithm") + 1], "DFLASH")
        self.assertEqual(
            server[server.index("--speculative-dflash-block-size") + 1], "16"
        )

        checker = command_line(result.stdout, "run_dflash_chat_serving_gate.py")
        self.assertEqual(checker[checker.index("--served-model") + 1], "gate-target")
        self.assertEqual(checker[checker.index("--prompt-json-path") + 1], str(prompt))

    def test_fresh_work_directory_is_enforced_before_external_commands(self):
        existing = self.root / "existing"
        existing.mkdir()
        env = self.clean_env()
        env.update(
            {
                "CONFIG": str(self.config),
                "TARGET_MODEL_PATH": "org/target-model",
                "DRAFT_CONFIG_PATH": str(self.draft_config),
                "SOURCE_DATA_PATH": str(self.source),
                "WORK_DIR": str(existing),
                "GATE_DRY_RUN": "1",
                "PYTHON": sys.executable,
            }
        )
        result = self.run_script(OVERFIT, env)
        self.assertEqual(result.returncode, 2)
        self.assertIn("WORK_DIR must be fresh", result.stderr)
        self.assertNotIn("select_overfit_sample.py", result.stdout)

    def test_serving_refuses_an_occupied_port_before_export(self):
        checkpoint = self.root / "occupied-port-checkpoint"
        checkpoint.mkdir()
        prompt = self.root / "occupied-port-prompt.json"
        prompt.write_text("{}\n", encoding="utf-8")
        work_dir = self.root / "occupied port serving"
        python_with_spaces = self.root / "python executable"
        python_with_spaces.write_text(
            "#!/usr/bin/env bash\n"
            'if [[ "$1" == - ]]; then exit 1; fi\n'
            'exec "$REAL_PYTHON" "$@"\n',
            encoding="utf-8",
        )
        python_with_spaces.chmod(0o755)
        env = self.clean_env()
        env.update(
            {
                "CHECKPOINT_PATH": str(checkpoint),
                "TARGET_MODEL_PATH": "org/target-model",
                "DRAFT_CONFIG_PATH": str(self.draft_config),
                "PROMPT_ARTIFACT_PATH": str(prompt),
                "WORK_DIR": str(work_dir),
                "SERVING_HOST": "127.0.0.1",
                "SERVING_PORT": "31009",
                "PYTHON": str(python_with_spaces),
                "SGLANG_PYTHON": str(python_with_spaces),
                "SPECFORGE_BIN": str(python_with_spaces),
                "REAL_PYTHON": sys.executable,
            }
        )
        result = self.run_script(SERVING, env)

        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("SERVING_PORT is already in use", result.stderr)
        self.assertNotIn("export --to hf", result.stdout)
        self.assertFalse(work_dir.exists())

    def test_gate_readme_exposes_orchestration_and_keeps_leaf_checks_reusable(self):
        readme = GATE_README.read_text(encoding="utf-8")
        self.assertIn("bash scripts/gates/run_disaggregated_overfit_gate.sh", readme)
        self.assertIn("bash scripts/gates/run_dflash_serving_gate.sh", readme)
        self.assertIn("GATE_DRY_RUN=1", readme)
        self.assertIn("examples/disagg/run_online.sh", readme)
        self.assertIn("Reusable leaf checks", readme)

    def test_docs_distinguish_hf_replicas_from_target_weight_tp(self):
        training = TRAINING_GUIDE.read_text(encoding="utf-8")
        customization = CUSTOMIZATION_GUIDE.read_text(encoding="utf-8")
        recipes = CONFIG_README.read_text(encoding="utf-8")

        for document in (training, customization, recipes):
            self.assertIn("DFlash", document)
            self.assertIn("Domino", document)
            self.assertIn("complete target", document)
            self.assertIn("replica", document)

        self.assertIn("SGLang target TP + target-DP", training)
        self.assertIn("SGLang target TP + target-DP", recipes)
        self.assertIn("Batch partitioning is therefore not evidence", training)
        self.assertIn("with `model.target_backend: hf` do not", customization)
        self.assertIn("output partitioning is independent", customization)
        self.assertIn("does not shard target weights", recipes)
        self.assertNotIn(
            "| DFlash | Yes, target TP + target-DP |",
            training,
        )
        self.assertNotIn(
            "| Domino | target TP + target-DP |",
            recipes,
        )


class TestNormalizeDFlashExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "normalize_dflash_export", NORMALIZE
        )
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_normalizes_dispatch_without_dropping_domino_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "architectures": ["DominoDraftModel"],
                        "auto_map": {"AutoModel": "domino.DominoDraftModel"},
                        "block_size": 16,
                        "dflash_config": {
                            "projector_type": "domino",
                            "gru_hidden_dim": 1024,
                        },
                    }
                ),
                encoding="utf-8",
            )

            normalized = self.module.normalize_export(str(path), 16)

            self.assertEqual(normalized["architectures"], ["DFlashDraftModel"])
            self.assertNotIn("auto_map", normalized)
            self.assertEqual(normalized["dflash_config"]["projector_type"], "domino")
            self.assertEqual(normalized["dflash_config"]["gru_hidden_dim"], 1024)
            self.assertEqual(json.loads(path.read_text()), normalized)

    def test_rejects_a_mismatched_block_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "block_size": 8,
                        "dflash_config": {"projector_type": "dflash"},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "expected 16"):
                self.module.normalize_export(str(path), 16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
