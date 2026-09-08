"""Dependency-light contracts for the retained gate helpers."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATE_DIR = ROOT / "scripts" / "gates"
COMMON = GATE_DIR / "_e2e_common.sh"
NORMALIZE = GATE_DIR / "normalize_dflash_export.py"


class TestGateHelpers(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="specforge gate ")
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_common_shell_is_syntax_valid(self):
        result = subprocess.run(
            ["bash", "-n", str(COMMON)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

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
            timeout=8,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_port_report_lists_all_conflicts_without_stopping_the_shell(self):
        fake_python = self.root / "fake python"
        fake_python.write_text(
            "#!/usr/bin/env bash\n"
            'case "$3" in\n'
            "    32001 | 37551) exit 1 ;;\n"
            "    *) exit 0 ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        script = r"""
source "$1"
gate_report_tcp_ports "$2" 127.0.0.1 \
    capture_port 32000 \
    serving_port 32001 \
    mooncake_rpc_port 37551
printf 'continued\n'
"""
        result = subprocess.run(
            [
                "bash",
                "-c",
                script,
                "gate-port-report-test",
                str(COMMON),
                str(fake_python),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("serving_port: 127.0.0.1:32001", result.stdout)
        self.assertIn("mooncake_rpc_port: 127.0.0.1:37551", result.stdout)
        self.assertNotIn("capture_port", result.stdout)
        self.assertTrue(result.stdout.endswith("continued\n"))


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

    def test_normalizes_dspark_for_sglang(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "architectures": ["DSparkDraftModel"],
                        "auto_map": {"AutoModel": "dspark.DSparkDraftModel"},
                        "block_size": 16,
                        "model_type": "qwen3",
                        "dflash_config": {
                            "projector_type": "dspark",
                            "markov_rank": 256,
                            "markov_head_type": "vanilla",
                            "enable_confidence_head": True,
                            "confidence_head_with_markov": True,
                            "mask_token_id": 248070,
                            "target_layer_ids": [2, 17, 32, 47, 62],
                        },
                    }
                ),
                encoding="utf-8",
            )

            normalized = self.module.normalize_export(str(path), 16)

            self.assertEqual(normalized["architectures"], ["Qwen3DSparkModel"])
            self.assertNotIn("auto_map", normalized)
            self.assertEqual(normalized["markov_rank"], 256)
            self.assertEqual(normalized["markov_head_type"], "vanilla")
            self.assertTrue(normalized["enable_confidence_head"])
            self.assertTrue(normalized["confidence_head_with_markov"])
            self.assertEqual(
                normalized["dflash_config"]["target_layer_ids"],
                [2, 17, 32, 47, 62],
            )
            self.assertEqual(json.loads(path.read_text()), normalized)

    def _dspark_moe_export_config(self, **overrides):
        config = {
            "architectures": ["DSparkDraftModel"],
            "auto_map": {"AutoModel": "dspark.DSparkDraftModel"},
            "block_size": 7,
            "model_type": "qwen3",
            "dflash_config": {
                "projector_type": "dspark",
                "markov_rank": 256,
                "markov_head_type": "vanilla",
                "enable_confidence_head": True,
                "confidence_head_with_markov": True,
                "moe_router_center": "sample",
                "target_layer_ids": [5, 19, 33, 47, 61],
            },
            # what specforge.export.to_hf writes for the qwen3_5_moe preset
            "moe_preset": "qwen3_5_moe",
            "n_routed_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 2048,
            "n_shared_experts": 1,
            "scoring_func": "softmax",
            "norm_topk_prob": True,
            "routed_scaling_factor": 1.0,
            "n_group": 1,
            "topk_group": 1,
            "topk_method": "greedy",
        }
        config.update(overrides)
        return config

    def test_normalizes_dspark_moe_for_sglang(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(self._dspark_moe_export_config()), encoding="utf-8"
            )

            normalized = self.module.normalize_export(str(path), 7)

            self.assertEqual(normalized["architectures"], ["Qwen3MoeDSparkModel"])
            self.assertNotIn("auto_map", normalized)
            self.assertEqual(normalized["markov_rank"], 256)
            # Qwen alias next to the DeepSeek-vocabulary field the exporter wrote
            self.assertEqual(normalized["num_experts"], 512)
            self.assertEqual(normalized["n_routed_experts"], 512)
            self.assertEqual(normalized["n_shared_experts"], 1)
            self.assertTrue(normalized["norm_topk_prob"])
            # centered router => the folded gate.bias is mandatory at load
            self.assertTrue(normalized["moe_router_bias"])
            self.assertEqual(json.loads(path.read_text()), normalized)

    def test_dspark_moe_without_centering_has_optional_router_bias(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            config = self._dspark_moe_export_config()
            del config["dflash_config"]["moe_router_center"]
            path.write_text(json.dumps(config), encoding="utf-8")
            normalized = self.module.normalize_export(str(path), 7)
            self.assertFalse(normalized["moe_router_bias"])

    def test_rejects_dspark_moe_recipes_the_sglang_model_cannot_serve(self):
        for overrides in (
            {"scoring_func": "sigmoid"},
            {"topk_method": "noaux_tc"},
            {"n_group": 8, "topk_group": 4},
            {"n_shared_experts": 2},
            {"num_experts_per_tok": 600},
            {"n_shared_experts": 1, "shared_expert_intermediate_size": 0},
            {"num_experts": 256},
        ):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "config.json"
                path.write_text(
                    json.dumps(self._dspark_moe_export_config(**overrides)),
                    encoding="utf-8",
                )
                with self.assertRaises(ValueError, msg=str(overrides)):
                    self.module.normalize_export(str(path), 7)

    def test_preserves_dflash2_architecture_and_nested_block_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "architectures": ["DFlash2DraftModel"],
                        "auto_map": {"AutoModel": "dflash2.DFlash2DraftModel"},
                        "dflash_config": {
                            "block_size": 8,
                            "conv_group_size": 16,
                            "conv_kernel_size": 2,
                            "selector_rank": 256,
                            "selector_top_k": 16,
                        },
                    }
                ),
                encoding="utf-8",
            )

            normalized = self.module.normalize_export(str(path), 8)

            self.assertEqual(normalized["architectures"], ["DFlash2DraftModel"])
            self.assertNotIn("auto_map", normalized)
            self.assertNotIn("block_size", normalized)
            self.assertEqual(normalized["dflash_config"]["block_size"], 8)
            self.assertEqual(json.loads(path.read_text()), normalized)

    def test_rejects_incomplete_dflash2_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "architectures": ["DFlash2DraftModel"],
                        "dflash_config": {
                            "block_size": 8,
                            "conv_group_size": 16,
                            "conv_kernel_size": 2,
                            "selector_rank": 256,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "selector_top_k"):
                self.module.normalize_export(str(path), 8)

    def test_rejects_mla_before_rewriting_the_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            original = {
                "architectures": ["DFlashDraftModel"],
                "auto_map": {"AutoModel": "dflash.DFlashDraftModel"},
                "block_size": 16,
                "dflash_config": {
                    "projector_type": "dflash",
                    "attention_mode": "mla",
                },
            }
            path.write_text(json.dumps(original), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "only GQA/MHA"):
                self.module.normalize_export(str(path), 16)

            self.assertEqual(json.loads(path.read_text()), original)

    def test_rejects_dspark_without_a_positive_markov_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "block_size": 16,
                        "model_type": "qwen3",
                        "dflash_config": {
                            "projector_type": "dspark",
                            "markov_rank": 0,
                            "markov_head_type": "vanilla",
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "positive integer markov_rank"):
                self.module.normalize_export(str(path), 16)

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
