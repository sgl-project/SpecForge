"""A canonical draft with MoE layers builds, checkpoints, restores and exports."""

import copy
import json
import os
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from specforge.checkpoint import save_draft
from specforge.config.draft.layer import LayerSpec
from specforge.config.draft.mlp import DenseFFNSpec
from specforge.export.to_hf import export_to_hf
from specforge.modeling import build_draft_model
from specforge.modeling.draft.layers.mlp import MoEFFN
from specforge.modeling.draft.models import DSparkDraftModel
from tests.test_modeling.test_canonical_dspark import model_config, model_inputs
from tests.test_modeling.test_draft_moe_ffn import expert_spec, moe_spec, router_spec


def moe_model_config(expert_bias=True):
    """A dense layer followed by a MoE layer, as drafts of DeepSeek-style targets."""
    node = moe_spec(
        experts=expert_spec(16),
        shared_expert=expert_spec(16),
        shared_expert_gate=True,
        router=router_spec(expert_bias=expert_bias),
    )
    return replace(
        model_config(),
        mlp_configs={
            "dense": DenseFFNSpec(intermediate_size=64, hidden_act="silu"),
            "moe": node,
        },
        layer_configs={
            "full_dense": LayerSpec(attention="full", mlp="dense"),
            "full_moe": LayerSpec(attention="full", mlp="moe"),
        },
        layer_types=("full_dense", "full_moe"),
    )


def moe_layers(model):
    return [module for module in model.modules() if isinstance(module, MoEFFN)]


class CanonicalMoETest(unittest.TestCase):
    def test_build_forward_backward_and_activation_checkpointing(self):
        config = moe_model_config()
        model = build_draft_model(config)
        (layer,) = moe_layers(model)
        self.assertIs(layer.config, config.mlp_configs["moe"])
        checkpointed = copy.deepcopy(model)
        checkpointed.gradient_checkpointing_enable()
        args, previous = model_inputs(config)
        output = model(*args, predecessor_ids=previous)
        expected = checkpointed(*args, predecessor_ids=previous)
        torch.testing.assert_close(
            (expected.logits, expected.confidence_scores),
            (output.logits, output.confidence_scores),
            rtol=0,
            atol=0,
        )
        (output.logits.sum() + output.confidence_scores.sum()).backward()
        (expected.logits.sum() + expected.confidence_scores.sum()).backward()
        # The MoE layer's routing travels out as a typed output; dense layers add none.
        (routing,) = output.expert_routing
        self.assertEqual(tuple(routing.expert_ids.shape), (16, 2))
        self.assertTrue(
            torch.equal(routing.expert_ids, expected.expert_routing[0].expert_ids)
        )
        self.assertEqual(tuple(routing.scores.shape), (16, 4))
        self.assertTrue(routing.scores.requires_grad)
        torch.testing.assert_close(
            routing.scores, expected.expert_routing[0].scores, rtol=0, atol=0
        )
        dense = build_draft_model(model_config())
        self.assertIsNone(dense(*args, predecessor_ids=previous).expert_routing)
        others = dict(checkpointed.named_parameters())
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            self.assertIsNotNone(parameter.grad, name)
            torch.testing.assert_close(
                parameter.grad, others[name].grad, rtol=0, atol=0
            )

    def test_restore_keeps_the_fp32_expert_bias_and_checks_the_node(self):
        config = moe_model_config()
        model = build_draft_model(config)
        (layer,) = moe_layers(model)
        with torch.no_grad():
            layer.gate.expert_bias.copy_(torch.tensor([0.5, -0.25, 0.0, 0.125]))
        args, previous = model_inputs(config)
        output = model(*args, predecessor_ids=previous).logits
        with TemporaryDirectory() as path:
            model.save_pretrained(path)
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(dtype=dtype):
                    restored = DSparkDraftModel.from_pretrained(path, dtype=dtype)
                    (restored_layer,) = moe_layers(restored)
                    self.assertEqual(restored_layer.gate.weight.dtype, dtype)
                    self.assertEqual(
                        restored_layer.gate.expert_bias.dtype, torch.float32
                    )
                    torch.testing.assert_close(
                        restored_layer.gate.expert_bias,
                        layer.gate.expert_bias,
                        rtol=0,
                        atol=0,
                    )
                    if dtype == torch.float32:
                        result = restored(*args, predecessor_ids=previous).logits
                        torch.testing.assert_close(result, output, rtol=0, atol=0)
            # A node without the bias rejects a checkpoint that carries one.
            config_path = Path(path, "config.json")
            payload = json.loads(config_path.read_text())
            payload["mlp_configs"]["moe"]["router"]["expert_bias"] = False
            config_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "unexpected_keys"):
                DSparkDraftModel.from_pretrained(path)

    def test_export_validates_router_state_and_runs_standalone(self):
        config = moe_model_config()
        model = build_draft_model(config).eval()
        state = model.state_dict()
        bias_key = next(key for key in state if key.endswith("gate.expert_bias"))
        model.validate_export_state(state)
        with self.assertRaisesRegex(ValueError, "router state"):
            model.validate_export_state(
                {key: value for key, value in state.items() if key != bias_key}
            )
        with self.assertRaisesRegex(ValueError, "fp32"):
            model.validate_export_state({**state, bias_key: state[bias_key].bfloat16()})
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.save(state, root / "expected_state.pt")
            save_draft(model, root / "source")
            export_to_hf(str(root / "source"), None, str(root / "hf"))
            bundled = {path.name for path in (root / "hf").glob("*.py")}
            self.assertIn("specforge_modeling_draft_layers_mlp_moe_experts.py", bundled)
            script = (
                Path(__file__).resolve().parent.parent
                / "test_checkpoint"
                / "helpers"
                / "check_standalone_export.py"
            )
            result = subprocess.run(
                [sys.executable, "-I", str(script), str(root / "hf")],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "TORCHDYNAMO_DISABLE": "1",
                    "HF_MODULES_CACHE": str(root / "modules"),
                },
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
