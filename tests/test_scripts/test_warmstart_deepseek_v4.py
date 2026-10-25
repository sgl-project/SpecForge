"""Native V4 checkpoint conversion and validation."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from scripts.warmstart_deepseek_v4_dspark import dequantize_weight, import_checkpoint
from tests.test_modeling.test_draft.test_deepseek_v4_dspark import tiny_config


class V4DequantizationTest(unittest.TestCase):
    def test_fp4_low_nibble_first_and_typed_e8m0(self):
        # Each packed byte contains +.5 then -1.5; scale value is 4, not exponent 4.
        weight = torch.full((2, 16), 0xB1, dtype=torch.uint8).view(torch.int8)
        scale = torch.full((2, 1), 4.0).to(torch.float8_e8m0fnu)
        output = dequantize_weight(weight, scale, (2, 32))
        torch.testing.assert_close(output, torch.tensor([2.0, -6.0]).repeat(2, 16))
        with self.assertRaisesRegex(ValueError, "typed E8M0"):
            dequantize_weight(weight, scale.view(torch.uint8), (2, 32))

    def test_fp8_block_scales_and_partial_edge(self):
        weight = torch.ones(129, 130).to(torch.float8_e4m3fn)
        scale = torch.tensor([[1.0, 2.0], [4.0, 8.0]]).to(torch.float8_e8m0fnu)
        output = dequantize_weight(weight, scale, (129, 130))
        self.assertEqual(output[127, 127].item(), 1.0)
        self.assertEqual(output[127, 128].item(), 2.0)
        self.assertEqual(output[128, 127].item(), 4.0)
        self.assertEqual(output[128, 129].item(), 8.0)

    def test_bad_scale_or_shape_fails(self):
        with self.assertRaises(ValueError):
            dequantize_weight(torch.ones(2, 2).to(torch.float8_e4m3fn), None, (2, 2))
        with self.assertRaises(ValueError):
            dequantize_weight(torch.ones(2, 2), None, (3, 2))


class V4ImportTest(unittest.TestCase):
    def test_complete_import_warm_start_and_missing_key_rejection(self):
        from specforge.modeling.draft.deepseek_v4_dspark import (
            DeepseekV4DSparkDraftModel,
        )
        from specforge.training.model_loading import warm_start_draft_model

        config = tiny_config()
        config.architectures = ["DeepseekV4DSparkDraftModel"]
        original = DeepseekV4DSparkDraftModel(config)
        for stage in original.mtp:
            stage.ffn.gate.bias.copy_(
                torch.tensor([0.123456, 0.234567, 0.345678, 0.456789])
            )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, draft = root / "source", root / "draft"
            source.mkdir()
            config.save_pretrained(draft)
            payload = config.to_dict()
            payload.update(
                model_type="deepseek_v4",
                num_hidden_layers=config.num_target_layers,
                compress_ratios=[0]
                * (config.num_target_layers + config.num_hidden_layers),
                dspark_block_size=4,
                dspark_target_layer_ids=config.dflash_config["target_layer_ids"],
                dspark_noise_token_id=config.dflash_config["mask_token_id"],
                dspark_markov_rank=config.dflash_config["markov_rank"],
            )
            (source / "config.json").write_text(json.dumps(payload))
            state = original.state_dict()
            save_file(state, str(source / "model.safetensors"))
            index = {"weight_map": {k: "model.safetensors" for k in state}}
            index_path = source / "model.safetensors.index.json"
            index_path.write_text(json.dumps(index))
            output = root / "imported"
            report = import_checkpoint(
                source, draft / "config.json", output, shard_size_bytes=4096
            )
            self.assertEqual(report["block_size"], 10)
            self.assertEqual(report["randomly_initialized"], [])
            direct = DeepseekV4DSparkDraftModel.from_pretrained(
                output, dtype=torch.bfloat16
            )
            torch.testing.assert_close(
                direct.mtp[0].ffn.gate.bias,
                original.mtp[0].ffn.gate.bias,
                rtol=0,
                atol=0,
            )
            loaded = DeepseekV4DSparkDraftModel(config).to(dtype=torch.bfloat16)
            warm_start_draft_model(
                loaded, str(output), draft_config=config, strategy="dspark"
            )
            loaded.to(dtype=torch.bfloat16)
            buffers = dict(original.named_buffers())
            for key, value in loaded.state_dict().items():
                expected = state[key] if key in buffers else state[key].bfloat16()
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
            index["weight_map"].pop("mtp.0.main_proj.weight")
            index_path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "missing=.*main_proj"):
                import_checkpoint(
                    source, draft / "config.json", root / "bad", dry_run=True
                )
            self.assertFalse((root / "bad").exists())

    def test_source_type_and_fp8_grid_are_checked_before_import(self):
        from scripts.warmstart_deepseek_v4_dspark import validate_source_config

        with self.assertRaisesRegex(ValueError, "native DeepSeek V4"):
            validate_source_config(tiny_config(), {"model_type": "unsupported"})
        with self.assertRaisesRegex(ValueError, "128x128"):
            validate_source_config(
                tiny_config(),
                {
                    "model_type": "deepseek_v4",
                    "quantization_config": {"weight_block_size": [32, 32]},
                },
            )
