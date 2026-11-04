"""The DeepSeek-V4.1 DSpark converter: dequantization, naming, expert fusion, round trips."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
from torch.testing import assert_close
from transformers import Qwen3Config

from specforge.export.deepseek_v41_dspark import (
    FP4_GROUP,
    FP8_BLOCK,
    check_against_draft,
    convert_official_to_specforge,
    convert_specforge_to_official,
    dequantize_fp4,
    dequantize_fp8,
    export_official,
    import_official,
    official_draft_config,
    write_checkpoint,
)
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.modeling.target.checkpoint import load_checkpoint_tensors

CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "deepseek-v4.1-flash-moe-dspark.json"
)
LAYERS, EXPERTS = 2, 4
# E2M1: sign, two exponent bits, one mantissa bit.
E2M1 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _config():
    """The checked-in V4.1 config shrunk to two stages with 128-aligned linears."""
    payload = json.loads(CONFIG.read_text())
    payload["dflash_config"].update(
        target_layer_ids=[1], markov_rank=8, mask_token_id=63
    )
    payload.update(
        hidden_size=256,
        num_attention_heads=4,
        head_dim=128,
        qk_rope_head_dim=64,
        q_lora_rank=128,
        o_groups=2,
        o_lora_rank=128,
        n_routed_experts=EXPERTS,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        intermediate_size=128,
        vocab_size=64,
        num_target_layers=4,
        num_hidden_layers=LAYERS,
        layer_types=["full_attention"] * LAYERS,
        ffn_layer_types=["moe"] * LAYERS,
        max_position_embeddings=64,
    )
    return payload


def _draft_state():
    torch.manual_seed(0)
    model = DSparkDraftModel(Qwen3Config(**_config()))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_()  # also the zero-initialized sinks and biases
        for layer in model.layers:
            layer.mlp.gate.expert_bias.normal_()
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _e8m0(exponents):
    return exponents.to(torch.uint8).view(torch.float8_e8m0fnu)


def _quantized_official(official):
    """Official tensors with exactly representable FP8 / FP4 data and power-of-two scales."""
    torch.manual_seed(1)
    quantized, expected = {}, {}
    for name, tensor in official.items():
        rest = name.split(".", 2)[2]
        scale_name = name[: -len(".weight")] + ".scale"
        if "ffn.experts." in rest:
            out, inp = tensor.shape
            codes = torch.randint(0, 16, (out, inp), dtype=torch.uint8)
            exponents = torch.randint(125, 130, (out, inp // FP4_GROUP))
            quantized[name] = (codes[:, 0::2] | (codes[:, 1::2] << 4)).view(torch.int8)
            quantized[scale_name] = _e8m0(exponents)
            values = torch.tensor(E2M1)[codes.long()].view(out, -1, FP4_GROUP)
            expected[name] = (values * _e8m0(exponents).float()[:, :, None]).view(
                out, inp
            )
        elif (
            tensor.dtype == torch.bfloat16
            and tensor.ndim == 2
            and all(size % FP8_BLOCK == 0 for size in tensor.shape)
        ):
            out, inp = tensor.shape
            codes = torch.randn(out, inp).to(torch.float8_e4m3fn)
            exponents = torch.randint(120, 134, (out // FP8_BLOCK, inp // FP8_BLOCK))
            quantized[name] = codes
            quantized[scale_name] = _e8m0(exponents)
            blocks = codes.float().view(
                out // FP8_BLOCK, FP8_BLOCK, inp // FP8_BLOCK, FP8_BLOCK
            )
            expected[name] = (blocks * _e8m0(exponents).float()[:, None, :, None]).view(
                out, inp
            )
        else:
            quantized[name] = expected[name] = tensor
            continue
        expected[name] = expected[name].to(torch.bfloat16)
    return quantized, expected


class DeepseekV41DSparkConverterTest(unittest.TestCase):
    def test_fp8_blocks_scale_by_their_e8m0_entry(self):
        weight = torch.ones(FP8_BLOCK, 2 * FP8_BLOCK).to(torch.float8_e4m3fn)
        scale = _e8m0(torch.tensor([[128, 126]]))  # 2 and 0.5, one per 32x32 block
        output = dequantize_fp8(weight, scale)
        self.assertEqual(output.dtype, torch.bfloat16)
        assert_close(
            output[:, :FP8_BLOCK], torch.full((FP8_BLOCK, FP8_BLOCK), 2.0).bfloat16()
        )
        assert_close(
            output[:, FP8_BLOCK:], torch.full((FP8_BLOCK, FP8_BLOCK), 0.5).bfloat16()
        )

    def test_fp4_decodes_the_low_nibble_first(self):
        packed = torch.zeros(1, FP4_GROUP // 2, dtype=torch.uint8)
        packed[0, 0], packed[0, 1] = 0xF2, 0x30  # codes 2, 15, 0, 3
        output = dequantize_fp4(packed.view(torch.int8), _e8m0(torch.tensor([[128]])))
        self.assertEqual(output.dtype, torch.bfloat16)
        assert_close(output[0, :4].float(), torch.tensor([2.0, -12.0, 0.0, 3.0]))
        self.assertEqual(output[0, 4:].abs().sum().item(), 0.0)

    def test_round_trip_through_the_official_layout(self):
        state = _draft_state()
        official = export_official(state)
        self.assertLessEqual(
            {
                "mtp.0.attn.wq_a.weight",
                "mtp.0.attn.attn_sink",
                "mtp.0.main_proj.weight",
                "mtp.0.main_norm.weight",
                "mtp.0.ffn.experts.3.w3.weight",
                "mtp.1.ffn.gate.bias",
                "mtp.1.ffn.gate.bias_vl",
                "mtp.1.norm.weight",
                "mtp.1.markov_head.embed.weight",
                "mtp.1.markov_head.head.weight",
                "mtp.1.confidence_head.proj.weight",
            },
            set(official),
        )
        self.assertNotIn("mtp.1.main_proj.weight", official)
        self.assertNotIn("mtp.0.norm.weight", official)
        self.assertFalse({k for k in official if "hc_head" in k})  # single-pass mHC
        # The image-token routing bias is written as a copy of the text one.
        bias, bias_vl = official["mtp.0.ffn.gate.bias"], official["mtp.0.ffn.gate.bias_vl"]
        assert_close(bias_vl, bias)
        self.assertEqual(official["mtp.0.ffn.gate.bias_vl"].dtype, torch.float32)
        self.assertEqual(official["mtp.0.attn.attn_sink"].dtype, torch.float32)
        self.assertEqual(official["mtp.0.hc_attn_fn"].dtype, torch.float32)
        self.assertEqual(official["mtp.0.ffn.gate.bias"].dtype, torch.float32)
        self.assertEqual(official["mtp.0.attn.wq_a.weight"].dtype, torch.bfloat16)
        back = import_official(official)  # drops bias_vl again
        self.assertEqual(set(back), set(state))
        for key, value in state.items():
            assert_close(back[key], value.to(back[key].dtype), atol=0, rtol=0, msg=key)
        check_against_draft(back, _config())

    def test_quantized_official_tensors_import_exactly(self):
        quantized, expected = _quantized_official(export_official(_draft_state()))
        self.assertTrue(any(v.dtype == torch.float8_e4m3fn for v in quantized.values()))
        self.assertTrue(any(v.dtype == torch.int8 for v in quantized.values()))
        imported = import_official(quantized)
        reference = import_official(expected)
        self.assertEqual(set(imported), set(reference))
        for key, value in reference.items():
            assert_close(imported[key], value, atol=0, rtol=0, msg=key)

    def test_directories_convert_both_ways(self):
        state = _draft_state()
        quantized, expected = _quantized_official(export_official(state))
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "official")
            draft = os.path.join(directory, "draft")
            official = os.path.join(directory, "exported")
            quantized["embed.weight"] = torch.zeros(4, 8)  # a target tensor to ignore
            write_checkpoint(quantized, source, max_shard_size="1MB")
            self.assertTrue(
                os.path.exists(os.path.join(source, "model.safetensors.index.json"))
            )
            config_path = os.path.join(directory, "draft.json")
            with open(config_path, "w") as handle:
                json.dump(_config(), handle)
            convert_official_to_specforge(source, draft, config_path)
            # The config says bf16; load in fp32 to compare the fp32 official tensors exactly.
            loaded = AutoDraftModel.from_pretrained(draft, dtype=torch.float32)
            reference = import_official(expected)
            for key, value in loaded.state_dict().items():
                assert_close(
                    value.to(reference[key].dtype),
                    reference[key],
                    atol=0,
                    rtol=0,
                    msg=key,
                )

            official_config = os.path.join(directory, "official.json")
            with open(official_config, "w") as handle:
                json.dump(
                    {
                        "text_config": {"dspark_block_size": 5, "hc_mult": 4},
                        "quantization_config": {"expert_dtype": "fp4"},
                    },
                    handle,
                )
            convert_specforge_to_official(draft, official, official_config)
            with open(os.path.join(official, "config.json")) as handle:
                self.assertEqual(
                    json.load(handle),
                    {"text_config": {"dspark_block_size": 5, "hc_mult": 4}},
                )
            exported = load_checkpoint_tensors(official)
            self.assertEqual(set(exported), set(expected))
            for key, value in expected.items():
                assert_close(
                    exported[key],
                    value.to(exported[key].dtype),
                    atol=0,
                    rtol=0,
                    msg=key,
                )

    def test_official_config_drops_the_quantization_fields(self):
        official = {
            "quantization_config": {"quant_method": "fp8", "expert_dtype": "fp4"},
            "text_config": {"hc_mult": 4},
        }
        expected = {"text_config": {"hc_mult": 4}}
        self.assertEqual(official_draft_config(official), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
