"""Warm-starting the Qwen MoE DSpark drafter from Qwen3.8-Flash-Next's MTP layer."""

import json
import os
import tempfile
import unittest

import torch
from safetensors.torch import save_file
from torch.testing import assert_close

from specforge.modeling.draft.mtp_warm_start import (
    MTP_PREFIX,
    apply_mtp_warm_start,
    mtp_layer_state,
    read_mtp_tensors,
)
from tests.test_modeling.test_qwen_moe_dspark import tiny_draft_config

try:
    from transformers.models.qwen4_exp import modeling_qwen4_exp
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
except ImportError:  # transformers < 5.16
    modeling_qwen4_exp = None

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, INTERMEDIATE = 64, 4, 2, 32, 16


def fake_mtp_tensors(experts=8, seed=0):
    """An MTP layer at the tiny drafter's shapes, with the release's names."""

    generator = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return (0.2 * torch.randn(*shape, generator=generator)).to(torch.bfloat16)

    return {
        "self_attn.q_proj.weight": randn(HEADS * HEAD_DIM * 2, HIDDEN),
        "self_attn.k_proj.weight": randn(KV_HEADS * HEAD_DIM, HIDDEN),
        "self_attn.v_proj.weight": randn(KV_HEADS * HEAD_DIM, HIDDEN),
        "self_attn.o_proj.weight": randn(HIDDEN, HEADS * HEAD_DIM),
        "self_attn.q_norm.weight": randn(HEAD_DIM),
        "self_attn.k_norm.weight": randn(HEAD_DIM),
        "self_attn.indexer.index_qk_proj.weight": randn(80, HIDDEN),
        "mlp.gate.weight": randn(experts, HIDDEN),
        "mlp.experts.gate_up_proj": randn(experts, 2 * INTERMEDIATE, HIDDEN),
        "mlp.experts.down_proj": randn(experts, HIDDEN, INTERMEDIATE),
        "mlp.shared_expert.gate_proj.weight": randn(INTERMEDIATE, HIDDEN),
        "mlp.shared_expert.up_proj.weight": randn(INTERMEDIATE, HIDDEN),
        "mlp.shared_expert.down_proj.weight": randn(HIDDEN, INTERMEDIATE),
        "mlp.shared_expert_gate.weight": randn(1, HIDDEN),
        "attn_hyper_connection.hc_norm.weight": randn(4 * HIDDEN),
    }


def draft(**overrides):
    from specforge.modeling.auto import AutoDraftModel

    torch.manual_seed(1)
    return AutoDraftModel.from_config(
        tiny_draft_config(**overrides), torch_dtype=torch.float32
    )


def write_checkpoint(directory, tensors):
    """A two-shard target checkpoint whose MTP tensors live in one shard."""

    mtp = {MTP_PREFIX + name: tensor for name, tensor in tensors.items()}
    other = {"model.language_model.embed_tokens.weight": torch.zeros(4, HIDDEN)}
    save_file(mtp, os.path.join(directory, "model-00002.safetensors"))
    save_file(other, os.path.join(directory, "model-00001.safetensors"))
    weight_map = {name: "model-00002.safetensors" for name in mtp}
    weight_map.update({name: "model-00001.safetensors" for name in other})
    with open(
        os.path.join(directory, "model.safetensors.index.json"), "w", encoding="utf-8"
    ) as stream:
        json.dump({"metadata": {}, "weight_map": weight_map}, stream)


class TestMapping(unittest.TestCase):
    def test_attention_is_copied_and_norms_move_to_the_drafters_convention(self):
        tensors = fake_mtp_tensors()
        model = draft()
        apply_mtp_warm_start(model, tensors)
        for index in range(len(model.layers)):
            attention = model.layers[index].self_attn
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                assert_close(
                    getattr(attention, name).weight,
                    tensors[f"self_attn.{name}.weight"].float(),
                )
            # The target's RMSNorm scales by (1 + w), the drafter's by w.
            x = torch.randn(3, HEAD_DIM)
            normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
            expected = normed * (1.0 + tensors["self_attn.q_norm.weight"].float())
            assert_close(attention.q_norm(x), expected, rtol=1e-2, atol=1e-2)

    @unittest.skipIf(modeling_qwen4_exp is None, "transformers has no qwen4_exp")
    def test_moe_computes_what_the_target_mtp_block_computes(self):
        tensors = fake_mtp_tensors()
        model = draft()
        apply_mtp_warm_start(model, tensors)
        target = modeling_qwen4_exp.Qwen4ExpTextSparseMoeBlock(
            Qwen4ExpTextConfig(
                hidden_size=HIDDEN,
                num_experts=8,
                num_experts_per_tok=3,
                moe_intermediate_size=INTERMEDIATE,
                shared_expert_intermediate_size=INTERMEDIATE,
                num_hidden_layers=1,
                layer_types=["linear_attention"],
            )
        ).float()
        target.load_state_dict(
            {
                name[len("mlp.") :]: tensor.float()
                for name, tensor in tensors.items()
                if name.startswith("mlp.")
            }
        )
        x = torch.randn(2, 5, HIDDEN)
        for layer in model.layers:
            assert_close(layer.mlp.eval()(x), target.eval()(x), rtol=1e-5, atol=1e-5)

    def test_unmapped_tensors_keep_their_fresh_initialization(self):
        model = draft()
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        written = set(apply_mtp_warm_start(model, fake_mtp_tensors(), layers=[1]))
        after = model.state_dict()
        for name, tensor in after.items():
            if name in written:
                continue
            self.assertTrue(torch.equal(tensor, before[name]), name)
        self.assertTrue(all(name.startswith("layers.1.") for name in written))
        self.assertTrue(torch.equal(after["fc.weight"], before["fc.weight"]))
        self.assertFalse(any("indexer" in name for name in written))

    def test_a_smaller_drafter_takes_the_listed_experts_in_order(self):
        tensors = fake_mtp_tensors(experts=8)
        model = draft(num_experts=4)
        ids = [6, 1, 3, 0]
        apply_mtp_warm_start(model, tensors, expert_ids=ids)
        layer = model.layers[0].mlp
        assert_close(layer.gate.weight, tensors["mlp.gate.weight"][ids].float())
        for local, source in enumerate(ids):
            gate_up = tensors["mlp.experts.gate_up_proj"][source].float()
            assert_close(layer.experts[local].gate_proj.weight, gate_up[:INTERMEDIATE])
            assert_close(layer.experts[local].up_proj.weight, gate_up[INTERMEDIATE:])
            assert_close(
                layer.experts[local].down_proj.weight,
                tensors["mlp.experts.down_proj"][source].float(),
            )

    def test_mismatches_are_refused(self):
        tensors = fake_mtp_tensors(experts=8)
        with self.assertRaisesRegex(ValueError, "pass expert_ids"):
            apply_mtp_warm_start(draft(num_experts=4), tensors)
        with self.assertRaisesRegex(ValueError, "distinct"):
            mtp_layer_state(tensors, expert_ids=[1, 1])
        quantized = dict(tensors)
        quantized["self_attn.q_proj.weight"] = torch.zeros(
            256, HIDDEN, dtype=torch.int8
        )
        with self.assertRaisesRegex(ValueError, "unquantized"):
            mtp_layer_state(quantized)
        with self.assertRaisesRegex(ValueError, "head|q_proj|shape"):
            apply_mtp_warm_start(draft(attn_output_gate=False), tensors)


# mtp.layers.0.* shapes in the released Qwen/Qwen3.8-Flash-Next safetensors
# headers (read 2026-10-09).
RELEASED_MTP_SHAPES = {
    "self_attn.q_proj.weight": (12288, 2560),
    "self_attn.k_proj.weight": (512, 2560),
    "self_attn.v_proj.weight": (512, 2560),
    "self_attn.o_proj.weight": (2560, 6144),
    "self_attn.q_norm.weight": (256,),
    "self_attn.k_norm.weight": (256,),
    "mlp.gate.weight": (512, 2560),
    "mlp.experts.gate_up_proj": (512, 1280, 2560),
    "mlp.experts.down_proj": (512, 2560, 640),
    "mlp.shared_expert.gate_proj.weight": (640, 2560),
    "mlp.shared_expert.up_proj.weight": (640, 2560),
    "mlp.shared_expert.down_proj.weight": (2560, 640),
    "mlp.shared_expert_gate.weight": (1, 2560),
}


class TestShippedConfigsMatchTheRelease(unittest.TestCase):
    def test_every_mapped_tensor_fits_the_shipped_drafters(self):
        from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
        from tests.test_modeling.test_qwen_moe_dspark import FLASH_NEXT_CONFIGS

        released = {
            name: torch.empty(shape, dtype=torch.bfloat16, device="meta")
            for name, shape in RELEASED_MTP_SHAPES.items()
        }
        for path, experts in zip(FLASH_NEXT_CONFIGS, (512, 64)):
            config = AutoDraftModelConfig.from_file(path)
            config._attn_implementation = "sdpa"
            with torch.device("meta"):
                model = AutoDraftModel.from_config(config)
            own = model.state_dict()
            ids = None if experts == 512 else list(range(0, 512, 8))
            state = mtp_layer_state(released, expert_ids=ids)
            for suffix, tensor in state.items():
                for index in range(len(model.layers)):
                    name = f"layers.{index}.{suffix}"
                    self.assertEqual(own[name].shape, tensor.shape, f"{path}: {name}")


class TestCheckpointRoundTrip(unittest.TestCase):
    def test_reader_opens_only_the_mtp_tensors(self):
        tensors = fake_mtp_tensors()
        with tempfile.TemporaryDirectory() as directory:
            write_checkpoint(directory, tensors)
            loaded = read_mtp_tensors(directory)
        self.assertEqual(set(loaded), set(tensors))

    def test_script_output_warm_starts_training(self):
        from scripts.warm_start_from_mtp import main
        from specforge.modeling.auto import AutoDraftModelConfig
        from specforge.training.model_loading import warm_start_draft_model

        tensors = fake_mtp_tensors()
        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "target")
            output = os.path.join(directory, "init")
            os.makedirs(target)
            write_checkpoint(target, tensors)
            config_path = os.path.join(directory, "draft.json")
            tiny_draft_config().to_json_file(config_path)
            main(
                [
                    "--target-model-path",
                    target,
                    "--draft-model-config",
                    config_path,
                    "--output-dir",
                    output,
                    "--dtype",
                    "float32",
                ]
            )
            with open(
                os.path.join(output, "mtp_warm_start.json"), encoding="utf-8"
            ) as stream:
                manifest = json.load(stream)
            self.assertEqual(manifest["layers"], [0, 1])

            config = AutoDraftModelConfig.from_file(config_path)
            config._attn_implementation = "sdpa"
            fresh = draft()
            report = warm_start_draft_model(
                fresh, output, draft_config=config, strategy="dspark"
            )
        self.assertEqual(report.missing_keys, ())
        assert_close(
            fresh.layers[1].self_attn.q_proj.weight,
            tensors["self_attn.q_proj.weight"].float(),
        )


if __name__ == "__main__":
    unittest.main()
