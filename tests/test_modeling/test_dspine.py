"""DSpine causal isolation, refinement parity, and persistent transfer state."""

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import Qwen3Config

from scripts.convert_dspine_checkpoint import convert_checkpoint
from specforge.algorithms.common.dflash_family_model import (
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.dspine import DSpineConfig, DSpineDraftModel


def tiny_config(head_dim=4, **options):
    return Qwen3Config(
        architectures=["DSpineDraftModel"],
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=16 // head_dim,
        num_key_value_heads=min(2, 16 // head_dim),
        num_hidden_layers=3,
        num_target_layers=8,
        head_dim=head_dim,
        vocab_size=32,
        rms_norm_eps=1e-6,
        block_size=4,
        layer_types=["full_attention"] * 3,
        dflash_config={"mask_token_id": 31, "target_layer_ids": [1, 4, 6]},
        dspine_config={"transfer_rank": 8, "message_size": 4, "top_k": 3, **options},
        attn_implementation="sdpa",
    )


def initialized_model(**options):
    torch.manual_seed(17)
    model = DSpineDraftModel(tiny_config(**options))
    head = torch.randn(32, 16)
    model.initialize_transfer_space(head, chunk_size=7)
    return model, head


def forward_inputs(model):
    anchors = torch.tensor([[2, 5], [1, 4]])
    length = 7
    token_ids = torch.randint(0, 30, (2, 2, 4))
    positions = torch.cat(
        (
            torch.arange(length).expand(2, -1),
            (anchors[..., None] + torch.arange(4)).flatten(1),
        ),
        dim=1,
    )
    return dict(
        position_ids=positions,
        attention_mask=create_dflash_sdpa_mask(
            anchors,
            torch.ones_like(anchors, dtype=torch.bool),
            length,
            4,
            "cpu",
            causal_block=True,
        ),
        noise_embedding=torch.randn(2, 8, 16),
        target_hidden=torch.randn(2, length, 48),
        block_token_ids=token_ids,
    )


class DSpineTest(unittest.TestCase):
    def test_rejects_bidirectional_attention(self):
        config = tiny_config()
        config.is_causal = False
        with self.assertRaisesRegex(ValueError, "DSpine requires causal"):
            DSpineDraftModel(config)

    def test_zero_writes_are_identity_against_the_shared_causal_backbone(self):
        (model, _) = initialized_model()
        inputs = forward_inputs(model)
        backbone = DFlashDraftModel(copy.deepcopy(model.config))
        backbone.load_state_dict(
            {
                key: value
                for (key, value) in model.state_dict().items()
                if key in backbone.state_dict()
            },
            strict=True,
        )
        expected = backbone(
            **{
                key: value
                for (key, value) in inputs.items()
                if key != "block_token_ids"
            }
        )
        actual = model(**inputs)
        torch.testing.assert_close(
            actual.hidden_states.flatten(1, 2), expected, atol=0, rtol=0
        )
        torch.testing.assert_close(
            model.injection.predecessor_gate.weight,
            torch.zeros_like(model.injection.predecessor_gate.weight),
        )

    def test_future_slots_and_other_blocks_cannot_change_a_prefix(self):
        (model, _) = initialized_model()
        with torch.no_grad():
            for write in model.injection.writes:
                write.weight.normal_(std=0.1)
        inputs = forward_inputs(model)
        expected = model(**inputs).hidden_states
        changed = {**inputs, "noise_embedding": inputs["noise_embedding"].clone()}
        changed["noise_embedding"][:, 2:] += 10 * torch.randn_like(
            changed["noise_embedding"][:, 2:]
        )
        actual = model(**changed).hidden_states
        torch.testing.assert_close(actual[:, 0, :2], expected[:, 0, :2], atol=0, rtol=0)
        changed = {**inputs, "block_token_ids": inputs["block_token_ids"].clone()}
        changed["block_token_ids"][..., 1:] = (
            changed["block_token_ids"][..., 1:] + 3
        ) % 30
        torch.testing.assert_close(
            model(**changed).hidden_states, expected, atol=0, rtol=0
        )

    def test_injection_preserves_anchors_and_only_replaces_shallow_predecessors(self):
        (model, _) = initialized_model(replacement_layers=1)
        with torch.no_grad():
            for write in model.injection.writes:
                write.weight.normal_(std=0.1)
        hidden = torch.randn(2, 2, 4, 16)
        labels = torch.randint(0, 30, (2, 2, 4))
        replace = torch.tensor([[True, False], [False, True]])
        for layer in range(3):
            normal = model.injection(hidden, layer, labels, model.transfer_codes)
            replaced = model.injection(
                hidden, layer, labels, model.transfer_codes, replace
            )
            torch.testing.assert_close(
                replaced[..., 0, :], hidden[..., 0, :], atol=0, rtol=0
            )
            torch.testing.assert_close(
                replaced[~replace], normal[~replace], atol=0, rtol=0
            )
            if layer:
                torch.testing.assert_close(replaced, normal, atol=0, rtol=0)
            else:
                self.assertFalse(
                    torch.equal(replaced[replace, 2:], normal[replace, 2:])
                )

    def test_transition_cache_matches_sequential_last_write_selection(self):
        (model, head) = initialized_model()
        with torch.no_grad():
            model.injection.writes[-1].weight.normal_(std=0.1)
        hidden = torch.randn(2, 2, 3, 16)
        candidates = torch.randn(2, 2, 3, 32).topk(3).indices
        anchor = torch.randint(0, 32, (2, 2))
        scores = model.transition_scores(hidden, candidates, anchor, head)
        cached = model.select_cached(scores, candidates)
        predecessor = anchor
        selected = []
        for position in range(3):
            refined = model.refine(hidden[..., position, :], predecessor)
            direct = torch.einsum(
                "...h,...kh->...k",
                refined,
                F.embedding(candidates[..., position, :], head),
            )
            predecessor = (
                candidates[..., position, :]
                .gather(-1, direct.argmax(-1)[..., None])
                .squeeze(-1)
            )
            selected.append(predecessor)
        torch.testing.assert_close(
            cached, torch.stack(selected, dim=-1), atol=0, rtol=0
        )
        for position in range(3):
            for index in range(3):
                predecessor = (
                    anchor if position == 0 else candidates[..., position - 1, index]
                )
                refined = model.refine(hidden[..., position, :], predecessor)
                direct = torch.einsum(
                    "...h,...kh->...k",
                    refined,
                    F.embedding(candidates[..., position, :], head),
                )
                torch.testing.assert_close(scores[..., position, index, :], direct)

    def test_zero_state_refinement_has_finite_gradients(self):
        (model, _) = initialized_model()
        hidden = torch.zeros(2, 16, requires_grad=True)
        model.refine(hidden, torch.tensor([1, 2])).sum().backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())

    def test_flex_and_dense_causal_masks_agree(self):
        for window in [None, 3]:
            with self.subTest(window=window):
                anchors = torch.tensor([[2, 5], [1, 4]])
                keep = torch.tensor([[True, True], [True, False]])
                arguments = dict(
                    anchor_positions=anchors,
                    block_keep_mask=keep,
                    S=7,
                    block_size=4,
                    device="cpu",
                    sliding_window=window,
                    causal_block=True,
                )
                dense = create_dflash_sdpa_mask(**arguments)
                block_mask = create_dflash_block_mask(**arguments)
                for batch in range(2):
                    for query in range(8):
                        keys = torch.arange(15)
                        actual = block_mask.mask_mod(
                            torch.tensor(batch),
                            torch.tensor(0),
                            torch.tensor(query),
                            keys,
                        )
                        torch.testing.assert_close(actual, dense[batch, 0, query])
                self.assertFalse(dense[1, 0, 4:].any())
                self.assertFalse(dense[0, 0, 0, 8:11].any())

    def test_fixed_whitened_codes_and_learned_readouts_survive_hf_reload(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            (model, _) = initialized_model()
            self.assertFalse(model.transfer_codes.requires_grad)
            torch.testing.assert_close(
                model.transfer_codes.norm(dim=-1), torch.ones(32)
            )
            with torch.no_grad():
                model.injection.readouts[0].weight.add_(0.2)
            inputs = forward_inputs(model)
            expected = model(**inputs).hidden_states
            model.save_pretrained(tmp_path)
            reloaded = DSpineDraftModel.from_pretrained(
                tmp_path, attn_implementation="sdpa"
            )
            self.assertTrue(reloaded.transfer_ready.item())
            torch.testing.assert_close(
                reloaded(**inputs).hidden_states, expected, atol=0, rtol=0
            )
            torch.testing.assert_close(
                reloaded.transfer_codes, model.transfer_codes, atol=0, rtol=0
            )

    def test_converter_retains_reference_weights_and_codes_without_reinitializing(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            (model, _) = initialized_model()
            state = {
                name: tensor
                for (name, tensor) in model.state_dict().items()
                if name not in {"transfer_codes", "transfer_ready"}
            }
            save_file(state, tmp_path / "model.safetensors")
            save_file(
                {"codes": model.transfer_codes}, tmp_path / "transfer.safetensors"
            )
            (tmp_path / "config.json").write_text(
                json.dumps({"config": model.config.to_dict()})
            )
            output = tmp_path / "export"
            convert_checkpoint(
                tmp_path / "model.safetensors",
                tmp_path / "config.json",
                tmp_path / "transfer.safetensors",
                output,
            )
            reloaded = DSpineDraftModel.from_pretrained(output)
            for name, expected in model.state_dict().items():
                torch.testing.assert_close(
                    reloaded.state_dict()[name], expected, atol=0, rtol=0
                )
            self.assertEqual(reloaded.block_size, 16)
            with self.assertRaises(FileExistsError):
                convert_checkpoint(
                    tmp_path / "model.safetensors",
                    tmp_path / "config.json",
                    tmp_path / "transfer.safetensors",
                    output,
                )

    def test_registry_resolves_dspine_and_uninitialized_transfer_space_is_rejected(
        self,
    ):
        model = AutoDraftModel.from_config(tiny_config())
        self.assertTrue(isinstance(model, DSpineDraftModel))
        with self.assertRaisesRegex(RuntimeError, "transfer space"):
            model(**forward_inputs(model))

    def test_invalid_options_fail_before_training(self):
        for options in [
            {"transfer_rank": 17},
            {"top_k": 0},
            {"message_size": 0},
            {"alignment_weight": float("nan")},
            {"replacement_hold_ratio": 0.5},
            {"top_k": 2.5},
        ]:
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, "DSpine"):
                    DSpineDraftModel(tiny_config(**options))

    def test_schedules_reach_the_paper_endpoints(self):
        options = DSpineConfig()
        self.assertEqual(options.schedule(0, 600), (0, 0.5))
        self.assertEqual(options.schedule(100, 600)[1], 0.5)
        self.assertLessEqual(
            abs(options.schedule(150, 600)[1] - 0.25), max(1e-12, 1e-06 * abs(0.25))
        )
        self.assertEqual(options.schedule(200, 600)[1], 0)
        self.assertEqual(options.schedule(146, 600)[0], 0.5)
        self.assertEqual(options.schedule(0, None)[1], 0)


if __name__ == "__main__":
    unittest.main()
