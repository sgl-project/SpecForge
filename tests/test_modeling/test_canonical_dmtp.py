"""DMTP fuses only first candidates and reuses ordinary context decoder execution."""

import copy
import unittest
from dataclasses import replace
from tempfile import TemporaryDirectory

import torch

from specforge.config.draft.architecture.block import ExternalFirstBlockSpec
from specforge.config.draft.codec import decode_config, encode_config
from specforge.modeling import build_draft_model
from specforge.modeling.draft.layers.attention.masks import AttentionLayout
from specforge.modeling.draft.layers.inputs import LayerInputs
from specforge.modeling.draft.models import DMTPDraftModel
from tests.test_config.test_draft_dmtp_spec import dmtp_config
from tests.test_modeling.test_draft_attention_masks import create_dflash_sdpa_mask


def model_config(*, bound=False, causal=False):
    base = dmtp_config()
    return replace(
        base,
        spec_config=replace(
            base.spec_config, block=ExternalFirstBlockSpec(block_size=4)
        ),
        attention_configs={
            "full": replace(base.attention_configs["full"], is_causal=causal)
        },
        **({} if bound else {"target_bindings": (), "target_logit_transform": None}),
    )


def model_inputs(config, device="cpu", dtype=torch.float32, *, keep_all=False):
    first_positions = torch.tensor([[2, 4], [3, 2]], device=device)
    keep = (
        torch.ones(2, 2, device=device, dtype=torch.bool)
        if keep_all
        else torch.tensor([[True, False], [False, True]], device=device)
    )
    positions = torch.cat(
        (
            torch.arange(5, device=device).expand(2, -1),
            (first_positions[..., None] + torch.arange(4, device=device)).flatten(1),
        ),
        dim=1,
    )
    mask = create_dflash_sdpa_mask(
        first_positions + config.spec_config.context_end_offset,
        keep,
        5,
        4,
        device,
        is_causal=config.attention_configs["full"].is_causal,
    )
    layout = AttentionLayout(
        query_positions=positions[:, 5:], key_positions=positions, visibility=mask
    )
    entries = [
        LayerInputs(attention_layout=layout, position_ids=positions)
        for _ in config.layer_types
    ]
    ids = torch.randint(64, (2, 8), device=device)
    context = torch.randn(2, 5, 64, device=device, dtype=dtype, requires_grad=True)
    mtp = torch.randn(2, 2, 48, device=device, dtype=dtype, requires_grad=True)
    return (ids, context, entries), mtp, keep


class CanonicalDMTPTest(unittest.TestCase):

    def test_inactive_nan_features_do_not_affect_gradients(self):
        for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            with self.subTest(device=device):
                config = model_config()
                actual = build_draft_model(config, device=device)
                reference = copy.deepcopy(actual)
                args, mtp, keep = model_inputs(config, device)
                padded = mtp.detach().masked_fill(~keep[..., None], float("nan"))
                padded.requires_grad_()
                output = actual(*args, mtp_features=padded, block_keep_mask=keep)
                expected = reference(*args, mtp_features=mtp, block_keep_mask=keep)
                torch.testing.assert_close(output.logits, expected.logits)
                output.logits.sum().backward()
                expected.logits.sum().backward()
                self.assertTrue(torch.isfinite(padded.grad).all())
                self.assertEqual(torch.count_nonzero(padded.grad[~keep]).item(), 0)
                for name, parameter in actual.named_parameters():
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    torch.testing.assert_close(
                        parameter.grad, dict(reference.named_parameters())[name].grad
                    )

    def test_target_width_can_differ_from_draft_width(self):
        base = model_config()
        config = replace(
            base,
            spec_config=replace(
                base.spec_config,
                target_features=replace(
                    base.spec_config.target_features, hidden_size=48
                ),
            ),
        )
        target = {"hidden_size": 48, "vocab_size": 64, "num_hidden_layers": 8}
        config = decode_config(encode_config(config), target_config=target)
        model = build_draft_model(config)
        (ids, _, entries), mtp, keep = model_inputs(config)
        context = torch.randn(2, 5, config.spec_config.target_features.input_width())
        output = model(ids, context, entries, mtp_features=mtp, block_keep_mask=keep)
        self.assertEqual(output.hidden_states.shape, (2, 8, 32))
        with self.assertRaisesRegex(ValueError, "target_features.hidden_size"):
            decode_config(
                encode_config(config), target_config=target | {"hidden_size": 32}
            )

    def test_only_first_slots_are_fused_and_blocks_are_independent(self):
        config = model_config()
        actual = build_draft_model(config)
        args, mtp, keep = model_inputs(config)
        captured = []
        hook = actual.backbone.register_forward_pre_hook(
            lambda _, values: captured.append(values[0].detach().clone())
        )
        try:
            actual(*args, mtp_features=mtp, block_keep_mask=keep, output_logits=False)
        finally:
            hook.remove()
        original = actual.embed_tokens(args[0]).reshape(2, 2, 4, 32)
        assembled = captured[0].reshape_as(original)
        torch.testing.assert_close(
            assembled[..., 1:, :], original[..., 1:, :], rtol=0, atol=0
        )
        torch.testing.assert_close(
            assembled[..., 0, :][~keep], original[..., 0, :][~keep], rtol=0, atol=0
        )
        fused = actual.mtp_fusion(original[..., 0, :], mtp)
        torch.testing.assert_close(
            assembled[..., 0, :][keep], fused[keep], rtol=0, atol=0
        )
        args, mtp, _ = model_inputs(config, keep_all=True)
        output = actual(*args, mtp_features=mtp)
        changed = mtp.detach().clone()
        changed[:, 0] += torch.randn_like(changed[:, 0])
        other = actual(*args, mtp_features=changed)
        torch.testing.assert_close(
            other.logits[:, 4:], output.logits[:, 4:], rtol=0, atol=0
        )
        self.assertFalse(torch.equal(other.logits[:, :4], output.logits[:, :4]))

    def test_checkpointing_binding_and_hf_restore(self):
        config = model_config(bound=True)
        actual = build_draft_model(config)
        weights = {name: torch.randn(64, 32) for name in config.target_bindings}
        for name, weight in weights.items():
            getattr(actual, name).load_target_weight(weight)
        checkpointed = copy.deepcopy(actual)
        checkpointed.gradient_checkpointing_enable()
        args, mtp, keep = model_inputs(config)
        # A frozen provider supplies detached features; fusion parameters still train.
        args = (args[0], args[1].detach(), args[2])
        mtp = mtp.detach()
        output = actual(*args, mtp_features=mtp, block_keep_mask=keep)
        expected = checkpointed(*args, mtp_features=mtp, block_keep_mask=keep)
        torch.testing.assert_close(output.logits, expected.logits, rtol=0, atol=0)
        output.logits.sum().backward()
        expected.logits.sum().backward()
        for name, parameter in actual.named_parameters():
            self.assertEqual(parameter.grad is not None, parameter.requires_grad, name)
            if parameter.requires_grad:
                torch.testing.assert_close(
                    parameter.grad,
                    dict(checkpointed.named_parameters())[name].grad,
                    rtol=0,
                    atol=0,
                )
        with TemporaryDirectory() as path:
            actual.save_pretrained(path)
            restored = DMTPDraftModel.from_pretrained(path)
        for name, weight in weights.items():
            module = getattr(restored, name)
            if module.weight.is_meta:
                module.to_empty(device="cpu")
            module.load_target_weight(weight)
        torch.testing.assert_close(
            restored(*args, mtp_features=mtp, block_keep_mask=keep).logits,
            output.logits,
            rtol=0,
            atol=0,
        )
        self.assertIn("mtp_fusion.fc.weight", actual.state_dict())
        self.assertNotIn("lm_head.weight", actual.state_dict())

    def test_input_contract_meta_and_hidden_only(self):
        config = model_config()
        actual = build_draft_model(config)
        args, mtp, keep = model_inputs(config)
        self.assertIsNone(actual(*args, mtp_features=mtp, output_logits=False).logits)
        with self.assertRaisesRegex(ValueError, "whole DMTP blocks"):
            actual(args[0][:, :-1], *args[1:], mtp_features=mtp)
        for features, mask in (
            (mtp[:, :1], keep),
            (mtp[..., :-1], keep),
            (mtp, keep.float()),
            (mtp, keep[:1]),
        ):
            with self.assertRaises(ValueError):
                actual(*args, mtp_features=features, block_keep_mask=mask)
        meta = build_draft_model(config, device="meta", dtype=torch.bfloat16)
        self.assertTrue(
            all(p.is_meta and p.dtype == torch.bfloat16 for p in meta.parameters())
        )
        tied = build_draft_model(replace(config, tie_word_embeddings=True))
        self.assertIs(tied.embed_tokens.weight, tied.lm_head.weight)


if __name__ == "__main__":
    unittest.main()
