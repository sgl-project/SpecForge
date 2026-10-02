import os
import unittest
from unittest import mock

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.modeling.draft.hspec import (
    DEFAULT_DFLASH_KERNELS,
    HSpecDecoderLayer,
    HSpecDraftModel,
    HSpecMamba2Reference,
    HSpecTargetKVAttention,
)


def _config(**overrides) -> Qwen3Config:
    method = {
        "mamba_num_heads": 4,
        "mamba_head_dim": 2,
        "mamba_state_size": 2,
        "mamba_groups": 2,
        "mamba_conv_kernel": 4,
        "num_hybrid_layers": 1,
        "num_partial_layers": 1,
        "target_kv_layer_ids": [17],
        "block_pattern": ["mamba", "attention", "mlp", "mlp"],
    }
    method.update(overrides)
    return Qwen3Config(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=4,
        head_dim=2,
        max_position_embeddings=32,
        vocab_size=32,
        block_size=3,
        num_target_layers=36,
        layer_types=["full_attention"] * 4,
        dflash_config={
            "projector_type": "dspark",
            "markov_rank": 4,
            "markov_head_type": "vanilla",
            "target_layer_ids": [1, 9, 17, 25, 33],
        },
        hspec_config=method,
    )


class HSpecReferenceTest(unittest.TestCase):
    def test_reference_recurrence_matches_stepwise_state(self):
        torch.manual_seed(3)
        module = HSpecMamba2Reference(_config())
        module.eval()
        hidden = torch.randn(2, 5, 8)
        state = torch.randn(2, module.num_heads, module.head_dim, module.state_size)

        with torch.no_grad():
            current = state.clone()
            projected = module.in_proj(hidden)
            gate, bc, dt = torch.split(
                projected,
                [module.intermediate_size, module.conv_dim, module.num_heads],
                dim=-1,
            )
            dt = torch.nn.functional.softplus(dt + module.dt_bias)
            convolved = module.conv1d(bc.transpose(1, 2))[
                ..., : hidden.shape[1]
            ].transpose(1, 2)
            convolved = torch.nn.functional.silu(convolved)
            value_states, state_b_states, _ = torch.split(
                convolved,
                [
                    module.intermediate_size,
                    module.n_groups * module.state_size,
                    module.n_groups * module.state_size,
                ],
                dim=-1,
            )
            decay = torch.exp(-dt[..., None] * torch.exp(module.A_log)[..., None])
            for step in range(hidden.shape[1]):
                current = current * decay[:, step][..., None]
                value = value_states[:, step].view(2, 4, 2)
                state_b = state_b_states[:, step].view(
                    2, module.n_groups, module.state_size
                )
                state_b = state_b.repeat_interleave(
                    module.num_heads // module.n_groups, dim=1
                )
                current = current + state_b.to(torch.float32).unsqueeze(2) * value.to(
                    torch.float32
                ).unsqueeze(-1) * dt[:, step].to(torch.float32).view(2, 4, 1, 1)

            actual, final_state = module(hidden, initial_state=state)

        self.assertEqual(module.state_shape, (4, 2, 2))
        self.assertEqual(actual.shape, hidden.shape)
        self.assertEqual(final_state.shape, state.shape)
        torch.testing.assert_close(final_state, current, rtol=1e-5, atol=1e-6)


class HSpecTargetKVAttentionTest(unittest.TestCase):
    def _attention(self, implementation):
        config = _config()
        config._attn_implementation = implementation
        return HSpecTargetKVAttention(config, 0, DEFAULT_DFLASH_KERNELS)

    def test_eager_and_sdpa_target_kv_attention_agree(self):
        torch.manual_seed(7)
        eager = self._attention("eager")
        sdpa = self._attention("sdpa")
        sdpa.load_state_dict(eager.state_dict())
        eager.eval()
        sdpa.eval()

        hidden = torch.randn(2, 3, 8)
        target_key = torch.randn(2, 4, 4)
        target_value = torch.randn(2, 4, 4)
        prefix_mask = torch.tensor(
            [[True, True, True, False], [True, False, False, False]]
        )
        position_embeddings = (
            hidden.new_ones(1, 4, 2),
            hidden.new_zeros(1, 4, 2),
        )

        with torch.no_grad():
            eager_output = eager(
                hidden,
                target_key=target_key,
                target_value=target_value,
                position_embeddings=position_embeddings,
                prefix_mask=prefix_mask,
            )
            sdpa_output = sdpa(
                hidden,
                target_key=target_key,
                target_value=target_value,
                position_embeddings=position_embeddings,
                prefix_mask=prefix_mask,
            )

        self.assertEqual(eager_output.shape, (2, 3, 8))
        torch.testing.assert_close(sdpa_output, eager_output, rtol=1e-4, atol=1e-5)

    def test_prefix_mask_excludes_disabled_context(self):
        config = _config()
        config._attn_implementation = "eager"
        attention = HSpecTargetKVAttention(config, 0, DEFAULT_DFLASH_KERNELS)
        attention.eval()
        hidden = torch.zeros(1, 1, 8)
        target_key = torch.randn(1, 2, 4, 2)
        target_value = torch.randn(1, 2, 4, 2)
        position_embeddings = (torch.ones(1, 1, 2), torch.zeros(1, 1, 2))
        with torch.no_grad():
            allowed = attention(
                hidden,
                target_key=target_key,
                target_value=target_value,
                position_embeddings=position_embeddings,
                prefix_mask=torch.tensor([[True, True, True, False]]),
            )
            blocked = attention(
                hidden,
                target_key=target_key,
                target_value=target_value,
                position_embeddings=position_embeddings,
                prefix_mask=torch.tensor([[False, False, False, False]]),
            )
        self.assertFalse(torch.allclose(allowed, blocked))


class HSpecDraftModelTest(unittest.TestCase):
    def test_layer_mapping_uses_configured_ids(self):
        model = HSpecDraftModel(_config())
        self.assertEqual(model.target_layer_ids, [1, 9, 17, 25, 33])
        self.assertEqual(
            [layer.mamba.state_shape for layer in model.hybrid_layers],
            [(4, 2, 2)] * 1,
        )

    def test_forward_requires_selected_target_kv(self):
        model = HSpecDraftModel(_config())
        model.eval()
        position_ids = torch.arange(3).unsqueeze(0)
        noise_embedding = torch.randn(1, 3, 8)
        target_hidden = torch.randn(1, 3, 40)
        target_last_hidden = torch.randn(1, 3, 8)
        with self.assertRaisesRegex(ValueError, "selected target K and V"):
            model(
                position_ids=position_ids,
                noise_embedding=noise_embedding,
                target_hidden=target_hidden,
                target_last_hidden_states=target_last_hidden,
            )


class HSpecSplitSourceAttentionTest(unittest.TestCase):
    def _attention(self, implementation="eager"):
        config = _config()
        config._attn_implementation = implementation
        attention = HSpecTargetKVAttention(config, 0, DEFAULT_DFLASH_KERNELS)
        attention.eval()
        return attention

    @staticmethod
    def _inputs():
        generator = torch.Generator().manual_seed(0)
        hidden = torch.randn(2, 3, 8, generator=generator)
        position_embeddings = (
            hidden.new_ones(1, 4, 2),
            hidden.new_zeros(1, 4, 2),
        )
        return hidden, position_embeddings

    @staticmethod
    def _run(attention, target_key, target_value, prefix_mask, merge_mode=None):
        hidden, position_embeddings = HSpecSplitSourceAttentionTest._inputs()
        env = {"SPECFORGE_HSPEC_ATTENTION_MERGE": merge_mode} if merge_mode else {}
        with mock.patch.dict(os.environ, env, clear=False):
            with torch.no_grad():
                return attention(
                    hidden,
                    target_key=target_key,
                    target_value=target_value,
                    position_embeddings=position_embeddings,
                    prefix_mask=prefix_mask,
                )

    def test_split_matches_fused_cat(self):
        torch.manual_seed(11)
        for implementation in ("eager", "sdpa"):
            with self.subTest(implementation=implementation):
                attention = self._attention(implementation)
                target_key = torch.randn(2, 4, 4)
                target_value = torch.randn(2, 4, 4)
                prefix_mask = torch.tensor(
                    [[True, True, True, False], [True, False, False, False]]
                )
                cat_output = self._run(
                    attention, target_key, target_value, prefix_mask, "cat"
                )
                split_output = self._run(
                    attention, target_key, target_value, prefix_mask, "split"
                )
                torch.testing.assert_close(
                    split_output, cat_output, rtol=1e-5, atol=1e-5
                )

    def test_split_accepts_all_target_kv_layouts(self):
        torch.manual_seed(12)
        attention = self._attention("eager")
        dense = torch.randn(2, 4, 2, 2)
        reference = self._run(attention, dense, dense, None, "split")
        for layout, expected in (
            (dense.transpose(1, 2).contiguous(), reference),
            (dense.reshape(2, 4, 4), reference),
        ):
            with self.subTest(layout=tuple(layout.shape)):
                output = self._run(attention, layout, layout, None, "split")
                torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
        shared = dense[0].reshape(4, 4)
        broadcast = dense[0].unsqueeze(0).expand(2, -1, -1, -1)
        shared_expected = self._run(
            attention, broadcast.contiguous(), broadcast.contiguous(), None, "split"
        )
        output = self._run(attention, shared, shared, None, "split")
        torch.testing.assert_close(output, shared_expected, rtol=1e-5, atol=1e-5)

    def test_empty_prefix_equals_block_only(self):
        torch.manual_seed(13)
        attention = self._attention("eager")
        empty_key = torch.randn(2, 0, 4)
        empty_value = torch.randn(2, 0, 4)
        empty_prefix = self._run(attention, empty_key, empty_value, None, "split")
        draft_only = self._run(attention, empty_key, empty_value, None, "cat")
        torch.testing.assert_close(empty_prefix, draft_only, rtol=1e-5, atol=1e-5)

    def test_fully_masked_prefix_equals_empty_prefix(self):
        torch.manual_seed(14)
        attention = self._attention("eager")
        target_key = torch.randn(2, 4, 4)
        target_value = torch.randn(2, 4, 4)
        blocked = self._run(
            attention,
            target_key,
            target_value,
            torch.zeros(2, 4, dtype=torch.bool),
            "split",
        )
        empty_key = torch.randn(2, 0, 4)
        empty_value = torch.randn(2, 0, 4)
        empty_prefix = self._run(attention, empty_key, empty_value, None, "split")
        torch.testing.assert_close(blocked, empty_prefix, rtol=1e-5, atol=1e-5)

    @unittest.skipIf(
        not (hasattr(torch, "npu") and torch.npu.is_available()),
        "requires an Ascend NPU",
    )
    def test_npu_split_source_matches_cpu(self):
        torch.manual_seed(15)
        cpu_attention = self._attention("eager")
        npu_attention = self._attention("eager")
        npu_attention.load_state_dict(cpu_attention.state_dict())
        npu_attention = npu_attention.to("npu")

        hidden = torch.randn(2, 3, 8)
        target_key = torch.randn(2, 6, 4)
        target_value = torch.randn(2, 6, 4)
        prefix_mask = torch.tensor(
            [[True] * 6, [True, True, False, False, False, False]]
        )
        position_embeddings = (
            hidden.new_ones(1, 4, 2),
            hidden.new_zeros(1, 4, 2),
        )
        with torch.no_grad():
            cpu_output = cpu_attention(
                hidden,
                target_key=target_key,
                target_value=target_value,
                position_embeddings=position_embeddings,
                prefix_mask=prefix_mask,
            )
            npu_output = npu_attention(
                hidden.to("npu"),
                target_key=target_key.to("npu"),
                target_value=target_value.to("npu"),
                position_embeddings=(
                    position_embeddings[0].to("npu"),
                    position_embeddings[1].to("npu"),
                ),
                prefix_mask=prefix_mask.to("npu"),
            ).cpu()
        torch.testing.assert_close(npu_output, cpu_output, rtol=1e-4, atol=1e-5)

    @unittest.skipIf(
        not (hasattr(torch, "npu") and torch.npu.is_available()),
        "requires an Ascend NPU",
    )
    def test_npu_hybrid_layer_forward_backward_split_source(self):
        torch.manual_seed(16)
        config = _config()
        config._attn_implementation = "eager"
        layer = HSpecDecoderLayer(config, 0, DEFAULT_DFLASH_KERNELS)
        layer = layer.to("npu")
        layer.train()

        hidden = torch.randn(2, 3, 8, device="npu", requires_grad=True)
        target_key = torch.randn(2, 5, 4, device="npu")
        target_value = torch.randn(2, 5, 4, device="npu")
        prefix_mask = torch.ones(2, 5, dtype=torch.bool, device="npu")
        position_embeddings = (
            torch.ones(1, 3, 2, device="npu"),
            torch.zeros(1, 3, 2, device="npu"),
        )
        output, state = layer(
            hidden,
            target_key=target_key,
            target_value=target_value,
            initial_state=None,
            prefix_mask=prefix_mask,
            position_embeddings=position_embeddings,
        )
        self.assertEqual(output.shape, (2, 3, 8))
        self.assertTrue(torch.isfinite(output).all())
        output.float().sum().backward()
        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.isfinite(hidden.grad).all())
        for name, parameter in layer.named_parameters():
            self.assertIsNotNone(parameter.grad, f"no gradient for {name}")


class HSpecDraftMaskSemanticsTest(unittest.TestCase):
    """End-to-end checks that the DFlash context/block masks reach attention."""

    def test_split_and_cat_agree_with_block_mask(self):
        torch.manual_seed(17)
        eager_config = _config()
        eager_config._attn_implementation = "eager"
        attention = HSpecTargetKVAttention(eager_config, 0, DEFAULT_DFLASH_KERNELS)
        attention.eval()

        hidden = torch.randn(1, 6, 8)
        target_key = torch.randn(1, 5, 4)
        target_value = torch.randn(1, 5, 4)
        # Two blocks of 3: queries only see their own block's draft keys
        # (bidirectional) and target tokens before their anchor.
        block_mask = torch.zeros(1, 6, 6, dtype=torch.bool)
        for query_index in range(6):
            for key_index in range(6):
                block_mask[:, query_index, key_index] = (
                    query_index // 3 == key_index // 3
                )
        prefix_mask = torch.zeros(1, 6, 5, dtype=torch.bool)
        prefix_mask[:, 0, :2] = True
        prefix_mask[:, 3, :4] = True
        position_embeddings = (
            hidden.new_ones(1, 6, 2),
            hidden.new_zeros(1, 6, 2),
        )
        outputs = {}
        for merge_mode in ("split", "cat"):
            with mock.patch.dict(
                os.environ,
                {"SPECFORGE_HSPEC_ATTENTION_MERGE": merge_mode},
                clear=False,
            ):
                with torch.no_grad():
                    outputs[merge_mode] = attention(
                        hidden,
                        target_key=target_key,
                        target_value=target_value,
                        position_embeddings=position_embeddings,
                        prefix_mask=prefix_mask,
                        block_mask=block_mask,
                    )
        torch.testing.assert_close(
            outputs["split"], outputs["cat"], rtol=1e-5, atol=1e-5
        )

    def test_forward_consumes_dflash_mask_and_stays_finite(self):
        torch.manual_seed(18)
        model = HSpecDraftModel(_config())
        model.eval()
        seq_len, block_size = 5, 3
        position_ids = torch.arange(seq_len + block_size).unsqueeze(0)
        noise_embedding = torch.randn(1, block_size, 8)
        target_hidden = torch.randn(1, seq_len, 40)
        target_last_hidden = torch.randn(1, seq_len, 8)
        anchor_positions = torch.tensor([[2]])
        target_keys = torch.randn(1, seq_len, 4)
        target_values = torch.randn(1, seq_len, 4)

        def forward_with(context_visible: bool):
            mask = torch.ones(1, block_size, seq_len + block_size, dtype=torch.bool)
            if not context_visible:
                # The anchor sits at position 2: target tokens >= 2 must be
                # invisible, otherwise the draft attends to the answer's own
                # future K/V (the leak the serving path cannot have).
                mask[..., :seq_len] = torch.tensor([[True, True, False, False, False]])
            with torch.no_grad():
                return model(
                    position_ids=position_ids,
                    attention_mask=mask,
                    noise_embedding=noise_embedding,
                    target_hidden=target_hidden,
                    target_keys=target_keys,
                    target_values=target_values,
                    prefix_masks=torch.ones(1, seq_len, dtype=torch.bool),
                    target_last_hidden_states=target_last_hidden,
                    anchor_positions=anchor_positions,
                )

        blocked = forward_with(context_visible=False)
        leaked = forward_with(context_visible=True)
        self.assertTrue(torch.isfinite(blocked).all())
        self.assertFalse(torch.allclose(blocked, leaked))

    def test_forward_requires_anchor_positions(self):
        model = HSpecDraftModel(_config())
        model.eval()
        with self.assertRaisesRegex(ValueError, "anchor_positions"):
            model(
                position_ids=torch.arange(8).unsqueeze(0),
                noise_embedding=torch.randn(1, 3, 8),
                target_hidden=torch.randn(1, 5, 40),
                target_keys=torch.randn(1, 5, 4),
                target_values=torch.randn(1, 5, 4),
                target_last_hidden_states=torch.randn(1, 5, 8),
                anchor_positions=None,
            )


class HSpecOnlineModelIntegrationTest(unittest.TestCase):
    """Drive the real _forward_draft_blocks -> draft forward -> loss chain.

    Unit tests feed 3-D masks directly into the attention; this test goes
    through the production path where create_dflash_sdpa_mask emits the
    4-D [batch, 1, query, prefix + query] mask the draft must consume.
    """

    def test_online_forward_backward_with_production_mask(self):
        from specforge.algorithms.common.dflash_family_model import OnlineHSpecModel

        torch.manual_seed(19)
        draft = HSpecDraftModel(_config())
        model = OnlineHSpecModel(
            draft_model=draft,
            target_lm_head=nn.Linear(8, 32, bias=False),
            target_embed_tokens=nn.Embedding(32, 8),
            mask_token_id=1,
            block_size=3,
            attention_backend="sdpa",
            num_anchors=4,
            dspark_confidence_head_alpha=1.0,
        )
        model.train()

        batch_size, seq_len = 2, 5
        generator = torch.Generator().manual_seed(1)
        input_ids = torch.randint(2, 32, (batch_size, seq_len), generator=generator)
        loss_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
        loss_mask[:, -2:] = 1
        prefix_masks = torch.ones(batch_size, seq_len, dtype=torch.long)
        prefix_masks[1, -2:] = 0

        loss, accuracy, metrics = model(
            input_ids=input_ids,
            hidden_states=torch.randn(batch_size, seq_len, 40, generator=generator),
            loss_mask=loss_mask,
            target_last_hidden_states=torch.randn(
                batch_size, seq_len, 8, generator=generator
            ),
            selected_target_k=torch.randn(batch_size, seq_len, 4, generator=generator),
            selected_target_v=torch.randn(batch_size, seq_len, 4, generator=generator),
            prefix_masks=prefix_masks,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        seed_grads = [
            name
            for name, parameter in model.named_parameters()
            if "seed_projs" in name and parameter.grad is not None
        ]
        self.assertTrue(seed_grads, "seed_projs received no gradient")
        self.assertTrue(
            bool(((accuracy >= 0) & (accuracy <= 1)).all()),
            f"accuracy out of range: {accuracy}",
        )


if __name__ == "__main__":
    unittest.main()
