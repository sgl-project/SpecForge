# coding=utf-8
"""MLA (DeepSeek) DFlash draft gates: the draft-ARCHITECTURE axis for DFlash.

The DFlash algorithm surface is untouched — same OnlineDFlashModel wrapper,
same block-parallel context+noise attention contract, same fc/norm/capture
surface — only the draft attention is MLA (compressed KV via kv_a/kv_b, split
nope/rope head dims, interleaved-pair RoPE, YaRN-aware softmax scale). Gates:

- registry: DeepseekDFlashDraftModel resolves through the draft registry and
  builds from a deepseek_v3 AutoConfig;
- attention shapes/grads: the MLA context+noise attention returns the noise-
  length output with v_head_dim geometry and is differentiable, for both the
  q_lora and no-q_lora projection branches;
- train smoke: a few optimizer steps through the UNCHANGED OnlineDFlashModel
  over synthetic captured features — finite loss, accuracy in [0, 1],
  trainable draft grads.

GPU-only. Run on the H200 box via rcli.
"""

import os
import tempfile
import unittest

import torch

CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, "MLA DFlash gates require CUDA")
class TestMLADFlash(unittest.TestCase):
    def test_registry_resolves_mla_dflash(self):
        from transformers import AutoConfig

        from specforge.modeling.draft.deepseek_dflash import DeepseekDFlashDraftModel
        from specforge.modeling.draft.registry import DRAFT_REGISTRY
        from tests.test_runtime import _fixtures as fx

        self.assertIn("DeepseekDFlashDraftModel", DRAFT_REGISTRY)
        self.assertIs(DRAFT_REGISTRY["DeepseekDFlashDraftModel"], DeepseekDFlashDraftModel)

        workdir = tempfile.mkdtemp(prefix="mla_dflash_cfg_")
        cfg_path = fx.write_mla_dflash_config(os.path.join(workdir, "cfg.json"))
        cfg = AutoConfig.from_pretrained(cfg_path)
        model = DeepseekDFlashDraftModel(cfg)
        # the capture schedule is derived from num_target_layers/num_hidden_layers
        self.assertEqual(len(model.target_layer_ids), cfg.num_hidden_layers)
        self.assertEqual(
            model.fc.in_features, len(model.target_layer_ids) * cfg.hidden_size
        )

    def test_mla_attention_shapes_and_grads(self):
        from transformers import AutoConfig

        from specforge.modeling.draft.deepseek_dflash import DeepseekDFlashAttention
        from tests.test_runtime import _fixtures as fx

        for q_lora_rank in (None, 24):
            with self.subTest(q_lora_rank=q_lora_rank):
                torch.manual_seed(0)
                workdir = tempfile.mkdtemp(prefix="mla_dflash_attn_")
                cfg_path = fx.write_mla_dflash_config(
                    os.path.join(workdir, "cfg.json"), q_lora_rank=q_lora_rank
                )
                cfg = AutoConfig.from_pretrained(cfg_path)
                attn = DeepseekDFlashAttention(cfg).cuda().to(torch.float32)

                bsz, ctx_len, q_len, h = 2, 20, 8, cfg.hidden_size
                noise = torch.randn(bsz, q_len, h, device="cuda", requires_grad=True)
                context = torch.randn(bsz, ctx_len, h, device="cuda")
                # position_ids: [context positions | draft positions], as
                # OnlineDFlashModel assembles them (len == ctx_len + q_len).
                ctx_pos = torch.arange(ctx_len, device="cuda")
                draft_pos = torch.arange(q_len, device="cuda") + 5
                position_ids = torch.cat([ctx_pos, draft_pos])[None].expand(bsz, -1)
                # full boolean mask [B, 1, q_len, ctx_len + q_len]
                mask = torch.ones(
                    bsz, 1, q_len, ctx_len + q_len, dtype=torch.bool, device="cuda"
                )

                out = attn(
                    hidden_states=noise,
                    target_hidden=context,
                    position_ids=position_ids,
                    attention_mask=mask,
                )
                self.assertEqual(tuple(out.shape), (bsz, q_len, h))
                self.assertTrue(torch.isfinite(out).all())
                out.sum().backward()
                self.assertIsNotNone(noise.grad)
                self.assertTrue(torch.isfinite(noise.grad).all())

    def test_train_smoke_through_online_dflash(self):
        from tests.test_runtime import _fixtures as fx

        torch.manual_seed(0)
        BS, SEQ, BLOCK = 2, 32, 4
        workdir = tempfile.mkdtemp(prefix="mla_dflash_smoke_")
        dflash_model, width, target_layer_ids = fx.build_dflash_mla(
            workdir, block_size=BLOCK, num_anchors=8
        )
        self.assertEqual(width, len(target_layer_ids) * dflash_model.draft_model.config.hidden_size)

        opt = torch.optim.AdamW(dflash_model.draft_model.parameters(), lr=1e-3)

        losses, accs = [], []
        for _ in range(3):
            input_ids = torch.randint(0, 256, (BS, SEQ), device="cuda")
            loss_mask = torch.ones(BS, SEQ, dtype=torch.long, device="cuda")
            hidden_states = torch.randn(
                BS, SEQ, width, device="cuda", dtype=torch.bfloat16
            )
            loss, acc, metrics = dflash_model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            accs.append(acc.item())

        self.assertTrue(all(torch.isfinite(torch.tensor(x)) for x in losses))
        self.assertTrue(all(0.0 <= a <= 1.0 for a in accs))
        self.assertIn("accuracy_denom", metrics)
        # at least one draft parameter received a gradient
        grads = [
            p.grad for p in dflash_model.draft_model.parameters() if p.grad is not None
        ]
        self.assertTrue(grads)


if __name__ == "__main__":
    unittest.main(verbosity=2)
