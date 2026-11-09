"""Reference math, state flow and training integration for hyper-connections."""

import copy
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.testing import assert_close

from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
from specforge.modeling.draft.dflash import (
    DFlashDraftModel,
    dflash_residual_resume_contract,
)
from specforge.modeling.draft.dflash2 import DFlash2DraftModel
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.modeling.draft.layers.config import ResidualConfig
from specforge.modeling.draft.layers.residual.base import (
    HyperConnectionState,
    collapse_streams,
    expand_streams,
)
from specforge.modeling.draft.layers.residual.gr import GatedResidual
from specforge.modeling.draft.layers.residual.mhc import ManifoldHyperConnection
from tests.test_modeling.test_layers.test_composition import config


def gr_reference(module, streams, update=None):
    # Independently normalize/read each branch, then use the concatenation for
    # the shared gates, matching Qwen's per-branch Gemma normalization.
    branches = []
    for i in range(module.count):
        x = streams[..., i, :].float()
        x = x / (x.square().mean(-1, keepdim=True) + module.eps).sqrt()
        branches.append((x * (1 + module.norm_weight[i].float())).to(streams.dtype))
    flat = torch.cat(branches, dim=-1)
    gate = torch.sigmoid(
        F.linear(
            F.silu(F.linear(flat, module.input_mix_weight_down.weight) / module.count),
            module.input_mix_weight_up.weight,
        )
    )
    gates = gate.chunk(module.count, dim=-1)
    read = sum(branch * g for branch, g in zip(branches, gates)) / module.count
    if update is None:
        return read
    injection = 2 * torch.sigmoid(
        F.linear(flat, module.block_inject_weight.weight) / module.count
    )
    written = torch.stack(
        [
            streams[..., i, :] + injection[..., i, None] * update
            for i in range(module.count)
        ],
        dim=-2,
    )
    return read, written


def mhc_reference(module, streams, incoming, update, norm):
    n = module.count
    flat = streams.flatten(-2).float()
    projected = flat @ module.mix_weight.float().T
    projected = (
        projected / (flat.square().mean(-1, keepdim=True) + module.norm_eps).sqrt()
    )
    scale = torch.cat(
        [
            module.mix_scale[0].expand(n),
            module.mix_scale[1].expand(n),
            module.mix_scale[2].expand(n * n),
        ]
    )
    coeff = projected * scale + module.mix_bias
    pre = coeff[..., :n].sigmoid() + module.eps
    post = 2 * coeff[..., n : 2 * n].sigmoid()
    matrix = (
        coeff[..., 2 * n :].reshape(*streams.shape[:-2], n, n).softmax(-1) + module.eps
    )
    matrix = matrix / (matrix.sum(-2, keepdim=True) + module.eps)
    for _ in range(module.sinkhorn_iters - 1):
        matrix = matrix / (matrix.sum(-1, keepdim=True) + module.eps)
        matrix = matrix / (matrix.sum(-2, keepdim=True) + module.eps)
    read = norm(
        sum(streams[..., i, :].float() * incoming[..., i, None] for i in range(n)).to(
            streams.dtype
        )
    )
    written = torch.stack(
        [
            sum(matrix[..., j, i, None] * streams[..., j, :].float() for j in range(n))
            + post[..., i, None] * update.float()
            for i in range(n)
        ],
        dim=-2,
    ).to(update.dtype)
    return read, written, pre


class HyperConnectionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_gr_forward_and_all_gradients_match_reference(self):
        module = GatedResidual(8, 3, 5, 1e-6)
        streams = torch.randn(2, 4, 3, 8, requires_grad=True)
        update = torch.randn(2, 4, 8, requires_grad=True)
        read, saved = module.read(HyperConnectionState(streams), nn.Identity())
        written = module.write(update, saved).streams
        ref_read, ref_written = gr_reference(module, streams, update)
        assert_close(read, ref_read)
        assert_close(written, ref_written)
        variables = (streams, update, *module.parameters())
        actual = torch.autograd.grad(
            read.square().sum() + written.square().sum(), variables, retain_graph=True
        )
        expected = torch.autograd.grad(
            ref_read.square().sum() + ref_written.square().sum(), variables
        )
        for a, b in zip(actual, expected):
            assert_close(a, b, atol=2e-5, rtol=2e-5)

    def test_mhc_forward_and_all_gradients_match_reference(self):
        module = ManifoldHyperConnection(8, 3, 1e-6, 20, 1e-6)
        # Non-symmetric routing is essential to catch a transposed combination.
        with torch.no_grad():
            module.mix_bias.normal_()
            module.mix_scale.fill_(0.3)
        streams = torch.randn(2, 4, 3, 8, requires_grad=True)
        incoming = torch.rand(2, 4, 3, requires_grad=True)
        update = torch.randn(2, 4, 8, requires_grad=True)
        norm = nn.RMSNorm(8, eps=1e-6)
        read, saved = module.read(HyperConnectionState(streams, incoming), norm)
        written = module.write(update, saved)
        ref_read, ref_written, ref_pre = mhc_reference(
            module, streams, incoming, update, norm
        )
        for a, b in (
            (read, ref_read),
            (written.streams, ref_written),
            (written.pre_mix, ref_pre),
        ):
            assert_close(a, b)
        variables = (
            streams,
            incoming,
            update,
            *module.parameters(),
            *norm.parameters(),
        )
        actual_loss = (
            read.square().sum()
            + written.streams.square().sum()
            + written.pre_mix.square().sum()
        )
        ref_loss = (
            ref_read.square().sum()
            + ref_written.square().sum()
            + ref_pre.square().sum()
        )
        actual = torch.autograd.grad(actual_loss, variables, retain_graph=True)
        expected = torch.autograd.grad(ref_loss, variables)
        for a, b in zip(actual, expected):
            assert_close(a, b, atol=3e-5, rtol=3e-5)
        assert_close(
            saved.combination.sum(-1), torch.ones_like(incoming), atol=2e-5, rtol=0
        )
        assert_close(
            saved.combination.sum(-2), torch.ones_like(incoming), atol=2e-5, rtol=0
        )

    def test_mhc_reads_incoming_mix_and_carries_new_mix_to_readout(self):
        module = ManifoldHyperConnection(8, 4, 1e-6, 20, 1e-6)
        hidden = torch.randn(2, 3, 8)
        state = expand_streams(hidden, 4, single_pass=True)
        assert_close(collapse_streams(state), hidden, atol=0, rtol=0)
        streams = torch.randn(2, 3, 4, 8)
        state = HyperConnectionState(streams, state.pre_mix)
        read, saved = module.read(state, nn.Identity())
        assert_close(read, streams[..., 0, :], atol=0, rtol=0)
        output = module.write(torch.zeros_like(hidden), saved)
        self.assertIs(output.pre_mix, saved.next_pre)
        assert_close(
            collapse_streams(output),
            (output.streams.float() * saved.next_pre.unsqueeze(-1)).sum(-2),
        )
        with self.assertRaisesRegex(ValueError, "incoming pre_mix"):
            module.read(HyperConnectionState(streams), nn.Identity())

    def test_mhc_coefficients_remain_float32_under_autocast(self):
        module = ManifoldHyperConnection(8, 4, 1e-6, 20, 1e-6)
        x = torch.randn(2, 3, 4, 8)
        expected = module.mixes(x)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = module.mixes(x)
        for a, b in zip(actual, expected):
            self.assertEqual(a.dtype, torch.float32)
            assert_close(a, b, atol=0, rtol=0)

    def test_invalid_config(self):
        for kwargs in (
            {"kind": "unknown"},
            {"num_streams": 0},
            {"gate_rank": True},
            {"sinkhorn_iters": -1},
            {"sinkhorn_eps": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ResidualConfig(**kwargs)


class ResidualModelTest(unittest.TestCase):
    def make_model(self, kind, cls=DFlashDraftModel, attention="gqa"):
        cfg = config(attention)
        cfg.draft_residual = {"kind": kind, "num_streams": 4, "gate_rank": 8}
        cfg.dflash_config.update(
            conv_kernel_size=2, conv_group_size=4, selector_rank=8, selector_top_k=4
        )
        return cls(cfg)

    def inputs(self):
        return dict(
            position_ids=torch.arange(7).unsqueeze(0),
            noise_embedding=torch.randn(2, 4, 16, requires_grad=True),
            target_hidden=torch.randn(2, 3, 16, requires_grad=True),
        )

    def test_family_models_compose_residuals_and_receive_gradients(self):
        for kind in ("gr", "mhc"):
            for cls in (DFlashDraftModel, DFlash2DraftModel, DSparkDraftModel):
                for attention in ("gqa", "mla"):
                    with self.subTest(kind=kind, cls=cls.__name__, attention=attention):
                        model = self.make_model(kind, cls, attention)
                        output = model(**self.inputs())
                        self.assertEqual(output.shape, (2, 4, 16))
                        output.square().mean().backward()
                        for name, param in model.named_parameters():
                            if "residual" in name:
                                self.assertIsNotNone(param.grad, name)
                                self.assertTrue(torch.isfinite(param.grad).all(), name)
                        if kind == "gr":
                            self.assertIsInstance(
                                model.layers[0].input_layernorm, nn.Identity
                            )
                            self.assertIsInstance(model.norm, nn.Identity)

    def test_dspark_objective_trains_residual_and_prediction_heads(self):
        for kind in ("gr", "mhc"):
            with self.subTest(kind=kind):
                cfg = config()
                cfg.is_causal = True
                cfg.draft_residual = {"kind": kind, "gate_rank": 8}
                cfg.dflash_config.update(
                    markov_rank=4, enable_confidence_head=True, mask_token_id=31
                )
                draft = DSparkDraftModel(cfg)
                model = OnlineDSparkModel(
                    draft,
                    nn.Linear(16, 32, bias=False).requires_grad_(False),
                    nn.Embedding(32, 16).requires_grad_(False),
                    mask_token_id=31,
                    block_size=4,
                    attention_backend="sdpa",
                    num_anchors=2,
                    dspark_confidence_head_alpha=1.0,
                    objective_chunk_blocks=1,
                )
                loss, _, _ = model(
                    input_ids=torch.randint(0, 30, (2, 12)),
                    hidden_states=torch.randn(2, 12, 16),
                    loss_mask=torch.ones(2, 12),
                    target_last_hidden_states=torch.randn(2, 12, 16),
                )
                loss.backward()
                for name, p in draft.named_parameters():
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
                for prefix in (
                    "layers.0.attention_residual",
                    "markov_head",
                    "confidence_head",
                ):
                    self.assertGreater(
                        sum(
                            p.grad.abs().sum().item()
                            for n, p in draft.named_parameters()
                            if n.startswith(prefix)
                        ),
                        0,
                        prefix,
                    )

    def test_checkpoint_recomputation_matches_outputs_and_gradients(self):
        for kind in ("gr", "mhc"):
            with self.subTest(kind=kind):
                model = self.make_model(kind)
                reference = copy.deepcopy(model)
                inputs = self.inputs()
                ref_inputs = {
                    k: v.detach().clone().requires_grad_(v.requires_grad)
                    for k, v in inputs.items()
                }
                # Exercise the HF layer checkpoint wrapper, including nested
                # residual state and mHC's differentiable incoming pre_mix.
                model.gradient_checkpointing_enable()
                actual = model(**inputs)
                expected = reference(**ref_inputs)
                assert_close(actual, expected, atol=0, rtol=0)
                actual.square().sum().backward()
                expected.square().sum().backward()
                for (name, a), (_, b) in zip(
                    model.named_parameters(), reference.named_parameters()
                ):
                    self.assertIsNotNone(a.grad, name)
                    assert_close(a.grad, b.grad, atol=0, rtol=0)

    def test_reentrant_checkpointing_is_rejected_before_training(self):
        for kind in ("gr", "mhc"):
            with self.assertRaisesRegex(ValueError, "use_reentrant=False"):
                self.make_model(kind).gradient_checkpointing_enable(
                    {"use_reentrant": True}
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_bfloat16_models_train_on_cuda(self):
        for kind in ("gr", "mhc"):
            model = self.make_model(kind).to(device="cuda", dtype=torch.bfloat16)
            inputs = {
                k: v.detach().to(
                    device="cuda",
                    dtype=torch.bfloat16 if v.is_floating_point() else v.dtype,
                )
                for k, v in self.inputs().items()
            }
            output = model(**inputs)
            self.assertEqual(output.dtype, torch.bfloat16)
            output.float().square().mean().backward()
            for name, p in model.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_resume_contract_records_non_parameter_semantics(self):
        standard = DFlashDraftModel(config())
        self.assertEqual(dflash_residual_resume_contract(standard), {})
        original = self.make_model("mhc")
        cfg = copy.deepcopy(original.config)
        cfg.draft_residual["sinkhorn_iters"] = 5
        modified = DFlashDraftModel(cfg)
        self.assertEqual(list(original.state_dict()), list(modified.state_dict()))
        self.assertNotEqual(
            dflash_residual_resume_contract(original),
            dflash_residual_resume_contract(modified),
        )
        self.assertNotEqual(
            dflash_residual_resume_contract(original),
            dflash_residual_resume_contract(self.make_model("gr")),
        )

    def test_save_load_preserves_config_weights_and_outputs(self):
        for kind in ("gr", "mhc"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                model = self.make_model(kind).eval()
                inputs = self.inputs()
                expected = model(**inputs)
                model.save_pretrained(directory)
                loaded = DFlashDraftModel.from_pretrained(directory).eval()
                self.assertEqual(loaded.residual_options, model.residual_options)
                assert_close(loaded(**inputs), expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
