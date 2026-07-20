from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from unittest import mock

import torch
from torch import nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm

from specforge.algorithms.dflash import providers
from specforge.modeling.draft import dflash_kernels
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.dflash_kernels import DFlashKernels


def _cfg(*, enabled: bool, strategy: str = "dflash"):
    return types.SimpleNamespace(
        model=types.SimpleNamespace(use_liger_kernel=enabled),
        training=types.SimpleNamespace(strategy=strategy),
    )


def _injected_kernels() -> DFlashKernels:
    return DFlashKernels(
        make_rms_norm=lambda hidden_size, eps: _InjectedRMSNorm(hidden_size, eps=eps),
        make_mlp=_InjectedMLP,
    )


class TestLigerKernelIntegration(unittest.TestCase):
    def test_disabled_config_does_not_import_liger(self):
        """Verify disabled config does not resolve or import Liger."""
        with mock.patch.object(dflash_kernels, "load_liger_dflash_kernels") as loader:
            kernels = providers.resolve_dflash_kernels(_cfg(enabled=False))
        self.assertIsNone(kernels)
        loader.assert_not_called()

    def test_dflash_returns_explicit_liger_factories(self):
        """Verify enabled DFlash config returns the selected Liger factories."""
        expected = _injected_kernels()
        with mock.patch.object(
            dflash_kernels,
            "load_liger_dflash_kernels",
            return_value=expected,
        ):
            kernels = providers.resolve_dflash_kernels(_cfg(enabled=True))
        self.assertIs(kernels, expected)

    def test_non_dflash_strategy_is_rejected_before_import(self):
        """Verify unsupported strategies fail before importing Liger."""
        with mock.patch.object(dflash_kernels, "load_liger_dflash_kernels") as loader:
            with self.assertRaisesRegex(ValueError, "strategy=dflash"):
                providers.resolve_dflash_kernels(_cfg(enabled=True, strategy="dspark"))
        loader.assert_not_called()

    def test_missing_optional_extra_has_an_actionable_error(self):
        """Verify a missing Liger extra raises actionable install guidance."""
        with mock.patch.dict(
            sys.modules,
            {"liger_kernel": None, "liger_kernel.transformers": None},
        ):
            with self.assertRaisesRegex(ImportError, r"specforge\[liger\]"):
                dflash_kernels.load_liger_dflash_kernels()

    def test_factories_are_injected_without_global_qwen3_patch(self):
        """Verify factories replace DFlash modules without global Qwen3 state."""
        injected = DFlashDraftModel(_draft_config(), dflash_kernels=_injected_kernels())
        layer = injected.layers[0]

        self.assertIsInstance(injected.norm, _InjectedRMSNorm)
        self.assertIsInstance(injected.hidden_norm, _InjectedRMSNorm)
        self.assertIsInstance(layer.input_layernorm, _InjectedRMSNorm)
        self.assertIsInstance(layer.post_attention_layernorm, _InjectedRMSNorm)
        self.assertIsInstance(layer.self_attn.q_norm, _InjectedRMSNorm)
        self.assertIsInstance(layer.self_attn.k_norm, _InjectedRMSNorm)
        self.assertIsInstance(layer.mlp, _InjectedMLP)

        vanilla = DFlashDraftModel(_draft_config())
        vanilla_layer = vanilla.layers[0]
        self.assertIsInstance(vanilla.norm, Qwen3RMSNorm)
        self.assertIsInstance(vanilla_layer.mlp, Qwen3MLP)

    def test_dflash_provider_passes_resolved_kernels_to_dedicated_builder(self):
        """Verify the DFlash provider forwards factories to its builder."""
        cfg = _cfg(enabled=True)
        draft_config = object()
        kernels = _injected_kernels()
        with (
            mock.patch.object(
                providers,
                "resolve_dflash_kernels",
                return_value=kernels,
            ),
            mock.patch(
                "specforge.algorithms.model_providers.build_dflash_draft",
                return_value=mock.sentinel.model,
            ) as build,
        ):
            model = providers.build_draft(cfg, draft_config)

        self.assertIs(model, mock.sentinel.model)
        build.assert_called_once_with(cfg, draft_config, kernels)


@unittest.skipUnless(
    importlib.util.find_spec("liger_kernel") is not None,
    "requires the specforge[liger] extra",
)
class TestRealLigerKernelIntegration(unittest.TestCase):
    def test_installed_liger_stays_disabled_without_opt_in(self):
        """Verify installing Liger alone does not enable its kernels."""
        self.assertIsNone(providers.resolve_dflash_kernels(_cfg(enabled=False)))

        vanilla = DFlashDraftModel(_draft_config())
        self.assertIsInstance(vanilla.norm, Qwen3RMSNorm)
        self.assertIsInstance(vanilla.layers[0].mlp, Qwen3MLP)

    def test_real_components_construct_with_compatible_state_dict(self):
        """Verify real Liger modules preserve DFlash checkpoint keys."""
        from liger_kernel.transformers import LigerRMSNorm, LigerSwiGLUMLP

        native = DFlashDraftModel(_draft_config())
        liger = DFlashDraftModel(
            _draft_config(),
            dflash_kernels=dflash_kernels.load_liger_dflash_kernels(),
        )

        self.assertIsInstance(liger.norm, LigerRMSNorm)
        self.assertIsInstance(liger.layers[0].mlp, LigerSwiGLUMLP)
        self.assertEqual(set(native.state_dict()), set(liger.state_dict()))
        liger.load_state_dict(native.state_dict(), strict=True)

        vanilla_after_liger = DFlashDraftModel(_draft_config())
        self.assertIsInstance(vanilla_after_liger.norm, Qwen3RMSNorm)
        self.assertIsInstance(vanilla_after_liger.layers[0].mlp, Qwen3MLP)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_real_components_match_forward_and_backward(self):
        """Verify real Liger matches native DFlash outputs and input gradients."""
        torch.manual_seed(0)
        device = torch.device("cuda")
        native = DFlashDraftModel(_draft_config()).to(device)
        liger = DFlashDraftModel(
            _draft_config(),
            dflash_kernels=dflash_kernels.load_liger_dflash_kernels(),
        ).to(device)
        liger.load_state_dict(native.state_dict(), strict=True)

        native_noise = torch.randn(2, 3, 16, device=device, requires_grad=True)
        liger_noise = native_noise.detach().clone().requires_grad_(True)
        target_hidden = torch.randn(2, 3, 16, device=device)
        full_position_ids = torch.arange(
            target_hidden.shape[1] + native_noise.shape[1],
            device=device,
        ).expand(native_noise.shape[0], -1)

        native_output = native(
            noise_embedding=native_noise,
            target_hidden=target_hidden.clone(),
            position_ids=full_position_ids,
        )
        liger_output = liger(
            noise_embedding=liger_noise,
            target_hidden=target_hidden.clone(),
            position_ids=full_position_ids,
        )
        torch.testing.assert_close(liger_output, native_output, rtol=1e-4, atol=1e-5)

        native_output.square().mean().backward()
        liger_output.square().mean().backward()
        torch.testing.assert_close(
            liger_noise.grad,
            native_noise.grad,
            rtol=2e-4,
            atol=2e-5,
        )


class _InjectedRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return hidden_states


class _InjectedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.proj(hidden_states)


def _draft_config():
    config = Qwen3Config(
        architectures=["DFlashDraftModel"],
        block_size=4,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        num_target_layers=4,
        head_dim=4,
        max_position_embeddings=64,
        vocab_size=32,
    )
    config._attn_implementation = "eager"
    return config


if __name__ == "__main__":
    unittest.main()
