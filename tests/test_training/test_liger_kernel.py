from __future__ import annotations

import contextlib
import importlib.util
import io
import itertools
import os
import re
import sys
import tomllib
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm

from specforge.algorithms.dflash import providers
from specforge.modeling.draft import dflash_kernels
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.dflash2 import DFlash2DraftModel
from specforge.modeling.draft.dflash_kernels import (
    DEFAULT_DFLASH_KERNELS,
    LIGER_REQUIREMENT,
    DFlashKernels,
    resolve_liger_kernel_choice,
)
from specforge.training.checkpoint import CheckpointManager

REPO_ROOT = Path(__file__).resolve().parents[2]

_PLATFORMS = ("cuda", "rocm", "npu", "xpu", "cpu")
_ACTIVATIONS = ("silu", "swish", "gelu")


def _cfg(requested):
    return types.SimpleNamespace(
        model=types.SimpleNamespace(use_liger_kernel=requested),
    )


def _injected_kernels() -> DFlashKernels:
    return DFlashKernels(
        make_rms_norm=lambda hidden_size, eps: _InjectedRMSNorm(hidden_size, eps=eps),
        make_mlp=_InjectedMLP,
    )


def _importable_liger_modules() -> dict:
    """Stand-ins for ``triton`` and ``liger_kernel`` so CPU tests can import them."""
    package = types.ModuleType("liger_kernel")
    transformers = types.ModuleType("liger_kernel.transformers")
    transformers.LigerRMSNorm = _InjectedRMSNorm
    transformers.LigerSwiGLUMLP = _InjectedMLP
    package.transformers = transformers
    return {
        "triton": types.ModuleType("triton"),
        "liger_kernel": package,
        "liger_kernel.transformers": transformers,
    }


_MISSING_LIGER_MODULES = {"liger_kernel": None, "liger_kernel.transformers": None}


def _broken_liger_modules() -> dict:
    """An installed Liger whose own import fails on a missing transformers API."""
    modules = _importable_liger_modules()

    def missing_transformers_api(name):
        raise ImportError(
            "cannot import name 'RemovedApi' from 'transformers'", name="transformers"
        )

    del modules["liger_kernel.transformers"].LigerRMSNorm
    modules["liger_kernel.transformers"].__getattr__ = missing_transformers_api
    return modules


def _expected_choice(requested, platform, hidden_act):
    """Documented resolution: an ``(enabled, reason)`` pair or an error regex."""
    supported_act = hidden_act in ("silu", "swish")
    names = {"cuda": "CUDA", "rocm": "ROCm", "npu": "Ascend NPU", "xpu": "XPU"}
    if requested is False:
        return False, "model.use_liger_kernel=false"
    if requested is None:
        if platform not in ("cuda", "rocm"):
            return False, f"auto: trainer device '{platform}' is not CUDA/ROCm"
        if not supported_act:
            return False, f"auto: draft hidden_act '{hidden_act}' is not silu/swish"
        return True, f"auto: DFlash on {names[platform]}"
    if platform == "cpu":
        return (
            r"requires a Triton GPU backend: CUDA or ROCm, or Ascend NPU/XPU "
            r"where it is unvalidated in SpecForge; this trainer's device is 'cpu'"
        )
    if not supported_act:
        return "requires the draft hidden_act to be 'silu' or 'swish'.*got 'gelu'"
    if platform in ("npu", "xpu"):
        return True, f"explicit, unvalidated in SpecForge on {names[platform]}"
    return True, "explicit"


class TestLigerKernelResolution(unittest.TestCase):
    def test_choice_matrix_covers_every_request_platform_and_activation(self):
        """Verify auto/true/false resolve per platform and draft activation."""
        for requested, platform, hidden_act in itertools.product(
            (None, True, False), _PLATFORMS, _ACTIVATIONS
        ):
            expected = _expected_choice(requested, platform, hidden_act)
            with self.subTest(
                requested=requested, platform=platform, hidden_act=hidden_act
            ):
                if isinstance(expected, str):
                    with self.assertRaisesRegex(ValueError, expected):
                        resolve_liger_kernel_choice(
                            requested, platform=platform, hidden_act=hidden_act
                        )
                    continue
                choice = resolve_liger_kernel_choice(
                    requested, platform=platform, hidden_act=hidden_act
                )
                self.assertEqual((choice.enabled, choice.reason), expected)
                self.assertEqual(
                    choice.unvalidated, choice.enabled and platform in ("npu", "xpu")
                )

    def test_trainer_resolution_with_liger_importable_or_missing(self):
        """Verify resolved kernels load, and a missing Liger is a hard error."""
        for requested, platform, hidden_act, importable in itertools.product(
            (None, True, False), _PLATFORMS, ("silu", "gelu"), (True, False)
        ):
            expected = _expected_choice(requested, platform, hidden_act)
            modules = (
                _importable_liger_modules() if importable else _MISSING_LIGER_MODULES
            )
            with (
                self.subTest(
                    requested=requested,
                    platform=platform,
                    hidden_act=hidden_act,
                    importable=importable,
                ),
                mock.patch.object(
                    dflash_kernels, "current_liger_platform", return_value=platform
                ),
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(CheckpointManager, "is_rank0", return_value=False),
            ):
                draft_config = types.SimpleNamespace(hidden_act=hidden_act)
                if isinstance(expected, str):
                    with self.assertRaisesRegex(ValueError, expected):
                        providers.resolve_dflash_kernels(_cfg(requested), draft_config)
                elif not expected[0]:
                    self.assertIsNone(
                        providers.resolve_dflash_kernels(_cfg(requested), draft_config)
                    )
                elif importable:
                    kernels = providers.resolve_dflash_kernels(
                        _cfg(requested), draft_config
                    )
                    self.assertIsInstance(
                        kernels.make_rms_norm(4, 1e-6), _InjectedRMSNorm
                    )
                else:
                    with self.assertRaisesRegex(
                        ImportError,
                        r"model\.use_liger_kernel resolved to true \("
                        + expected[1]
                        + r"\) but Liger could not be imported.*"
                        r"or set model\.use_liger_kernel: false",
                    ):
                        providers.resolve_dflash_kernels(_cfg(requested), draft_config)

    def test_explicit_false_does_not_import_liger(self):
        """Verify an explicit false never resolves or imports Liger."""
        for platform in _PLATFORMS:
            with (
                self.subTest(platform=platform),
                mock.patch.object(
                    dflash_kernels, "current_liger_platform", return_value=platform
                ),
                mock.patch.object(
                    dflash_kernels, "load_liger_dflash_kernels"
                ) as loader,
                mock.patch.object(CheckpointManager, "is_rank0", return_value=False),
            ):
                kernels = providers.resolve_dflash_kernels(
                    _cfg(False), types.SimpleNamespace(hidden_act="silu")
                )
            self.assertIsNone(kernels)
            loader.assert_not_called()

    def test_import_error_hint_matches_the_platform_install(self):
        """Verify every hint installs the pinned Liger, --no-deps off CUDA."""
        pinned = re.escape(f'"{LIGER_REQUIREMENT}"')
        for platform, hint in (
            ("cuda", rf"Install it \(pip install {pinned}\)"),
            ("rocm", rf"pip install --no-deps {pinned} into the existing ROCm"),
            ("npu", rf"pip install --no-deps {pinned} into the existing torch_npu"),
            ("xpu", rf"pip install --no-deps {pinned} into the existing XPU"),
        ):
            choice = resolve_liger_kernel_choice(
                True, platform=platform, hidden_act="silu"
            )
            with (
                self.subTest(platform=platform),
                mock.patch.dict(sys.modules, _MISSING_LIGER_MODULES),
                self.assertRaisesRegex(ImportError, hint),
            ):
                dflash_kernels.load_liger_dflash_kernels(choice)

    def test_default_choice_uses_this_process_platform_hint(self):
        """Verify a caller without a choice gets its own platform's hint."""
        with (
            mock.patch.object(
                dflash_kernels, "current_liger_platform", return_value="rocm"
            ),
            mock.patch.dict(sys.modules, _MISSING_LIGER_MODULES),
            self.assertRaisesRegex(ImportError, r"pip install --no-deps .*ROCm"),
        ):
            dflash_kernels.load_liger_dflash_kernels()

    def test_missing_triton_is_reported_even_when_liger_is_installed(self):
        """Verify a missing Triton raises the same actionable error."""
        modules = {**_importable_liger_modules(), "triton": None}
        with (
            mock.patch.dict(sys.modules, modules),
            self.assertRaisesRegex(ImportError, r"could not be imported: .*triton"),
        ):
            dflash_kernels.load_liger_dflash_kernels()

    def test_installed_liger_that_fails_internally_is_not_told_to_install(self):
        """Verify an import failure inside an installed Liger names the cause."""
        with (
            mock.patch.dict(sys.modules, _broken_liger_modules()),
            self.assertRaises(ImportError) as raised,
        ):
            dflash_kernels.load_liger_dflash_kernels()
        message = str(raised.exception)
        self.assertRegex(
            message,
            r"importing Liger failed inside an installed package: cannot import "
            r"name 'RemovedApi' from 'transformers'\. Check that the installed "
            rf"liger-kernel matches {re.escape(LIGER_REQUIREMENT)}",
        )
        self.assertNotIn("Install it", message)

    def test_install_pin_matches_packaging_and_docs(self):
        """Verify packaging and every documented install use the hint's pin."""
        with open(REPO_ROOT / "pyproject.toml", "rb") as stream:
            project = tomllib.load(stream)["project"]
        self.assertIn(
            f"{LIGER_REQUIREMENT}; sys_platform == 'linux'", project["dependencies"]
        )
        for name in ("pyproject.toml", "pyproject_xpu.toml"):
            with open(REPO_ROOT / name, "rb") as stream:
                extras = tomllib.load(stream)["project"]["optional-dependencies"]
            self.assertEqual([LIGER_REQUIREMENT], extras["liger"], name)

        installs = []
        for root in ("docs/sections", "docs/web/.vitepress/theme", "examples"):
            for path in sorted((REPO_ROOT / root).rglob("*")):
                if path.suffix not in (".md", ".vue", ".yaml") or not path.is_file():
                    continue
                for line in path.read_text(encoding="utf-8").splitlines():
                    if re.search(r"pip install.*liger-kernel", line):
                        installs.append((path.relative_to(REPO_ROOT), line))
        self.assertTrue(installs)
        for path, line in installs:
            self.assertIn(f'"{LIGER_REQUIREMENT}"', line, str(path))

    def test_resolved_choice_is_printed_once_on_rank0(self):
        """Verify rank 0 reports the resolved choice and other ranks stay quiet."""
        draft_config = types.SimpleNamespace(hidden_act="silu")
        for is_rank0, expected in (
            (True, "[dflash] Liger kernels: on (auto: DFlash on CUDA)\n"),
            (False, ""),
        ):
            output = io.StringIO()
            with (
                self.subTest(rank0=is_rank0),
                mock.patch.object(CheckpointManager, "is_rank0", return_value=is_rank0),
                mock.patch.object(
                    dflash_kernels, "current_liger_platform", return_value="cuda"
                ),
                mock.patch.dict(sys.modules, _importable_liger_modules()),
                contextlib.redirect_stdout(output),
            ):
                providers.resolve_dflash_kernels(_cfg(None), draft_config)
            self.assertEqual(expected, output.getvalue())

    def test_explicit_true_on_unvalidated_backend_warns(self):
        """Verify NPU/XPU honour an explicit true and warn it is unvalidated."""
        for platform, name in (("npu", "Ascend NPU"), ("xpu", "XPU")):
            output = io.StringIO()
            with (
                self.subTest(platform=platform),
                mock.patch.object(
                    dflash_kernels, "current_liger_platform", return_value=platform
                ),
                mock.patch.dict(sys.modules, _importable_liger_modules()),
                mock.patch.object(CheckpointManager, "is_rank0", return_value=True),
                contextlib.redirect_stdout(output),
            ):
                kernels = providers.resolve_dflash_kernels(
                    _cfg(True), types.SimpleNamespace(hidden_act="silu")
                )
            self.assertIsNotNone(kernels)
            self.assertEqual(
                f"[dflash] Liger kernels: on (explicit, unvalidated in SpecForge "
                f"on {name}); compare the loss against model.use_liger_kernel: "
                "false\n",
                output.getvalue(),
            )

    def test_platform_detection_tells_rocm_apart_and_honours_override(self):
        """Verify the real platform probe: ROCm via torch.version.hip, env wins."""
        for device, hip, expected in (
            ("cuda", "6.4.0", "rocm"),
            ("cuda", None, "cuda"),
            ("npu", None, "npu"),
            ("cpu", "6.4.0", "cpu"),
        ):
            with (
                self.subTest(device=device, hip=hip),
                mock.patch("specforge.utils.get_device_type", return_value=device),
                mock.patch.object(torch.version, "hip", hip),
            ):
                self.assertEqual(expected, dflash_kernels.current_liger_platform())

        for forced, hip, expected in (
            ("npu", None, "npu"),
            ("cuda", "6.4.0", "rocm"),
        ):
            with (
                self.subTest(SPECFORGE_DEVICE=forced, hip=hip),
                mock.patch.dict(os.environ, {"SPECFORGE_DEVICE": forced}),
                mock.patch.object(torch.version, "hip", hip),
            ):
                self.assertEqual(expected, dflash_kernels.current_liger_platform())


class TestLigerKernelIntegration(unittest.TestCase):
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
        cfg = _cfg(True)
        draft_config = object()
        kernels = _injected_kernels()
        with (
            mock.patch.object(
                providers,
                "resolve_dflash_kernels",
                return_value=kernels,
            ) as resolve,
            mock.patch(
                "specforge.algorithms.model_providers.build_dflash_draft",
                return_value=mock.sentinel.model,
            ) as build,
        ):
            model = providers.build_draft(cfg, draft_config)

        self.assertIs(model, mock.sentinel.model)
        resolve.assert_called_once_with(cfg, draft_config)
        build.assert_called_once_with(cfg, draft_config, kernels)


@unittest.skipUnless(
    importlib.util.find_spec("liger_kernel") is not None,
    "requires liger-kernel",
)
class TestRealLigerKernelIntegration(unittest.TestCase):
    def test_explicit_false_keeps_native_modules_with_liger_installed(self):
        """Verify an explicit false keeps the native modules."""
        with contextlib.redirect_stdout(io.StringIO()):
            kernels = providers.resolve_dflash_kernels(_cfg(False), _draft_config())
        self.assertIsNone(kernels)

        vanilla = DFlashDraftModel(_draft_config(), dflash_kernels=kernels)
        self.assertIsInstance(vanilla.norm, Qwen3RMSNorm)
        self.assertIsInstance(vanilla.layers[0].mlp, Qwen3MLP)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_default_resolves_to_liger_on_a_gpu_trainer(self):
        """Verify the auto default builds Liger modules on CUDA/ROCm."""
        from liger_kernel.transformers import LigerRMSNorm, LigerSwiGLUMLP

        with contextlib.redirect_stdout(io.StringIO()):
            kernels = providers.resolve_dflash_kernels(_cfg(None), _draft_config())
        model = DFlashDraftModel(_draft_config(), dflash_kernels=kernels)
        self.assertIsInstance(model.norm, LigerRMSNorm)
        self.assertIsInstance(model.layers[0].mlp, LigerSwiGLUMLP)

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

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_dflash2_bf16_matches_native_outputs_and_gradients(self):
        """Verify bf16 Liger DFlash2 is as accurate as native for GQA and MLA."""
        for attention_mode in ("gqa", "mla"):
            with self.subTest(attention_mode=attention_mode):
                _assert_bf16_parity(
                    self,
                    _dflash2_config(attention_mode),
                    dflash_kernels.load_liger_dflash_kernels(),
                )


# Relative Frobenius error against the fp32 reference that bf16 Liger may
# always reach, about 2.5 bf16 unit roundoffs (2**-8). A native bf16 run of the
# drafts below lands between 0.7% and 2.4%.
_BF16_RELATIVE_ERROR_FLOOR = 1e-2


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = actual.float(), expected.float()
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item()


def _run_dflash2(model, noise, target_hidden, probe):
    """Forward and backward one draft; return output, input and weight grads."""
    noise = noise.detach().clone().requires_grad_(True)
    target_hidden = target_hidden.detach().clone().requires_grad_(True)
    position_ids = torch.arange(
        target_hidden.shape[1] + noise.shape[1], device=noise.device
    ).expand(noise.shape[0], -1)
    output = model(
        noise_embedding=noise,
        target_hidden=target_hidden,
        position_ids=position_ids,
    )
    (output.float() * probe).sum().backward()
    tensors = {
        "output": output.detach(),
        "noise.grad": noise.grad,
        "target_hidden.grad": target_hidden.grad,
    }
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            tensors[f"{name}.grad"] = parameter.grad
    return tensors


def _assert_bf16_parity(test, config, liger_kernels, *, device="cuda"):
    """Compare bf16 Liger and bf16 native against an fp32 native reference.

    Liger passes when its error is within twice native bf16's own error, or
    within the absolute bf16 floor, for the output, both input gradients and
    every weight gradient.
    """
    torch.manual_seed(0)
    reference = DFlash2DraftModel(config)
    with torch.no_grad():
        # Move norm weights off ones and the zero-initialized convolution
        # projection off zero so every kernel path carries signal.
        for parameter in reference.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    state = reference.state_dict()
    native = DFlash2DraftModel(config, dflash_kernels=DEFAULT_DFLASH_KERNELS)
    liger = DFlash2DraftModel(config, dflash_kernels=liger_kernels)
    native.load_state_dict(state, strict=True)
    liger.load_state_dict(state, strict=True)
    reference = reference.to(device)
    native = native.to(device, torch.bfloat16)
    liger = liger.to(device, torch.bfloat16)

    width = len(reference.target_layer_ids) * config.hidden_size
    noise = torch.randn(2, 8, config.hidden_size, device=device).to(torch.bfloat16)
    target_hidden = torch.randn(2, 12, width, device=device).to(torch.bfloat16)
    probe = torch.randn(2, 8, config.hidden_size, device=device)

    expected = _run_dflash2(reference, noise.float(), target_hidden.float(), probe)
    baseline = _run_dflash2(native, noise, target_hidden, probe)
    actual = _run_dflash2(liger, noise, target_hidden, probe)

    test.assertEqual(set(expected), set(actual))
    test.assertEqual(set(expected), set(baseline))
    for name, reference_tensor in expected.items():
        liger_error = _relative_error(actual[name], reference_tensor)
        native_error = _relative_error(baseline[name], reference_tensor)
        test.assertLessEqual(
            liger_error,
            max(2.0 * native_error, _BF16_RELATIVE_ERROR_FLOOR),
            f"{name}: bf16 Liger relative error {liger_error:.3e} vs native "
            f"{native_error:.3e} against fp32",
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


def _dflash2_config(attention_mode: str) -> Qwen3Config:
    """Two-layer DFlash2 draft; MLA also covers the low-rank query norm."""
    mla = (
        {
            "q_lora_rank": 32,
            "kv_lora_rank": 32,
            "qk_nope_head_dim": 16,
            "qk_rope_head_dim": 16,
            "v_head_dim": 16,
        }
        if attention_mode == "mla"
        else {}
    )
    config = Qwen3Config(
        architectures=["DFlash2DraftModel"],
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        num_target_layers=4,
        head_dim=32,
        max_position_embeddings=128,
        vocab_size=64,
        layer_types=["full_attention"] * 2,
        dflash_config={
            "attention_mode": attention_mode,
            "block_size": 4,
            "conv_group_size": 4,
            "conv_kernel_size": 2,
            "mask_token_id": 63,
            "selector_rank": 4,
            "selector_top_k": 3,
            "target_layer_ids": [1, 2],
        },
        **mla,
    )
    config._attn_implementation = "sdpa"
    return config


if __name__ == "__main__":
    unittest.main()
