import unittest
from importlib.metadata import PackageNotFoundError
from unittest import mock

import torch

from specforge.ops import dflash_kernels


class _Norm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))

    def forward(self, hidden_states):
        return hidden_states * self.weight


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_norm = _Norm()
        self.k_norm = _Norm()


class _MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(4, 8, bias=False)
        self.up_proj = torch.nn.Linear(4, 8, bias=False)
        self.down_proj = torch.nn.Linear(8, 4, bias=False)

    def forward(self, inputs):
        return self.down_proj(torch.relu(self.gate_proj(inputs)) * self.up_proj(inputs))


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = _Norm()
        self.post_attention_layernorm = _Norm()
        self.self_attn = _Attention()
        self.mlp = _MLP()


class _Draft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(), _Layer()])
        self.norm = _Norm()
        self.hidden_norm = _Norm()


def _liger_rms_forward(self, hidden_states):
    return hidden_states


def _liger_mlp_forward(self, inputs):
    return inputs


class TestDFlashDraftKernels(unittest.TestCase):
    def tearDown(self):
        dflash_kernels._load_liger_draft_forwards.cache_clear()

    def test_torch_backend_does_not_load_or_replace_kernels(self):
        draft = _Draft()
        original_forward = draft.layers[0].mlp.forward

        with mock.patch.object(
            dflash_kernels, "_load_liger_draft_forwards"
        ) as load_liger:
            dflash_kernels.configure_dflash_draft_kernels(draft, "torch")

        load_liger.assert_not_called()
        self.assertEqual(draft.draft_kernel_backend, "torch")
        self.assertIs(draft.layers[0].mlp.forward.__func__, original_forward.__func__)

    def test_liger_binding_is_local_and_preserves_parameters_and_state_keys(self):
        draft = _Draft()
        untouched = _Draft()
        keys_before = tuple(draft.state_dict())
        parameters_before = {name: value for name, value in draft.named_parameters()}

        with mock.patch.object(
            dflash_kernels,
            "_load_liger_draft_forwards",
            return_value=(_liger_rms_forward, _liger_mlp_forward),
        ):
            dflash_kernels.configure_dflash_draft_kernels(draft, "liger")

        self.assertEqual(draft.draft_kernel_backend, "liger")
        self.assertEqual(tuple(draft.state_dict()), keys_before)
        self.assertEqual(set(dict(draft.named_parameters())), set(parameters_before))
        for name, parameter in draft.named_parameters():
            self.assertIs(parameter, parameters_before[name])
        self.assertIs(draft.layers[0].mlp.forward.__func__, _liger_mlp_forward)
        self.assertIsNot(untouched.layers[0].mlp.forward.__func__, _liger_mlp_forward)
        for norm in (
            draft.norm,
            draft.hidden_norm,
            draft.layers[0].input_layernorm,
            draft.layers[0].post_attention_layernorm,
            draft.layers[0].self_attn.q_norm,
            draft.layers[0].self_attn.k_norm,
        ):
            self.assertIs(norm.forward.__func__, _liger_rms_forward)
            self.assertFalse(norm.in_place)
            self.assertEqual(norm.casting_mode, "llama")

    def test_liger_backend_fails_fast_when_package_is_missing(self):
        with mock.patch.object(
            dflash_kernels,
            "version",
            side_effect=PackageNotFoundError("liger-kernel"),
        ):
            with self.assertRaisesRegex(RuntimeError, r"specforge\[liger\]"):
                dflash_kernels.validate_dflash_draft_kernel_backend("liger")

    def test_liger_backend_rejects_unpinned_version(self):
        with mock.patch.object(dflash_kernels, "version", return_value="0.9.0"):
            with self.assertRaisesRegex(RuntimeError, "requires liger-kernel==0.8.0"):
                dflash_kernels.validate_dflash_draft_kernel_backend("liger")

    def test_unknown_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "draft_kernel_backend"):
            dflash_kernels.validate_dflash_draft_kernel_backend("automatic")


if __name__ == "__main__":
    unittest.main()
