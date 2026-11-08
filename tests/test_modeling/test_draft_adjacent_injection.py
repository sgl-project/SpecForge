"""Structural injection nodes, runtime replacement, and HF initialization."""

import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from specforge.config.draft.architecture.dspine import AdjacentInjectionSpec
from specforge.modeling.draft.dspine import AdjacentInjection as LegacyInjection
from specforge.modeling.draft.dspine import DSpineConfig
from specforge.modeling.draft.heads import AdjacentInjection
from specforge.modeling.draft.pretrained import DraftPreTrainedModel
from tests.test_modeling.test_draft_pretrained import model_config


class DFlashDraftModel(DraftPreTrainedModel):
    """Test-only shell for the generic HF initialization/restoration path."""

    def __init__(self, config):
        super().__init__(config)
        self.injection = AdjacentInjection(
            AdjacentInjectionSpec(transfer_rank=4, message_size=8),
            hidden_size=16,
            num_layers=2,
        )
        self.post_init()


class AdjacentInjectionTest(unittest.TestCase):
    def test_runtime_mask_has_no_implicit_training_layer_cutoff(self):
        legacy = LegacyInjection(
            16, 2, DSpineConfig(transfer_rank=4, message_size=8, replacement_layers=1)
        )
        node = AdjacentInjectionSpec(transfer_rank=4, message_size=8)
        module = AdjacentInjection(node, hidden_size=16, num_layers=2)
        with torch.no_grad():
            for parameter in legacy.parameters():
                parameter.normal_(std=0.1)
        module.load_state_dict(legacy.state_dict(), strict=True)
        hidden = torch.randn(2, 2, 4, 16)
        ids = torch.randint(0, 32, (2, 2, 4))
        codes = torch.randn(32, 4)
        replace = torch.ones(2, 2, dtype=torch.bool)
        self.assertIs(module.config, node)
        self.assertFalse(hasattr(module, "replacement_layers"))
        for layer in (0, 1):
            expected = legacy(hidden, layer, ids, codes, replace)
            actual = module(hidden, layer, ids, codes, replace if layer == 0 else None)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                actual[..., 0, :], hidden[..., 0, :], rtol=0, atol=0
            )
        # The standalone head honors an explicitly supplied mask even at later layers.
        self.assertFalse(
            torch.equal(
                module(hidden, 1, ids, codes, replace), module(hidden, 1, ids, codes)
            )
        )

    def test_identity_initialization_meta_and_loaded_parameters(self):
        node = AdjacentInjectionSpec(transfer_rank=4, message_size=8)
        module = AdjacentInjection(node, hidden_size=16, num_layers=2, device="meta")
        module.to_empty(device="cpu")
        module.reset_draft_parameters()
        special = [
            module.bias,
            module.predecessor_gate.weight,
            *(write.weight for write in module.writes),
        ]
        for parameter in special:
            torch.testing.assert_close(parameter, torch.zeros_like(parameter))
            with torch.no_grad():
                parameter.fill_(7)
            parameter._is_hf_initialized = True
        module.reset_draft_parameters()
        for parameter in special:
            torch.testing.assert_close(parameter, torch.full_like(parameter, 7))

    def test_generic_hf_initialization_and_roundtrip(self):
        model = DFlashDraftModel(model_config())
        hidden = torch.randn(2, 2, 4, 16)
        ids = torch.randint(0, 32, (2, 2, 4))
        codes = torch.randn(32, 4)
        torch.testing.assert_close(
            model.injection(hidden, 0, ids, codes), hidden, rtol=0, atol=0
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(std=0.1)
        output = model.injection(hidden, 0, ids, codes)
        model.post_init()
        torch.testing.assert_close(
            model.injection(hidden, 0, ids, codes), output, rtol=0, atol=0
        )
        with TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = DFlashDraftModel.from_pretrained(directory)
        torch.testing.assert_close(
            restored.state_dict(), model.state_dict(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            restored.injection(hidden, 0, ids, codes), output, rtol=0, atol=0
        )

    def test_invalid_constructor_inputs_fail_before_allocation(self):
        node = AdjacentInjectionSpec(transfer_rank=4, message_size=8)
        with patch("torch.nn.Linear") as allocate:
            for width, layers in ((0, 2), (16, 0), (True, 2), (16, 1.5)):
                with self.assertRaises(ValueError):
                    AdjacentInjection(node, hidden_size=width, num_layers=layers)
            with self.assertRaises(TypeError):
                AdjacentInjection(DSpineConfig(), hidden_size=16, num_layers=2)
            allocate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
