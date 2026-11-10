"""DSpine injection is structure; replacement masks and losses belong outside."""

import dataclasses
import json
import unittest
from typing import ClassVar
from unittest.mock import patch

from specforge.config.draft import registry
from specforge.config.draft.architecture import DSpineSpec
from specforge.config.draft.architecture.dspine import AdjacentInjectionSpec
from specforge.config.draft.codec import decode_config, encode_config
from specforge.config.draft.attention import AttentionSpec
from specforge.config.draft.layer import LayerSpec
from specforge.config.draft.residual import ResidualSpec
from tests.test_config.draft_fixtures import draft_config


def dspine_config():
    base = draft_config()
    return dataclasses.replace(
        base,
        architecture="DSpineDraftModel",
        attention_configs={
            "full": dataclasses.replace(base.attention_configs["full"], is_causal=True)
        },
        spec_config=DSpineSpec(
            block=base.spec_config.block,
            target_features=base.spec_config.target_features,
            injection=AdjacentInjectionSpec(transfer_rank=8, message_size=4),
        ),
    )


class DSpineSpecTest(unittest.TestCase):
    def test_causal_capability_does_not_require_an_is_causal_field(self):
        @dataclasses.dataclass(frozen=True, kw_only=True)
        class CausalAttention(AttentionSpec):
            type: ClassVar[str] = "test_always_causal"

            def capabilities(self) -> frozenset[str]:
                return frozenset({"causal", "target_context_kv"})

            @property
            def rotary_input_dim(self):
                return None

        with patch.dict(registry._SCHEMAS):
            registry.register_component_schema("attention", CausalAttention)
            dataclasses.replace(
                dspine_config(),
                attention_configs={"full": CausalAttention(rope_parameters=None)},
            )

    def test_injection_structure_round_trips_without_schedules(self):
        node = dspine_config()
        payload = encode_config(node)
        self.assertEqual(
            payload["spec_config"]["injection"], {"transfer_rank": 8, "message_size": 4}
        )
        restored = decode_config(json.loads(json.dumps(payload)))
        self.assertEqual(restored, node)
        self.assertEqual(hash(restored), hash(node))
        self.assertEqual(node.spec_config.protocol.num_draft_tokens, 7)
        self.assertEqual(node.spec_config.protocol.verify_window, 8)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            node.spec_config.injection.transfer_rank = 4

    def test_injection_fields_are_required_positive_integers(self):
        for name in ("transfer_rank", "message_size"):
            for value in (0, -1, True, 2.0, "4"):
                with self.assertRaisesRegex(ValueError, name):
                    AdjacentInjectionSpec(
                        **({"transfer_rank": 8, "message_size": 4} | {name: value})
                    )
            payload = encode_config(dspine_config())
            payload["spec_config"]["injection"].pop(name)
            with self.assertRaisesRegex(ValueError, name):
                decode_config(payload)
        with self.assertRaisesRegex(ValueError, "injection"):
            dataclasses.replace(dspine_config().spec_config, injection={})

    def test_transfer_rank_checks_hidden_width_and_centered_vocabulary(self):
        base = dspine_config()
        for hidden, vocab, rank in ((4, 64, 5), (32, 8, 8)):
            spec = dataclasses.replace(
                base.spec_config,
                injection=AdjacentInjectionSpec(transfer_rank=rank, message_size=4),
                target_features=dataclasses.replace(
                    base.spec_config.target_features, mask_token_id=0
                ),
            )
            with self.assertRaisesRegex(ValueError, "transfer_rank"):
                dataclasses.replace(
                    base,
                    hidden_size=hidden,
                    vocab_size=vocab,
                    draft_vocab_size=vocab,
                    spec_config=spec,
                )
        dataclasses.replace(
            base,
            spec_config=dataclasses.replace(
                base.spec_config,
                injection=AdjacentInjectionSpec(transfer_rank=32, message_size=4),
            ),
        )

    def test_every_attention_must_be_explicitly_causal(self):
        base = dspine_config()
        attention = base.attention_configs["full"]
        for sliding_window in (None, 16):
            with self.assertRaisesRegex(ValueError, "attention_configs.other.*causal"):
                dataclasses.replace(
                    base,
                    attention_configs={
                        "full": attention,
                        "other": dataclasses.replace(
                            attention, is_causal=False, sliding_window=sliding_window
                        ),
                    },
                    layer_configs={
                        "first": LayerSpec(attention="full", mlp="dense"),
                        "second": LayerSpec(attention="other", mlp="dense"),
                    },
                    layer_types=("first", "second"),
                )
        self.assertTrue(attention.is_causal)

    def test_injection_requires_declared_single_stream_residuals(self):
        @dataclasses.dataclass(frozen=True, kw_only=True)
        class MultipleStreams(ResidualSpec):
            type: ClassVar[str] = "test_multiple_streams"

        with patch.dict(registry._SCHEMAS):
            registry.register_component_schema("residual", MultipleStreams)
            with self.assertRaisesRegex(ValueError, "single_stream"):
                dataclasses.replace(dspine_config(), residual_config=MultipleStreams())

        @dataclasses.dataclass(frozen=True, kw_only=True)
        class MutableCapabilities(ResidualSpec):
            type: ClassVar[str] = "test_mutable_residual"
            capabilities: ClassVar[set[str]] = {"single_stream"}

        with self.assertRaisesRegex(TypeError, "capabilities must be a method"):
            registry.register_component_schema("residual", MutableCapabilities)

    def test_training_and_candidate_selection_are_not_model_structure(self):
        for field in (
            "ce_weight",
            "alignment_weight",
            "alignment_warmup_steps",
            "replacement_layers",
            "replacement_probability",
            "replacement_hold_ratio",
            "replacement_end_ratio",
            "candidate_mass_threshold",
            "top_k",
        ):
            payload = encode_config(dspine_config())
            payload["spec_config"][field] = 1
            with self.assertRaisesRegex(ValueError, field):
                decode_config(payload)
            payload = encode_config(dspine_config())
            payload["spec_config"]["injection"][field] = 1
            with self.assertRaisesRegex(ValueError, field):
                decode_config(payload)


if __name__ == "__main__":
    unittest.main()
