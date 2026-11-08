"""The effective draft config owns immutable structure and validates its graph."""

import dataclasses
import subprocess
import sys
import unittest
from typing import ClassVar
from unittest.mock import patch

from specforge.config.draft import DraftConfig, registry
from specforge.config.draft.architecture import DFlash2Spec, DFlashSpec
from specforge.config.draft.attention import GQASpec
from specforge.config.draft.conv import GroupedCausalConvSpec
from specforge.config.draft.layer import LayerSpec
from specforge.config.draft.logits import MuPLogitTransformSpec
from specforge.config.draft.registry import register_component_schema, register_spec
from specforge.config.draft.rope import DefaultRoPESpec
from tests.test_config.draft_fixtures import draft_config


class DraftConfigNodesTest(unittest.TestCase):
    def test_graph_is_an_immutable_snapshot(self):
        source = {"full_dense": LayerSpec(attention="full", mlp="dense")}
        node = draft_config(layer_configs=source)
        before = hash(node)
        source.clear()
        self.assertEqual(node.num_hidden_layers, 2)
        self.assertEqual(len(node.layer_configs), 1)
        self.assertEqual(
            node, draft_config(target_bindings=("lm_head", "embed_tokens"))
        )
        self.assertEqual(hash(node), before)
        for obj, field, value in (
            (node, "hidden_size", 128),
            (node.spec_config.target_features, "mask_token_id", 0),
            (node.layer_configs, "_entries", ()),
        ):
            with self.assertRaises(dataclasses.FrozenInstanceError):
                setattr(obj, field, value)
        changed = dataclasses.replace(node, layer_types=("full_dense",))
        self.assertEqual(changed.num_hidden_layers, 1)
        self.assertEqual(node.num_hidden_layers, 2)
        with self.assertRaisesRegex(TypeError, "num_hidden_layers"):
            draft_config(num_hidden_layers=3)

    def test_dimensions_and_unknown_fields_are_strict(self):
        for name in (
            "hidden_size",
            "vocab_size",
            "draft_vocab_size",
            "max_position_embeddings",
        ):
            for value in (True, 0, -1, 2.0, "32"):
                with self.assertRaisesRegex(ValueError, name):
                    draft_config(**{name: value})
        for value in (True, -0.1, float("nan"), "0.02"):
            with self.assertRaisesRegex(ValueError, "initializer_range"):
                draft_config(initializer_range=value)
        with self.assertRaisesRegex(ValueError, "boolean"):
            draft_config(tie_word_embeddings=0)
        for field in ("model_type", "schema_version", "auto_map", "weights_from"):
            with self.assertRaisesRegex(TypeError, field):
                draft_config(**{field: {}})

    def test_bindings_are_closed_and_logit_settings_are_conditional(self):
        for value in ([], ("decoder",), ("lm_head", "lm_head"), (None,), ({},)):
            with self.assertRaisesRegex(ValueError, "target_bindings"):
                draft_config(target_bindings=value)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            draft_config(tie_word_embeddings=True)
        with self.assertRaisesRegex(ValueError, "explicit supported"):
            draft_config(target_logit_transform=None)
        with self.assertRaisesRegex(ValueError, "requires lm_head"):
            draft_config(target_bindings=("embed_tokens",))
        draft_config(
            target_bindings=(), target_logit_transform=None, tie_word_embeddings=True
        )
        draft_config(target_logit_transform=MuPLogitTransformSpec(multiplier=8))

    def test_model_and_architecture_nodes_must_match(self):
        with self.assertRaisesRegex(ValueError, "unknown architecture"):
            draft_config(architecture="UnknownDraftModel")
        with self.assertRaisesRegex(ValueError, "DFlash2Spec"):
            draft_config(architecture="DFlash2DraftModel")
        for name in (
            "norm_config",
            "residual_config",
            "rope_parameters",
            "spec_config",
        ):
            with self.assertRaisesRegex(ValueError, name):
                draft_config(**{name: {}})
        with self.assertRaisesRegex(ValueError, "mask_token_id"):
            node = draft_config().spec_config
            draft_config(
                spec_config=dataclasses.replace(
                    node,
                    target_features=dataclasses.replace(
                        node.target_features, mask_token_id=64
                    ),
                )
            )
        with self.assertRaisesRegex(ValueError, "draft_vocab_size"):
            draft_config(draft_vocab_size=32)

    def test_rope_geometry_is_validated_against_attention(self):
        with self.assertRaisesRegex(
            ValueError, "attention_configs.full.*even rotary dimension"
        ):
            draft_config(
                rope_parameters=DefaultRoPESpec(
                    rope_theta=10000, partial_rotary_factor=0.375
                )
            )

    def test_dflash2_checks_its_own_constraints(self):
        base = draft_config()
        spec = DFlash2Spec(
            block=base.spec_config.block,
            target_features=base.spec_config.target_features,
            selector_rank=8,
            selector_top_k=4,
            output_multiplier=1.0,
            final_logit_softcapping=None,
        )
        with self.assertRaisesRegex(ValueError, "requires sublayer_conv"):
            dataclasses.replace(
                base, architecture="DFlash2DraftModel", spec_config=spec
            )
        node = dataclasses.replace(
            base,
            architecture="DFlash2DraftModel",
            spec_config=spec,
            conv_configs={"conv": GroupedCausalConvSpec(kernel_size=3, group_size=8)},
            layer_configs={
                "full_dense": LayerSpec(
                    attention="full", mlp="dense", sublayer_conv="conv"
                )
            },
        )
        self.assertEqual(node.spec_config.protocol.num_draft_tokens, 7)
        with self.assertRaisesRegex(ValueError, "selector_top_k"):
            dataclasses.replace(
                node, spec_config=dataclasses.replace(spec, selector_top_k=65)
            )

    def test_capabilities_are_declared_by_the_schema(self):
        @dataclasses.dataclass(frozen=True, kw_only=True)
        class WithoutContext(GQASpec):
            type: ClassVar[str] = "test_without_context"
            capabilities: ClassVar[frozenset[str]] = frozenset()

        with patch.dict(registry._SCHEMAS):
            register_component_schema("attention", WithoutContext)
            node = WithoutContext(
                **dataclasses.asdict(draft_config().attention_configs["full"])
            )
            with self.assertRaisesRegex(ValueError, "requires target_context_kv"):
                draft_config(attention_configs={"full": node})

    def test_architecture_registration_keeps_validation_separate(self):
        seen = []

        def validate(node):
            seen.append(node.architecture)

        with patch.dict(registry._SPECS):
            register_spec("TestDraft", DFlashSpec, validate)
            register_spec("TestDraft", DFlashSpec, validate)
            node = draft_config(architecture="TestDraft")
            self.assertIsInstance(node, DraftConfig)
            self.assertEqual(seen, ["TestDraft"])
            with self.assertRaisesRegex(ValueError, "already registered"):
                register_spec("TestDraft", DFlash2Spec, validate)
            with self.assertRaisesRegex(TypeError, "ArchitectureSpec"):
                register_spec("Invalid", GQASpec, validate)
            with self.assertRaisesRegex(TypeError, "callable"):
                register_spec("Invalid", DFlashSpec, None)
            with self.assertRaisesRegex(ValueError, "class identifier"):
                register_spec("invalid/name", DFlashSpec, validate)

    def test_mutable_capability_declarations_are_rejected(self):
        @dataclasses.dataclass(frozen=True, kw_only=True)
        class MutableCapabilities(GQASpec):
            type: ClassVar[str] = "test_mutable_capabilities"
            capabilities: ClassVar[set[str]] = {"target_context_kv"}

        with self.assertRaisesRegex(TypeError, "capabilities must be a frozenset"):
            register_component_schema("attention", MutableCapabilities)

    def test_model_config_import_is_dependency_free(self):
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                """
import sys
from tests.test_config.draft_fixtures import draft_config
config = draft_config()
assert config.num_hidden_layers == 2
assert not any(x in sys.modules for x in ('torch', 'transformers', 'pydantic', 'specforge.runtime'))
""",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
