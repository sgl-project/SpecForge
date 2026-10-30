"""Native draft RoPE must be checked before model or weight allocation."""

import copy
import unittest
from types import SimpleNamespace
from unittest import mock

from specforge.config.draft import AutoDraftConfig
from specforge.config.draft.mtp import Qwen3_5MTPConfig
from specforge.config.draft.rope import resolve_rope_type, validate_native_rope_type
from tests.draft_config_utils import draft_settings


class NativeRopeTypeTest(unittest.TestCase):
    def test_default_and_legacy_representations_are_equivalent(self):
        inline = SimpleNamespace(rope_parameters={"rope_type": "default"})
        for fields in (
            {},
            {"rope_scaling": None},
            {"rope_scaling": {"type": "default"}},
            {"rope_parameters": {"type": "default"}},
            {"rope_type": "default"},
        ):
            with self.subTest(fields=fields):
                native = SimpleNamespace(**fields)
                original = copy.deepcopy(vars(native))
                validate_native_rope_type(inline, native, supported_types={"default"})
                self.assertEqual(vars(native), original)

    def test_type_mismatch_is_rejected_in_both_directions(self):
        for inline_type, native_type in (("default", "linear"), ("linear", "default")):
            with self.subTest(inline=inline_type, native=native_type):
                with self.assertRaisesRegex(ValueError, "rope_type does not match"):
                    validate_native_rope_type(
                        SimpleNamespace(rope_parameters={"rope_type": inline_type}),
                        SimpleNamespace(
                            rope_scaling={"type": native_type, "factor": 2.0}
                        ),
                        supported_types={"default", "linear"},
                    )

    def test_matching_unsupported_types_are_rejected(self):
        config = SimpleNamespace(rope_parameters={"rope_type": "linear", "factor": 2.0})
        with self.assertRaisesRegex(ValueError, "unsupported native draft rope_type"):
            validate_native_rope_type(config, config, supported_types={"default"})

    def test_ambiguous_or_malformed_declarations_do_not_default(self):
        for fields in (
            {"rope_parameters": {"rope_type": "default", "type": "linear"}},
            {
                "rope_parameters": {"rope_type": "default"},
                "rope_scaling": {"type": "linear"},
            },
            {"rope_parameters": {"full_attention": {"rope_type": "linear"}}},
            {"rope_scaling": {"factor": 2.0}},
            {"rope_parameters": {"rope_type": None}},
            {"rope_parameters": []},
        ):
            with self.subTest(fields=fields):
                with self.assertRaises(ValueError):
                    resolve_rope_type(SimpleNamespace(**fields), source="native draft")


class NativeMtpRopeTest(unittest.TestCase):
    def setUp(self):
        self.config = AutoDraftConfig.finalize(
            AutoDraftConfig.from_dict(draft_settings("mtp"))
        )

    def test_nested_text_config_with_matching_rope_passes(self):
        native = SimpleNamespace(text_config=copy.deepcopy(self.config))
        before = copy.deepcopy(native.text_config.to_dict())
        Qwen3_5MTPConfig.validate_native_config(self.config, native)
        self.assertEqual(native.text_config.to_dict(), before)

    def test_native_scaling_is_rejected_for_modern_and_legacy_configs(self):
        for field, parameters in (
            ("rope_parameters", {"rope_type": "linear", "factor": 2.0}),
            ("rope_scaling", {"type": "linear", "factor": 2.0}),
        ):
            with self.subTest(field=field):
                fields = self.config.to_dict()
                fields.pop("rope_parameters", None)
                fields.pop("rope_scaling", None)
                fields[field] = parameters
                native = SimpleNamespace(text_config=SimpleNamespace(**fields))
                with self.assertRaisesRegex(ValueError, "rope_type does not match"):
                    Qwen3_5MTPConfig.validate_native_config(self.config, native)

    def test_equal_types_still_require_matching_numerical_parameters(self):
        for field, value in (("rope_theta", 12345.0), ("partial_rotary_factor", 0.5)):
            with self.subTest(field=field):
                native = copy.deepcopy(self.config)
                setattr(native, field, value)
                with self.assertRaisesRegex(ValueError, field):
                    Qwen3_5MTPConfig.validate_native_config(self.config, native)

    def test_direct_mtp_config_rejects_legacy_scaled_rope(self):
        config = copy.deepcopy(self.config)
        del config.rope_parameters
        config.rope_scaling = {"type": "linear", "factor": 2.0}
        with self.assertRaisesRegex(ValueError, "supports only default RoPE"):
            Qwen3_5MTPConfig.from_model_config(config)

    def test_provider_rejects_mismatch_before_constructing_model_or_reading_weights(
        self,
    ):
        from specforge.algorithms.mtp.providers import build_draft

        native = copy.deepcopy(self.config)
        native.rope_parameters = {"rope_type": "linear", "factor": 2.0}
        cfg = SimpleNamespace(
            model=SimpleNamespace(
                draft_checkpoint_path=None,
                target_model_path="unused/native",
                cache_dir=None,
                trust_remote_code=False,
            )
        )
        with (
            mock.patch(
                "specforge.modeling.target.target_utils.load_target_config",
                return_value=native,
            ),
            mock.patch(
                "specforge.modeling.auto.AutoDraftModel.from_config"
            ) as construct,
            mock.patch(
                "specforge.algorithms.mtp.providers._init_from_native_mtp"
            ) as load,
        ):
            with self.assertRaisesRegex(ValueError, "rope_type does not match"):
                build_draft(cfg, self.config)
        construct.assert_not_called()
        load.assert_not_called()


class DraftCheckpointRopeTest(unittest.TestCase):
    def test_dspark_checkpoint_type_or_scaling_mismatch_fails_before_loading_weights(
        self,
    ):
        from specforge.training.model_loading import warm_start_draft_model

        for inline_rope, saved_rope, field in (
            (
                {"rope_type": "default"},
                {"rope_type": "linear", "factor": 2.0},
                "rope_type",
            ),
            (
                {"rope_type": "linear", "factor": 2.0},
                {"rope_type": "linear", "factor": 4.0},
                "factor",
            ),
        ):
            with self.subTest(field=field):
                payload = draft_settings("dspark")
                current = AutoDraftConfig.finalize(
                    AutoDraftConfig.from_dict(
                        {**payload, "rope_parameters": inline_rope}
                    )
                )
                saved = AutoDraftConfig.finalize(
                    AutoDraftConfig.from_dict(
                        {**payload, "rope_parameters": saved_rope}
                    )
                )
                destination = mock.Mock()
                with (
                    mock.patch(
                        "specforge.config.draft.AutoDraftConfig.from_pretrained",
                        return_value=saved,
                    ),
                    mock.patch(
                        "specforge.modeling.auto.AutoDraftModel.from_pretrained"
                    ) as load,
                ):
                    with self.assertRaisesRegex(ValueError, field):
                        warm_start_draft_model(
                            destination,
                            "org/dspark-draft",
                            draft_config=current,
                            strategy="dspark",
                        )
                load.assert_not_called()
                destination.load_state_dict.assert_not_called()
