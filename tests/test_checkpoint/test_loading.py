"""Public loading and strict initialization, with small real checkpoints."""

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from specforge.checkpoint import save_draft
from specforge.checkpoint.resolve import _ADAPTERS, register_adapter
from specforge.checkpoint.tensors import TensorStore
from specforge.config.draft.codec import decode_legacy_config
from specforge.loading import initialize_draft, load_draft, read_draft
from specforge.modeling import build_draft_model
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.config import AutoDraftConfig
from tests.test_config.draft_fixtures import draft_config
from tests.test_config.test_draft_legacy_dflash import legacy_payload
from tests.test_config.test_draft_legacy_domino import domino_payload
from tests.test_config.test_draft_legacy_dspark import dspark_payload
from tests.test_config.test_draft_legacy_dspine import dspine_payload
from tests.test_config.test_draft_legacy_eagle3 import eagle_payload
from tests.test_config.test_draft_legacy_mtp import mtp_payload
from tests.test_config.test_draft_legacy_peagle import peagle_payload
from tests.test_modeling import test_canonical_dflash as _dflash
from tests.test_modeling import test_canonical_domino as _domino
from tests.test_modeling import test_canonical_eagle3 as _eagle3
from tests.test_modeling import test_canonical_peagle as _peagle
from tests.test_modeling.test_canonical_dflash import inputs_for


def _mtp_legacy_name(name):
    # the MTP forward-parity map, inlined: its test module is retired later in the stack
    for prefix, legacy in (
        ("fusion.", ""),
        ("backbone.layers.", "layers."),
        ("backbone.final_norm.", "norm."),
    ):
        if name.startswith(prefix):
            return legacy + name[len(prefix) :]
    return name


# canonical -> legacy names, as the forward-parity tests prove them
_LEGACY_NAMES = {
    "legacy_payload": lambda name: _dflash.legacy_name(name) or name,
    "dflash2_payload": lambda name: _dflash.legacy_name(name) or name,
    "dspark_payload": lambda name: _dflash.legacy_name(name) or name,
    "dspine_payload": lambda name: _dflash.legacy_name(name) or name,
    "domino_payload": _domino.reference_name,
    "eagle_payload": _eagle3.legacy_name,
    "peagle_payload": _peagle.legacy_name,
    "mtp_payload": _mtp_legacy_name,
}


def dflash2_payload():
    payload = legacy_payload() | {"architectures": ["DFlash2DraftModel"]}
    payload["dflash_config"].update(
        conv_kernel_size=3, conv_group_size=8, selector_rank=4, selector_top_k=3
    )
    return payload


def write_target(path, extra=None):
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"hidden_size": 32, "vocab_size": 64, "num_hidden_layers": 128})
    )
    tensors = {
        "model.embed_tokens.weight": torch.randn(64, 32),
        "lm_head.weight": torch.randn(64, 32),
    } | (extra or {})
    save_file(tensors, path / "model.safetensors")
    return tensors


class LoadingTest(unittest.TestCase):
    def test_legacy_mtp_binds_mup_head_in_training_dtype(self):
        payload = mtp_payload() | dict(
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=64,
            draft_vocab_size=64,
            mtp_config={"share_lm_head": True},
        )
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "target"
            weights = write_target(target)
            config_path = target / "config.json"
            target_config = json.loads(config_path.read_text()) | {
                "logits_mup_width_multiplier": 3
            }
            config_path.write_text(json.dumps(target_config))
            config = decode_legacy_config(payload, target_config=target_config)
            for dtype in (torch.float32, torch.bfloat16):
                model = initialize_draft(config, target=target, dtype=dtype)
                expected = weights["lm_head.weight"].to(dtype) / 3
                torch.testing.assert_close(
                    model.lm_head.weight, expected, rtol=0, atol=0
                )
                hidden = torch.randn(2, 32, dtype=dtype)
                torch.testing.assert_close(
                    model.lm_head(hidden),
                    torch.nn.functional.linear(hidden, expected),
                    rtol=0,
                    atol=0,
                )

    def test_explicit_legacy_settings_preserve_recorded_values(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures/legacy_peagle"
        payload = json.loads((fixture / "config.json").read_text())
        payload.pop("norm_before_residual", None)
        with TemporaryDirectory() as tmp:
            path, target = Path(tmp) / "draft", Path(tmp) / "target"
            path.mkdir()
            write_target(target)
            state = {"draft_state_dict": load_file(fixture / "model.safetensors")}
            torch.save(state, path / "training_state.pt")
            settings = {"mask_token_id": 63, "norm_before_residual": True}
            config = read_draft(
                path, target=target, legacy_config=payload, legacy_settings=settings
            ).config
            self.assertEqual(config.spec_config.mask_token_id, 63)
            self.assertTrue(config.spec_config.norm_before_residual)
            state["peagle_mask_token_id"] = 62
            torch.save(state, path / "training_state.pt")
            with self.assertRaisesRegex(ValueError, "explicit mask_token_id conflicts"):
                read_draft(
                    path, target=target, legacy_config=payload, legacy_settings=settings
                )

    def test_legacy_dflash_records_must_match_supplied_config(self):
        """A config-less DFlash run imported with a stale JSON must not load silently."""
        payload = dflash2_payload()
        with TemporaryDirectory() as tmp:
            target, source = Path(tmp) / "target", Path(tmp) / "source"
            target.mkdir()
            (target / "config.json").write_text(
                json.dumps(
                    {
                        "hidden_size": payload["hidden_size"],
                        "vocab_size": payload["vocab_size"],
                        "num_hidden_layers": payload["num_target_layers"],
                    }
                )
            )
            source.mkdir()
            (source / "config.json").write_text(json.dumps(payload))
            save_file({"fc.weight": torch.zeros(1)}, source / "model.safetensors")
            spec = read_draft(source, target=target).config.spec_config
            matching = {
                "dflash_target_layer_ids": spec.target_features.target_layer_ids,
                "dflash_mask_token_id": spec.target_features.mask_token_id,
                "dflash_block_size": spec.block.block_size,
            }
            torch.save(matching, source / "training_state.pt")
            read_draft(source, target=target)
            for key, value in (
                ("dflash_mask_token_id", spec.target_features.mask_token_id + 1),
                ("dflash_block_size", spec.block.block_size + 1),
                (
                    "dflash_target_layer_ids",
                    spec.target_features.target_layer_ids[::-1],
                ),
            ):
                with self.subTest(key=key):
                    torch.save(matching | {key: value}, source / "training_state.pt")
                    with self.assertRaisesRegex(ValueError, key):
                        read_draft(source, target=target)

    def test_legacy_peagle_residual_order_comes_from_resume_contract(self):
        """main's YAML may override norm_before_residual; the JSON can omit it."""
        payload = peagle_payload() | dict(
            hidden_size=32, vocab_size=64, draft_vocab_size=64
        )
        payload.pop("norm_before_residual", None)
        with TemporaryDirectory() as tmp:
            target, source = Path(tmp) / "target", Path(tmp) / "source"
            write_target(target)
            source.mkdir()
            (source / "config.json").write_text(json.dumps(payload))
            save_file({"fc.weight": torch.zeros(1)}, source / "model.safetensors")
            for recorded in (True, False):
                with self.subTest(recorded=recorded):
                    torch.save(
                        {
                            "peagle_mask_token_id": 63,
                            "peagle_norm_before_residual": recorded,
                        },
                        source / "training_state.pt",
                    )
                    spec = read_draft(source, target=target).config.spec_config
                    self.assertIs(spec.norm_before_residual, recorded)

    def test_legacy_capture_layers_come_from_training_provenance(self):
        fixture = Path(__file__).resolve().parents[1] / "fixtures/legacy_peagle"
        payload = json.loads((fixture / "config.json").read_text())
        payload.pop("eagle_config")
        with TemporaryDirectory() as tmp:
            path, target = Path(tmp) / "draft", Path(tmp) / "target"
            path.mkdir()
            write_target(target)
            torch.save(
                {
                    "draft_state_dict": load_file(fixture / "model.safetensors"),
                    "specforge_model_provenance": (("capture_layers", (2, 4, 6)),),
                    "peagle_mask_token_id": 63,
                },
                path / "training_state.pt",
            )
            for architecture in ("PEagleDraftModel", "LlamaForCausalLMEagle3"):
                payload["architectures"] = [architecture]
                config = read_draft(path, target=target, legacy_config=payload).config
                self.assertEqual(
                    config.spec_config.target_features.target_layer_ids, (2, 4, 6)
                )
                for layers in ([2, 4, 6], [1, 2, 3]):
                    supplied = payload | {
                        "eagle_config": {"eagle_aux_hidden_state_layer_ids": layers}
                    }
                    if layers == [2, 4, 6]:
                        self.assertEqual(
                            read_draft(
                                path, target=target, legacy_config=supplied
                            ).config,
                            config,
                        )
                    else:
                        with self.assertRaisesRegex(
                            ValueError, "capture_layers conflict"
                        ):
                            read_draft(path, target=target, legacy_config=supplied)

    def test_canonical_forward_restore_and_structure_guard(self):
        with TemporaryDirectory() as tmp:
            path, target = Path(tmp) / "draft", Path(tmp) / "target"
            write_target(target)
            original = initialize_draft(draft_config(), target=target)
            args = inputs_for(original.config.draft)
            save_draft(original, path)
            # Canonical and historical DFlash share a class name, not a format.
            self.assertEqual(
                original.config.architectures, legacy_payload()["architectures"]
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                read_draft(path, adapter="legacy")
            with self.assertRaisesRegex(ValueError, "canonical"):
                AutoDraftConfig.from_dict(original.config.to_dict())
            with self.assertRaisesRegex(ValueError, "canonical"):
                AutoDraftModel.from_config(original.config)
            with patch("huggingface_hub.snapshot_download", return_value=str(path)):
                restored = load_draft("test/draft", target=target)
            torch.testing.assert_close(
                restored(*args).logits, original(*args).logits, rtol=0, atol=0
            )
            self.assertEqual(read_draft(path).config, original.config.draft)
            with self.assertRaisesRegex(ValueError, "structure"):
                load_draft(
                    path,
                    target=target,
                    draft=replace(original.config.draft, initializer_range=0.1),
                )
            (target / "config.json").write_text(
                json.dumps(
                    {
                        "hidden_size": 32,
                        "vocab_size": 64,
                        "num_hidden_layers": 128,
                        "logits_mup_width_multiplier": 2,
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "target_logit_transform"):
                load_draft(path, target=target)

    def test_warm_start_and_target_copy_are_explicit_and_disjoint(self):
        config = draft_config(target_bindings=(), target_logit_transform=None)
        with TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "source", Path(tmp) / "target"
            original = build_draft_model(config)
            save_draft(original, source)
            extra = {
                "teacher." + k.replace("self_attn.", "attention."): v.clone()
                for k, v in original.backbone.layers[0].state_dict().items()
            }
            write_target(target, extra)
            copy = {
                "backbone.layers.1": {
                    "from": "teacher",
                    "rename": {"attention": "self_attn"},
                }
            }
            fresh = initialize_draft(config, target=target, from_target=copy)
            torch.testing.assert_close(
                fresh.backbone.layers[1].state_dict(),
                original.backbone.layers[0].state_dict(),
                rtol=0,
                atol=0,
            )
            changed = replace(
                config,
                spec_config=replace(
                    config.spec_config,
                    block=replace(config.spec_config.block, block_size=4),
                ),
            )
            warm = initialize_draft(changed, weights_from=source)
            self.assertEqual(warm.config.draft, changed)
            torch.testing.assert_close(
                warm.state_dict(), original.state_dict(), rtol=0, atol=0
            )
            with patch.object(
                TensorStore, "read", side_effect=AssertionError("preflight read")
            ):
                for kwargs in (
                    {
                        "weights_from": source,
                        "target": target,
                        "from_target": copy,
                    },
                    {
                        "target": target,
                        "from_target": copy,
                        "dtype": torch.bfloat16,
                    },
                    {"weights_from": source, "mapping": {"fusion": "fusion"}},
                ):
                    with self.assertRaises(ValueError):
                        initialize_draft(config, **kwargs)
            partial = {
                name: name
                for name in (
                    "backbone.layers.0",
                    "backbone.final_norm",
                    "fusion",
                    "embed_tokens",
                    "lm_head",
                )
            }
            combined = initialize_draft(
                config,
                target=target,
                weights_from=source,
                mapping=partial,
                from_target=copy,
            )
            torch.testing.assert_close(
                combined.backbone.layers[1].self_attn.q_proj.weight,
                original.backbone.layers[0].self_attn.q_proj.weight,
                rtol=0,
                atol=0,
            )
            allowed = ["backbone", "embed_tokens", "lm_head"]
            initialize_draft(
                config,
                weights_from=source,
                mapping={"fusion": "fusion"},
                allow_new=allowed,
            )

    def test_all_legacy_model_layouts_and_configless_training_weights(self):
        factories = [
            legacy_payload,
            dflash2_payload,
            dspark_payload,
            dspine_payload,
            domino_payload,
            eagle_payload,
            peagle_payload,
            mtp_payload,
        ]
        for factory in factories:
            with self.subTest(factory=factory.__name__), TemporaryDirectory() as tmp:
                payload = factory() | dict(
                    hidden_size=32,
                    intermediate_size=64,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=8,
                    vocab_size=64,
                    draft_vocab_size=64,
                    bos_token_id=None,
                    eos_token_id=None,
                    pad_token_id=None,
                )
                if "dflash_config" in payload:
                    payload.update(
                        num_hidden_layers=2,
                        num_target_layers=128,
                        layer_types=["full_attention"] * 2,
                    )
                    payload["dflash_config"].update(
                        mask_token_id=63, target_layer_ids=[1, 3]
                    )
                if factory in (eagle_payload, peagle_payload):
                    payload["target_hidden_size"] = 32
                if factory is eagle_payload:
                    payload["draft_vocab_size"] = 32
                    payload["fc_norm"] = True
                old = AutoDraftModel.from_config(
                    AutoDraftConfig.from_dict(payload)
                ).float()
                if factory is eagle_payload:
                    old.t2d[32:] = False
                path, target = Path(tmp) / "draft", Path(tmp) / "target"
                path.mkdir()
                target_weights = write_target(target)
                with torch.no_grad():
                    if factory is eagle_payload or factory is mtp_payload:
                        old.embed_tokens.weight.copy_(
                            target_weights["model.embed_tokens.weight"]
                        )
                    if factory is mtp_payload:
                        old.mtp.lm_head.weight.copy_(target_weights["lm_head.weight"])
                (path / "config.json").write_text(json.dumps(payload))
                state = old.state_dict()
                if factory in (domino_payload, dspark_payload):
                    state = {
                        (
                            "logit_head." + k
                            if k.startswith(
                                (
                                    "prefix_gru.",
                                    "embed_proj.",
                                    "markov_head.",
                                    "confidence_head.",
                                )
                            )
                            else k
                        ): v
                        for k, v in state.items()
                    }
                torch.save(state, path / "pytorch_model.bin")
                if factory is peagle_payload:
                    torch.save({"peagle_mask_token_id": 63}, path / "training_state.pt")
                if factory in (domino_payload, dspark_payload):
                    expected = read_draft(path, target=target).config
                    payload["architectures"] = ["DFlashDraftModel"]
                    (path / "config.json").write_text(json.dumps(payload))
                    self.assertEqual(read_draft(path, target=target).config, expected)
                restored = load_draft(path, target=target)
                self.assertEqual(
                    restored.config.draft, read_draft(path, target=target).config
                )
                artifact = read_draft(path, target=target)
                # by value, through the maps the forward-parity tests prove
                old_state = old.state_dict()
                to_legacy = _LEGACY_NAMES[factory.__name__]
                for key, value in restored.state_dict().items():
                    if key == "vocabulary.ready":
                        continue
                    legacy = to_legacy(key)
                    if legacy not in old_state:
                        legacy = "mtp." + legacy
                    torch.testing.assert_close(
                        value, old_state[legacy], rtol=0, atol=0, msg=key
                    )
                if factory is eagle_payload:
                    self.assertTrue(restored.vocabulary.ready)
                    invalid = dict(old.state_dict())
                    invalid["t2d"] = ~invalid["t2d"]
                    torch.save(invalid, path / "pytorch_model.bin")
                    with self.assertRaisesRegex(
                        ValueError, "same ascending target IDs"
                    ):
                        load_draft(path, target=target)
                    state = {
                        k: v
                        for k, v in old.state_dict().items()
                        if k not in {"t2d", "d2t"}
                    }
                    torch.save(state, path / "pytorch_model.bin")
                    with self.assertRaisesRegex(ValueError, "missing draft tensors"):
                        load_draft(path, target=target)
                    warm = initialize_draft(
                        artifact.config,
                        target=target,
                        weights_from=path,
                        allow_new=["vocabulary"],
                    )
                    self.assertFalse(bool(warm.vocabulary.ready))
                if factory is legacy_payload:
                    (path / "pytorch_model.bin").unlink()
                    (path / "config.json").unlink()
                    torch.save(
                        {"draft_state_dict": old.state_dict(), "step": 42},
                        path / "training_state.pt",
                    )
                    with self.assertRaisesRegex(ValueError, "legacy_config"):
                        load_draft(path, target=target)
                    load_draft(path, target=target, legacy_config=payload)

    def test_format_detection_and_ambiguous_target_keys(self):
        with TemporaryDirectory() as tmp:
            path, target = Path(tmp) / "draft", Path(tmp) / "target"
            save_draft(build_draft_model(draft_config()), path)
            write_target(target, {"embed_tokens.weight": torch.randn(64, 32)})
            with self.assertRaisesRegex(ValueError, "expected one target key"):
                load_draft(path, target=target)
            register_adapter(
                "test_duplicate", lambda p: True, _ADAPTERS["canonical"][1]
            )
            try:
                with self.assertRaisesRegex(ValueError, "adapter"):
                    read_draft(path)
                self.assertEqual(
                    read_draft(path, adapter="canonical").config, draft_config()
                )
            finally:
                del _ADAPTERS["test_duplicate"]
            payload = json.loads((path / "config.json").read_text())
            payload["specforge_artifact"] = {"format": "sglang", "version": 1}
            (path / "config.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "serving-only"):
                read_draft(path)
