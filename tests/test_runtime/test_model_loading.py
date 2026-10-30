# coding=utf-8
"""Inline draft settings and compatible weights-only checkpoint loading."""

import json
import os
import tempfile
import unittest
from unittest import mock

import torch

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.config import Config
from specforge.training.model_loading import (
    resolve_draft_config,
    warm_start_draft_model,
)


def _run_config(strategy: str, **model_overrides) -> Config:
    model = {
        "target_model_path": "target/model",
        **model_overrides,
    }
    if strategy == "eagle3":
        model["vocab_mapping_path"] = "/mapping.pt"
    data = (
        {"train_data_path": "/train.jsonl"}
        if strategy == "peagle"
        else {"hidden_states_path": "/features"}
    )
    payload = {
        "model": model,
        "data": data,
        "training": {"strategy": strategy},
    }
    if strategy == "peagle":
        model["target_backend"] = "sglang"
        payload["training"].update(
            {
                "attention_backend": "flex_attention",
                "batch_size": 1,
                "max_steps": 1,
            }
        )
        payload["deployment"] = {
            "mode": "disaggregated",
            "disaggregated": {
                "control_dir": "/control",
                "backend": "mooncake",
                "server_urls": ["http://capture:30000"],
            },
        }
    return Config.model_validate(payload)


def _draft_config_provider(strategy: str):
    registration = builtin_algorithm_registry().resolve(strategy)
    return registration.providers.model.draft_config


def _draft_payload(architecture: str, *, layers: int = 1, block_size=None):
    payload = {
        "architectures": [architecture],
        "model_type": "qwen3" if architecture == "DFlashDraftModel" else "llama",
        "vocab_size": 128,
        "draft_vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 256,
        "rms_norm_eps": 1e-5,
        "hidden_act": "silu",
        "pad_token_id": 0,
        "tie_word_embeddings": False,
    }
    if block_size is not None:
        payload.update(
            {
                "block_size": block_size,
                "num_target_layers": 12,
                "dflash_config": {},
            }
        )
    return payload


class DraftConfigResolutionTest(unittest.TestCase):
    def test_inline_settings_are_authoritative_for_every_weight_source(self):
        import copy

        from tests.draft_config_utils import draft_settings

        for checkpoint in (None, "org/draft", "/run/training_state.pt"):
            payload = draft_settings("dflash")
            original = copy.deepcopy(payload)
            cfg = _run_config(
                "dflash", draft_model_config=payload, draft_checkpoint_path=checkpoint
            )
            with mock.patch(
                "specforge.config.draft.AutoDraftConfig.from_pretrained"
            ) as read:
                resolved = resolve_draft_config(
                    cfg, provider=_draft_config_provider("dflash")
                )
            read.assert_not_called()
            self.assertEqual(resolved.dflash_config["target_layer_ids"], (6,))
            self.assertEqual(cfg.model.draft_model_config, original)
            with self.assertRaises(TypeError):
                resolved.dflash_config["mask_token_id"] = 9
            self.assertEqual(cfg.model.draft_model_config, original)

    def test_null_vocabulary_inherits_inline_vocab(self):
        from tests.draft_config_utils import draft_settings

        payload = draft_settings("dflash")
        payload["draft_vocab_size"] = None
        cfg = _run_config("dflash", draft_model_config=payload)
        resolved = resolve_draft_config(cfg, provider=_draft_config_provider("dflash"))
        self.assertEqual(resolved.draft_vocab_size, payload["vocab_size"])

    def test_missing_training_settings_fail_before_allocation(self):
        from tests.draft_config_utils import draft_settings

        for strategy, field in (
            ("dflash", "mask_token_id"),
            ("eagle3", "eagle_aux_hidden_state_layer_ids"),
        ):
            payload = draft_settings(strategy)
            method = "dflash_config" if strategy == "dflash" else "eagle_config"
            payload[method].pop(field)
            cfg = _run_config(strategy, draft_model_config=payload)
            with self.assertRaisesRegex(ValueError, field):
                resolve_draft_config(cfg, provider=_draft_config_provider(strategy))

    def test_contract_rejects_same_shape_different_semantics(self):
        from specforge.config.draft import AutoDraftConfig
        from specforge.config.draft.contract import (
            draft_config_contract,
            validate_draft_config_contract,
        )
        from tests.draft_config_utils import draft_settings

        payload = draft_settings("dflash")
        current = AutoDraftConfig.from_dict(payload)
        for field, value in (("rope_theta", 12345), ("is_causal", True)):
            saved = AutoDraftConfig.from_dict({**payload, field: value})
            with self.assertRaisesRegex(ValueError, field):
                validate_draft_config_contract(
                    draft_config_contract(current),
                    draft_config_contract(saved),
                    source="checkpoint",
                )

    def test_contract_ignores_execution_and_serialization_metadata(self):
        from specforge.config.draft import AutoDraftConfig
        from specforge.config.draft.contract import draft_config_contract
        from tests.draft_config_utils import draft_settings

        current = AutoDraftConfig.from_dict(draft_settings("dflash"))
        saved = AutoDraftConfig.from_dict(current.to_dict())
        saved._name_or_path = "/old/path"
        saved._attn_implementation = "sdpa"
        saved.dtype = torch.bfloat16
        self.assertEqual(draft_config_contract(current), draft_config_contract(saved))

    def test_native_mtp_normalizes_hf_rope_parameters(self):
        from specforge.config.draft import AutoDraftConfig
        from specforge.modeling.draft.models.specialized.qwen3_5_mtp.model import (
            PartialRotaryEmbedding,
        )
        from tests.draft_config_utils import draft_settings

        payload = draft_settings("mtp")
        payload["rope_parameters"] = {
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.5,
        }
        config = AutoDraftConfig.finalize(AutoDraftConfig.from_dict(payload))
        self.assertEqual(config.rope_theta, 10000000)
        self.assertEqual(config.partial_rotary_factor, 0.5)
        rotary = PartialRotaryEmbedding(config, config.head_dim)
        expected = 1.0 / (10000000 ** (torch.arange(0, 4, 2).float() / 4))
        torch.testing.assert_close(rotary.inv_freq, expected)

    def test_native_mtp_semantics_checked_without_mutating_inline_config(self):
        import copy
        from types import SimpleNamespace

        from specforge.config.draft import AutoDraftConfig
        from specforge.config.draft.mtp import Qwen3_5MTPConfig
        from tests.draft_config_utils import draft_settings

        current = AutoDraftConfig.finalize(
            AutoDraftConfig.from_dict(draft_settings("mtp"))
        )
        original = copy.deepcopy(current.to_dict())
        target = SimpleNamespace(**current.to_dict())
        target.num_hidden_layers = 48
        Qwen3_5MTPConfig.validate_native_config(current, target)
        target.partial_rotary_factor = 0.5
        with self.assertRaisesRegex(ValueError, "partial_rotary_factor"):
            Qwen3_5MTPConfig.validate_native_config(current, target)
        self.assertEqual(current.to_dict(), original)


class _TinyDraft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(4, 3)
        self.proj = torch.nn.Linear(3, 2)


class WarmStartTest(unittest.TestCase):
    def setUp(self):
        from specforge.config.draft import AutoDraftConfig
        from tests.draft_config_utils import draft_settings

        self.config = AutoDraftConfig.from_dict(draft_settings("dflash"))

    def _write_runtime_state(self, directory, state, *, strategy="dflash"):
        path = os.path.join(directory, "training_state.pt")
        from specforge.config.draft.contract import draft_config_contract

        torch.save(
            {
                "specforge_model_provenance": (
                    ("draft_settings", draft_config_contract(self.config)),
                ),
                "draft_state_dict": state,
                "strategy": strategy,
                # Warm start must ignore every field below.
                "global_step": 91,
                "epoch": 7,
                "backend": {
                    "optimizer": {"state": {1: {"step": torch.tensor(91)}}},
                    "rng": torch.tensor([123]),
                },
            },
            path,
        )
        return path

    def test_semantic_mismatch_rejected_before_any_weight_is_loaded(self):
        source = _TinyDraft()
        destination = _TinyDraft()
        original = {k: v.clone() for k, v in destination.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_runtime_state(directory, source.state_dict())
            self.config.rope_theta = 12345
            with self.assertRaisesRegex(ValueError, "rope_theta"):
                warm_start_draft_model(
                    destination, path, draft_config=self.config, strategy="dflash"
                )
        for key, value in destination.state_dict().items():
            self.assertTrue(torch.equal(value, original[key]))

    def test_missing_saved_settings_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "training_state.pt")
            torch.save({"draft_state_dict": _TinyDraft().state_dict()}, path)
            with self.assertRaisesRegex(ValueError, "no saved draft settings"):
                warm_start_draft_model(
                    _TinyDraft(), path, draft_config=self.config, strategy="dflash"
                )

    def test_specforge_checkpoint_loads_only_draft_weights(self):
        torch.manual_seed(1)
        source = _TinyDraft()
        destination = _TinyDraft()
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_runtime_state(directory, source.state_dict())
            with mock.patch("torch.load", wraps=torch.load) as load:
                report = warm_start_draft_model(
                    destination,
                    path,
                    draft_config=self.config,
                    strategy="dflash",
                )
        self.assertEqual(report.checkpoint_format, "specforge")
        self.assertEqual(report.loaded_keys, len(source.state_dict()))
        self.assertTrue(report.loaded_embedding)
        self.assertTrue(
            all(
                torch.equal(destination.state_dict()[key], value)
                for key, value in source.state_dict().items()
            )
        )
        self.assertTrue(load.call_args.kwargs["weights_only"])

    def test_eagle_checkpoint_may_omit_target_copied_embedding(self):
        source = _TinyDraft()
        destination = _TinyDraft()
        original_embedding = destination.embed_tokens.weight.detach().clone()
        state = {
            key: value
            for key, value in source.state_dict().items()
            if "embed" not in key
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_runtime_state(directory, state, strategy="eagle3")
            report = warm_start_draft_model(
                destination,
                path,
                draft_config=self.config,
                strategy="eagle3",
                allow_missing_embedding=True,
            )
        self.assertFalse(report.loaded_embedding)
        self.assertIn("embed_tokens.weight", report.missing_keys)
        self.assertTrue(
            torch.equal(destination.embed_tokens.weight, original_embedding)
        )
        self.assertTrue(torch.equal(destination.proj.weight, source.proj.weight))

    def test_missing_non_embedding_weights_fail_closed(self):
        source = _TinyDraft()
        destination = _TinyDraft()
        state = {"embed_tokens.weight": source.embed_tokens.weight.detach().clone()}
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_runtime_state(directory, state)
            with self.assertRaisesRegex(ValueError, "missing draft weights"):
                warm_start_draft_model(
                    destination,
                    path,
                    draft_config=self.config,
                    strategy="dflash",
                )

    def test_runtime_checkpoint_strategy_must_match(self):
        source = _TinyDraft()
        destination = _TinyDraft()
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_runtime_state(
                directory, source.state_dict(), strategy="peagle"
            )
            with self.assertRaisesRegex(ValueError, "written by strategy"):
                warm_start_draft_model(
                    destination,
                    path,
                    draft_config=self.config,
                    strategy="eagle3",
                )

    def test_pretrained_repo_uses_registered_hf_loader(self):
        source = _TinyDraft()
        destination = _TinyDraft()
        with (
            mock.patch(
                "specforge.modeling.auto.AutoDraftModel.from_pretrained",
                return_value=(source, {"missing_keys": []}),
            ) as load,
            mock.patch(
                "specforge.config.draft.AutoDraftConfig.from_pretrained",
                return_value=self.config,
            ),
        ):
            report = warm_start_draft_model(
                destination,
                "org/base-draft",
                draft_config=self.config,
                strategy="dflash",
                cache_dir="/cache",
                trust_remote_code=True,
            )
        self.assertEqual(report.checkpoint_format, "pretrained")
        load.assert_called_once_with(
            "org/base-draft",
            config=mock.ANY,
            cache_dir="/cache",
            trust_remote_code=True,
            output_loading_info=True,
        )
        self.assertTrue(torch.equal(destination.proj.weight, source.proj.weight))

    def test_nested_draft_head_checkpoint_keys_are_migrated(self):
        from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig

        cases = (
            (
                "domino",
                {
                    **_draft_payload("DominoDraftModel", layers=1, block_size=4),
                    "layer_types": ["full_attention"],
                    "dflash_config": {
                        "projector_type": "domino",
                        "emb_dim": 16,
                        "gru_hidden_dim": 16,
                        "pure_draft_prefix_len": 0,
                        "shift_label": False,
                    },
                },
                ("prefix_gru.", "embed_proj."),
            ),
            (
                "dspark",
                {
                    **_draft_payload("DSparkDraftModel", layers=1, block_size=4),
                    "layer_types": ["full_attention"],
                    "dflash_config": {
                        "projector_type": "dspark",
                        "markov_rank": 8,
                        "markov_head_type": "vanilla",
                        "confidence_head_alpha": 1.0,
                        "confidence_head_with_markov": True,
                    },
                },
                ("markov_head.", "confidence_head."),
            ),
        )
        for strategy, payload, head_prefixes in cases:
            with (
                self.subTest(strategy=strategy),
                tempfile.TemporaryDirectory() as directory,
            ):
                config_path = os.path.join(directory, "config.json")
                with open(config_path, "w", encoding="utf-8") as stream:
                    json.dump(payload, stream)
                config = AutoDraftModelConfig.from_file(config_path)
                source = AutoDraftModel.from_config(config)
                destination = AutoDraftModel.from_config(config)
                checkpoint_state = {}
                migrated = []
                for key, value in source.state_dict().items():
                    checkpoint_key = key
                    for head_prefix in head_prefixes:
                        if key.startswith(head_prefix):
                            checkpoint_key = "logit_head." + key
                            migrated.append(checkpoint_key)
                            break
                    checkpoint_state[checkpoint_key] = value.detach().clone()
                self.assertTrue(migrated)
                self.config = config
                state_path = self._write_runtime_state(
                    directory, checkpoint_state, strategy=strategy
                )
                report = warm_start_draft_model(
                    destination,
                    state_path,
                    draft_config=config,
                    strategy=strategy,
                )
                self.assertEqual(report.loaded_keys, len(checkpoint_state))
                self.assertTrue(
                    all(
                        torch.equal(destination.state_dict()[key], value)
                        for key, value in source.state_dict().items()
                    )
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
