# coding=utf-8
"""Draft config sources, target-derived defaults, and weights-only loading."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.config import Config
from specforge.training.model_loading import (
    load_draft_config_source,
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


def _target_config(*, layers: int = 12):
    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        pad_token_id=0,
    )


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
    def test_config_resolution_does_not_initialize_cuda_model_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "draft.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(_draft_payload("DominoDraftModel", block_size=16), stream)
            script = """
import sys
import torch
import specforge.data.preprocessing
import specforge.modeling.auto
from specforge.training.model_loading import load_draft_config_source

config = load_draft_config_source(sys.argv[1])
assert config.architectures == ["DominoDraftModel"]
assert config.block_size == 16
assert "yunchang" not in sys.modules
assert not torch.cuda.is_initialized()
"""
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = ""
            subprocess.run(
                [sys.executable, "-c", script, path],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )

    def test_target_derived_defaults_match_legacy_trainers(self):
        cases = (
            ("eagle3", "LlamaForCausalLMEagle3", 1, None),
            ("peagle", "PEagleDraftModel", 4, None),
            ("dflash", "DFlashDraftModel", 1, 16),
        )
        for strategy, architecture, layers, block_size in cases:
            with self.subTest(strategy=strategy):
                cfg = _run_config(strategy)
                with mock.patch(
                    "transformers.AutoConfig.from_pretrained",
                    return_value=_target_config(),
                ):
                    resolved = resolve_draft_config(
                        cfg,
                        provider=_draft_config_provider(strategy),
                    )
                self.assertEqual(resolved.architectures, [architecture])
                self.assertEqual(resolved.num_hidden_layers, layers)
                if block_size is not None:
                    self.assertEqual(resolved.block_size, block_size)
                    self.assertEqual(resolved.num_target_layers, 12)
                    self.assertEqual(len(resolved.dflash_config["target_layer_ids"]), 1)
                else:
                    self.assertEqual(resolved.draft_vocab_size, 32000)

    def test_dflash_typed_overrides_rebuild_capture_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "draft.json")
            payload = _draft_payload("DFlashDraftModel", layers=5, block_size=16)
            payload["dflash_config"] = {"target_layer_ids": [1, 3, 5, 7, 9]}
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(payload, stream)
            cfg = _run_config(
                "dflash",
                draft_model_config=path,
                draft_num_hidden_layers=2,
                draft_block_size=8,
            )
            resolved = resolve_draft_config(
                cfg,
                provider=_draft_config_provider("dflash"),
            )
        self.assertEqual(resolved.num_hidden_layers, 2)
        self.assertEqual(resolved.block_size, 8)
        self.assertEqual(len(resolved.dflash_config["target_layer_ids"]), 2)

    def test_local_json_and_directory_are_equivalent_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "config.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(_draft_payload("LlamaForCausalLMEagle3"), stream)
            from_file = load_draft_config_source(path)
            from_directory = load_draft_config_source(directory)
        self.assertEqual(from_file.to_dict(), from_directory.to_dict())

    def test_hf_repository_is_a_supported_config_source(self):
        remote = _target_config(layers=1)
        remote.architectures = ["LlamaForCausalLMEagle3"]
        remote.draft_vocab_size = 32
        with mock.patch(
            "transformers.AutoConfig.from_pretrained", return_value=remote
        ) as load:
            resolved = load_draft_config_source(
                "org/draft-model",
                cache_dir="/cache",
                trust_remote_code=True,
            )
        self.assertEqual(resolved.architectures, ["LlamaForCausalLMEagle3"])
        load.assert_called_once_with(
            "org/draft-model", cache_dir="/cache", trust_remote_code=True
        )

    def test_hf_warm_checkpoint_supplies_config_when_not_explicit(self):
        remote = _target_config(layers=1)
        remote.architectures = ["LlamaForCausalLMEagle3"]
        remote.draft_vocab_size = 32
        cfg = _run_config("eagle3", draft_checkpoint_path="org/base-draft")
        with mock.patch(
            "transformers.AutoConfig.from_pretrained", return_value=remote
        ) as load:
            resolved = resolve_draft_config(
                cfg,
                provider=_draft_config_provider("eagle3"),
            )
        self.assertEqual(resolved.architectures, ["LlamaForCausalLMEagle3"])
        load.assert_called_once()
        self.assertEqual(load.call_args.args[0], "org/base-draft")


class _TinyDraft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(4, 3)
        self.proj = torch.nn.Linear(3, 2)


class WarmStartTest(unittest.TestCase):
    def _write_runtime_state(self, directory, state, *, strategy="dflash"):
        path = os.path.join(directory, "training_state.pt")
        torch.save(
            {
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
        Path(directory, "_SUCCESS").touch()
        return path

    def test_markerless_runtime_checkpoint_is_rejected(self):
        source = _TinyDraft()
        destination = _TinyDraft()
        with tempfile.TemporaryDirectory() as directory:
            torch.save(
                {"draft_state_dict": source.state_dict(), "strategy": "dflash"},
                os.path.join(directory, "training_state.pt"),
            )
            with self.assertRaisesRegex(ValueError, "missing _SUCCESS"):
                warm_start_draft_model(
                    destination,
                    os.path.join(directory, "training_state.pt"),
                    draft_config=object(),
                    strategy="dflash",
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
                    draft_config=object(),
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
                draft_config=object(),
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
                    draft_config=object(),
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
                    draft_config=object(),
                    strategy="eagle3",
                )

    def test_pretrained_repo_uses_registered_hf_loader(self):
        source = _TinyDraft()
        destination = _TinyDraft()
        with mock.patch(
            "specforge.modeling.auto.AutoDraftModel.from_pretrained",
            return_value=(source, {"missing_keys": []}),
        ) as load:
            report = warm_start_draft_model(
                destination,
                "org/base-draft",
                draft_config=object(),
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
