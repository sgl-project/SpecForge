"""Public loading boundaries for warm starts, target I/O and legacy provenance."""

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from specforge.checkpoint import save_draft
from specforge.checkpoint.initialization import _TARGET_KEYS, resolve_bindings
from specforge.checkpoint.resolve import target_configuration
from specforge.checkpoint.tensors import TensorStore
from specforge.loading import initialize_draft, load_draft, read_draft
from specforge.modeling import build_draft_model
from tests.test_checkpoint.test_loading import write_target
from tests.test_config.draft_fixtures import draft_config
from tests.test_config.test_draft_legacy_mtp import mtp_payload


class LoadingReviewTest(unittest.TestCase):
    def test_warm_start_selects_only_needed_weights(self):
        config = draft_config(target_bindings=(), target_logit_transform=None)
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            original = build_draft_model(config)
            save_draft(original, source)
            smaller = replace(config, layer_types=config.layer_types[:1])
            with self.assertLogs("specforge.checkpoint.initialization", "WARNING"):
                model = initialize_draft(
                    smaller, weights_from=source, allow_new=["fusion"]
                )
            torch.testing.assert_close(
                model.backbone.layers[0].state_dict(),
                original.backbone.layers[0].state_dict(),
            )
            self.assertFalse(
                torch.equal(model.fusion.fc.weight, original.fusion.fc.weight)
            )
            target = Path(tmp) / "target"
            write_target(target)
            with self.assertLogs("specforge.checkpoint.initialization", "WARNING"):
                model = initialize_draft(
                    draft_config(), weights_from=source, target=target
                )
            self.assertEqual(model.embed_tokens.bound_weight().shape, (64, 32))

    def test_warm_start_does_not_recheck_source_against_new_target(self):
        config = draft_config(target_bindings=(), target_logit_transform=None)
        old = replace(
            config,
            spec_config=replace(
                config.spec_config,
                target_features=replace(
                    config.spec_config.target_features, target_layer_ids=(90, 100)
                ),
            ),
        )
        with TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "source", Path(tmp) / "target"
            save_draft(build_draft_model(old), source)
            write_target(target)
            (target / "config.json").write_text(
                json.dumps(dict(hidden_size=32, vocab_size=64, num_hidden_layers=8))
            )
            initialize_draft(config, target=target, weights_from=source)
            with self.assertRaisesRegex(ValueError, "target_layer_ids"):
                load_draft(source, target=target)

    def test_target_weights_are_opened_only_after_config_checks(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp)
            (target / "config.json").write_text(
                json.dumps(dict(hidden_size=32, vocab_size=64, num_hidden_layers=128))
            )
            with (
                patch(
                    "huggingface_hub.snapshot_download", return_value=tmp
                ) as download,
                patch("specforge.loading.TensorStore") as store,
            ):
                initialize_draft(
                    draft_config(target_bindings=(), target_logit_transform=None),
                    target="test/target",
                )
                self.assertEqual(
                    download.call_args.kwargs["allow_patterns"], ["config.json"]
                )
                store.assert_not_called()
                with self.assertRaisesRegex(ValueError, "hidden_size"):
                    initialize_draft(
                        replace(draft_config(), hidden_size=16), target="test/target"
                    )
                store.assert_not_called()

    def test_fp8_unselected_weights_bind_but_selected_weights_fail(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "target"
            weights = write_target(
                target, {"mlp.weight": torch.ones(32, 32).to(torch.float8_e4m3fn)}
            )
            save_file(
                {
                    k: v if k == "mlp.weight" else v.bfloat16()
                    for k, v in weights.items()
                },
                target / "model.safetensors",
            )
            initialize_draft(draft_config(), target=target)
            with self.assertRaisesRegex(ValueError, "mlp.weight.*quantized"):
                initialize_draft(
                    draft_config(), target=target, from_target={"fusion.fc": "mlp"}
                )

    def test_binding_layouts_and_top_level_tie_flag(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps(
                    {"tie_word_embeddings": True, "text_config": {"hidden_size": 32}}
                )
            )
            config = target_configuration(root)
            for module, keys in _TARGET_KEYS.items():
                for key in keys:
                    with self.subTest(key=key):
                        save_file({key: torch.ones(64, 32)}, root / "model.safetensors")
                        refs = resolve_bindings(
                            replace(
                                draft_config(),
                                target_bindings=(module,),
                                target_logit_transform=(
                                    draft_config().target_logit_transform
                                    if module == "lm_head"
                                    else None
                                ),
                            ),
                            TensorStore(root),
                            config,
                        )
                        self.assertEqual(refs[module].key, key)
            save_file({"embed.weight": torch.ones(64, 32)}, root / "model.safetensors")
            self.assertEqual(
                resolve_bindings(draft_config(), TensorStore(root), config)[
                    "lm_head"
                ].key,
                "embed.weight",
            )

    def test_legacy_contract_and_saved_bound_table_are_checked(self):
        payload = mtp_payload() | dict(hidden_size=32, vocab_size=64)
        with TemporaryDirectory() as tmp:
            target, source = Path(tmp) / "target", Path(tmp) / "source"
            weights = write_target(target)
            source.mkdir()
            (source / "config.json").write_text(json.dumps(payload))
            # read_draft exposes the old bound tables without loading a model.
            save_file(
                {"mtp.lm_head.weight": weights["lm_head.weight"]},
                source / "model.safetensors",
            )
            torch.save({"mtp_share_lm_head": False}, source / "training_state.pt")
            with self.assertRaisesRegex(ValueError, "mtp_share_lm_head"):
                read_draft(source, target=target)
            torch.save(
                {"specforge_model_provenance": (("load_target_embedding", False),)},
                source / "training_state.pt",
            )
            with self.assertRaisesRegex(ValueError, "load_target_embedding"):
                read_draft(source, target=target)
            torch.save({}, source / "training_state.pt")
            checkpoint = read_draft(source, target=target)
            model = build_draft_model(checkpoint.config)
            state = {
                "mtp."
                + k.replace("backbone.final_norm.", "norm.")
                .removeprefix("backbone.")
                .removeprefix("fusion."): v
                for k, v in model.state_dict().items()
            }
            state["mtp.lm_head.weight"] = weights["lm_head.weight"]
            save_file(state, source / "model.safetensors")
            load_draft(source, target=target)
            state["mtp.lm_head.weight"] = state["mtp.lm_head.weight"] + 1
            save_file(state, source / "model.safetensors")
            with self.assertRaisesRegex(ValueError, "mtp.lm_head.weight differs"):
                load_draft(source, target=target)
            with patch(
                "huggingface_hub.snapshot_download", return_value=str(source)
            ) as download:
                read_draft("test/draft", target=target)
                self.assertIn(
                    "training_state.pt", download.call_args.kwargs["allow_patterns"]
                )
