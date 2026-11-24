"""Historical draft checkpoints convert to canonical directories the package reads."""

import argparse
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from scripts.convert_legacy_checkpoint import convert_checkpoint, main
from specforge.checkpoint import save_draft
from specforge.config.draft.codec import decode_config, encode_config
from specforge.export.canonical import read_export
from specforge.loading import initialize_draft, load_draft, read_draft
from specforge.modeling import build_draft_model

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/convert_legacy_checkpoint.py"
# Layouts and expected conversions recorded on the last tree whose package read
# these checkpoints; see the README beside them.
FIXTURES = REPO_ROOT / "tests/fixtures/legacy_checkpoints"
# configuration changes that keep the tensor layout of the checkpoint they vary
VARIANTS = json.loads((FIXTURES / "variants.json").read_text())
CHECKPOINTS = (
    "dflash",
    "dflash_mla",
    "dflash2",
    "domino",
    "dspark",
    "dspark_mha",
    "dspine",
    "eagle3",
    "peagle",
    "mtp",
    "mtp_shared_head",
    *VARIANTS,
)


def _fixture(name):
    if name not in VARIANTS:
        return json.loads((FIXTURES / f"{name}.json").read_text())
    variant = VARIANTS[name]
    fixture = _fixture(variant["base"])
    fixture["config"] |= variant["config"]
    fixture["target_config"] |= variant["target_config"]
    fixture["expected"]["config"] |= variant["expected"]
    return fixture


def _write_target(path, fixture, *, weights=True):
    config = fixture["target_config"]
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    generator = torch.Generator().manual_seed(1)
    shape = (config["vocab_size"], config["hidden_size"])
    tables = {
        "model.embed_tokens.weight": torch.randn(shape, generator=generator),
        "lm_head.weight": torch.randn(shape, generator=generator),
    }
    if weights:
        save_file(tables, path / "model.safetensors")
    return tables


def _legacy_state(fixture, tables, *, dtype=None):
    """The fixture's legacy state dict with drawn floating-point values."""
    generator = torch.Generator().manual_seed(0)
    state = {}
    for key, info in fixture["tensors"].items():
        kind = getattr(torch, info["dtype"])
        if "values" in info:
            value = torch.tensor(info["values"], dtype=kind)
        else:
            value = torch.randn(info["shape"], generator=generator).to(dtype or kind)
        state[key] = value.reshape(info["shape"])
    # a legacy model saved its target tables as it held them: muP already folded
    multiplier = fixture["target_config"].get("logits_mup_width_multiplier", 1)
    for module, key in fixture["expected"]["bound_copies"].items():
        if module == "lm_head":
            state[key] = tables["lm_head.weight"] / multiplier
        else:
            state[key] = tables["model.embed_tokens.weight"].clone()
    return state


def _snapshot(directory):
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


class _Case:
    """One fixture written as a historical checkpoint beside its target."""

    def __init__(self, root, name, *, dtype=None, target_weights=True):
        self.root = Path(root)
        self.fixture = _fixture(name)
        self.payload = self.fixture["config"]
        self.settings = self.fixture["settings"]
        self.expected = self.fixture["expected"]
        self.target = self.root / "target"
        self.tables = _write_target(self.target, self.fixture, weights=target_weights)
        self.state = _legacy_state(self.fixture, self.tables, dtype=dtype)
        self.source = self.root / "legacy"
        self.source.mkdir()
        self.output = self.root / "converted"

    def write(self, *, config=True, weights="model.safetensors", state=None):
        state = self.state if state is None else state
        if config:
            (self.source / "config.json").write_text(json.dumps(self.payload))
        if weights.endswith(".safetensors"):
            save_file(state, self.source / weights)
        else:
            torch.save(state, self.source / weights)
        return self

    def convert(self, source=None, **options):
        convert_checkpoint(
            self.source if source is None else source,
            self.output,
            **{"target_model": self.target, **self.settings, **options},
        )


class ConvertLegacyCheckpointTest(unittest.TestCase):
    def assert_converted(self, case, *, names=None):
        """The output holds the expected configuration and every source tensor."""
        names = case.expected["names"] if names is None else names
        payload = json.loads((case.output / "config.json").read_text())
        self.assertEqual(payload["model_type"], "specforge_draft")
        self.assertEqual(
            payload["specforge_artifact"], {"format": "specforge", "version": 1}
        )
        config = read_draft(case.output, target=case.target).config
        self.assertEqual(encode_config(config), case.expected["config"])
        self.assertEqual(config, decode_config(case.expected["config"]))
        converted = load_file(case.output / "model.safetensors")
        self.assertEqual(sorted(converted), sorted(names.values()))
        # only the copies of target-bound tables stay behind
        dropped = case.expected["bound_copies"].values()
        self.assertEqual(sorted(case.state), sorted([*names, *dropped]))
        for legacy, canonical in names.items():
            self.assertEqual(converted[canonical].dtype, case.state[legacy].dtype)
            self.assertTrue(
                torch.equal(converted[canonical], case.state[legacy]), canonical
            )

    def test_every_architecture_converts_as_the_package_once_read_it(self):
        self.assertEqual(
            sorted(set(CHECKPOINTS) - set(VARIANTS)),
            sorted(p.stem for p in FIXTURES.glob("*.json") if p.stem != "variants"),
        )
        for name in CHECKPOINTS:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                case = _Case(tmp, name).write()
                before = _snapshot(case.source)
                case.convert()
                self.assertEqual(_snapshot(case.source), before)
                self.assertEqual(
                    sorted(path.name for path in case.output.iterdir()),
                    ["config.json", "model.safetensors"],
                )
                self.assert_converted(case)
                # weight import (initialization.checkpoint) and export read it
                model = load_draft(case.output, target=case.target)
                state = model.state_dict()
                for legacy, canonical in case.expected["names"].items():
                    self.assertTrue(
                        torch.equal(state[canonical], case.state[legacy]), canonical
                    )
                checkpoint, exported, _ = read_export(case.output, target=case.target)
                self.assertEqual(checkpoint.config, model.config.draft)
                self.assertEqual(sorted(exported), sorted(state))

    def test_weight_files_training_states_and_separate_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = _Case(tmp, "dflash", dtype=torch.bfloat16)
            case.write(weights="pytorch_model.bin").convert()
            self.assert_converted(case)
            payload = json.loads((case.output / "config.json").read_text())
            self.assertEqual(payload["dtype"], "bfloat16")
        # a run directory of the old trainer: weights and records, no config.json
        with tempfile.TemporaryDirectory() as tmp:
            case = _Case(tmp, "dflash2")
            config_path = case.root / "old.json"
            config_path.write_text(json.dumps(case.payload))
            method = case.payload["dflash_config"]
            records = {
                "draft_state_dict": case.state,
                "step": 42,
                "dflash_block_size": case.payload["block_size"],
                "dflash_mask_token_id": method["mask_token_id"],
                "dflash_target_layer_ids": method["target_layer_ids"],
            }
            step = case.source / "run-step42"
            step.mkdir()
            torch.save(records, step / "training_state.pt")
            with self.assertRaisesRegex(ValueError, "--draft-config"):
                case.convert()
            case.convert(draft_config=config_path)
            self.assert_converted(case)
            # the supplied configuration must be the one the run recorded
            torch.save(
                records | {"dflash_block_size": case.payload["block_size"] + 1},
                step / "training_state.pt",
            )
            case.output = case.root / "stale"
            with self.assertRaisesRegex(ValueError, "dflash_block_size"):
                case.convert(draft_config=config_path)
        with tempfile.TemporaryDirectory() as tmp:
            case = _Case(tmp, "dspine").write(config=False, weights="weights.bin")
            case.convert(case.source / "weights.bin", draft_config=case.payload)
            self.assert_converted(case)
            case.write()
            case.output = case.root / "twice"
            with self.assertRaisesRegex(ValueError, "omit --draft-config"):
                case.convert(draft_config=case.payload)
        # scripts older than that trainer pickled their arguments beside the
        # weights; such a state records no settings
        with tempfile.TemporaryDirectory() as tmp:
            case = _Case(tmp, "eagle3").write()
            state = {"args": argparse.Namespace(lr=1e-4), "epoch": 3}
            torch.save(state, case.source / "training_state.pt")
            with self.assertLogs("scripts.legacy_checkpoint.reader", "WARNING"):
                case.convert()
            self.assert_converted(case)

    def test_projector_checkpoints_saved_as_dflash_with_logit_head_names(self):
        heads = ("prefix_gru.", "embed_proj.", "markov_head.", "confidence_head.")
        for name in ("domino", "dspark"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                case = _Case(tmp, name)
                rename = {
                    key: "logit_head." + key if key.startswith(heads) else key
                    for key in case.state
                }
                self.assertNotEqual(list(rename), list(rename.values()))
                case.payload = case.payload | {"architectures": ["DFlashDraftModel"]}
                case.state = {rename[key]: value for key, value in case.state.items()}
                case.write().convert()
                names = {
                    rename[key]: canonical
                    for key, canonical in case.expected["names"].items()
                }
                self.assert_converted(case, names=names)

    def test_recipe_settings_resolve_what_the_old_configuration_omits(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = _Case(tmp, "peagle").write()
            case.settings = {}
            with self.assertRaisesRegex(ValueError, "--mask-token-id"):
                case.convert()
            torch.save({"peagle_mask_token_id": 63}, case.source / "training_state.pt")
            case.convert()
            self.assert_converted(case)
            case.output = case.root / "conflict"
            with self.assertRaisesRegex(ValueError, "explicit mask_token_id conflicts"):
                case.convert(mask_token_id=62)
            case.convert(
                mask_token_id=63, norm_before_residual=True, capture_layers=[6, 2, 4]
            )
            spec = read_draft(case.output, target=case.target).config.spec_config
            self.assertEqual(spec.mask_token_id, 63)
            self.assertIs(spec.norm_before_residual, True)
            self.assertEqual(spec.target_features.target_layer_ids, (2, 4, 6))
        with tempfile.TemporaryDirectory() as tmp:
            case = _Case(tmp, "eagle3").write()
            torch.save(
                {"specforge_model_provenance": (("capture_layers", (2, 4, 6)),)},
                case.source / "training_state.pt",
            )
            with self.assertRaisesRegex(ValueError, "explicit capture_layers"):
                case.convert(capture_layers=[1, 2, 3])
            case.convert(capture_layers=[2, 4, 6])
            features = read_draft(case.output).config.spec_config.target_features
            self.assertEqual(features.target_layer_ids, (2, 4, 6))
        # a binding the run recorded cannot differ from the supplied configuration
        for name, records, error in (
            ("mtp_shared_head", {"mtp_share_lm_head": False}, "mtp_share_lm_head"),
            (
                "eagle3",
                {"specforge_model_provenance": (("load_target_embedding", False),)},
                "load_target_embedding",
            ),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                case = _Case(tmp, name).write()
                torch.save(records, case.source / "training_state.pt")
                with self.assertRaisesRegex(ValueError, f"recorded {error}=False"):
                    case.convert()
        # the settings belong to the EAGLE family only; DFlash configs carry theirs
        for setting, error in (
            ({"mask_token_id": 63}, "only supported for legacy P-EAGLE"),
            ({"norm_before_residual": False}, "only supported for legacy P-EAGLE"),
            ({"capture_layers": [1, 2, 3]}, "only supported for legacy EAGLE"),
        ):
            with self.subTest(setting=setting), tempfile.TemporaryDirectory() as tmp:
                case = _Case(tmp, "dflash").write()
                with self.assertRaisesRegex(ValueError, error):
                    case.convert(**setting)
                self.assertFalse(case.output.exists())

    def test_saved_target_tables_must_equal_the_target(self):
        for name, key in (
            ("eagle3", "embed_tokens.weight"),
            ("mtp_shared_head", "mtp.lm_head.weight"),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                case = _Case(tmp, name)
                trained = dict(case.state)
                trained[key] = trained[key].clone()
                trained[key][0, 0] += 1
                case.write(state=trained)
                with self.assertRaisesRegex(ValueError, f"saved {key} differs"):
                    case.convert()
                self.assertFalse(case.output.exists())
        # without a saved copy, the target's configuration alone is read
        with tempfile.TemporaryDirectory() as tmp:
            case = _Case(tmp, "dflash", target_weights=False).write()
            case.convert()
            self.assertEqual(
                sorted(load_file(case.output / "model.safetensors")),
                sorted(case.expected["names"].values()),
            )
        with tempfile.TemporaryDirectory() as tmp:
            case = _Case(tmp, "eagle3", target_weights=False).write()
            with self.assertRaises(FileNotFoundError):
                case.convert()

    def test_rejected_inputs_raise_named_errors_and_write_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = _Case(tmp, "dflash").write()
            (case.root / "existing").mkdir()
            listing = sorted(path.name for path in case.root.rglob("*"))

            def rejects(error, message, **options):
                with self.assertRaisesRegex(error, message):
                    case.convert(**options)
                self.assertEqual(
                    sorted(path.name for path in case.root.rglob("*")), listing
                )

            # even an empty directory: the output is always a new one
            case.output = case.root / "existing"
            rejects(FileExistsError, "existing")
            case.output = case.source / "converted"
            rejects(ValueError, "outside the source checkpoint")
            case.output = case.root / "converted"
            rejects(ValueError, "target model is required", target_model=None)
            rejects(FileNotFoundError, "missing", target_model=case.root / "missing")
            incomplete = {k: v for k, v in case.state.items() if k != "fc.weight"}
            case.write(state=incomplete)
            rejects(ValueError, "missing draft tensors.*fusion.fc.weight")
            case.write(state=case.state | {"extra.weight": torch.zeros(1)})
            rejects(ValueError, "unexpected draft tensors.*extra.weight")
            case.payload = case.payload | {"architectures": ["GPT2LMHeadModel"]}
            case.write()
            rejects(ValueError, "unsupported legacy architectures")
        with tempfile.TemporaryDirectory() as tmp:
            case = _Case(tmp, "dflash")
            config = decode_config(case.expected["config"])
            save_draft(build_draft_model(config), case.source)
            with self.assertRaisesRegex(ValueError, "already canonical"):
                case.convert()
            self.assertFalse(case.output.exists())

    def test_failed_write_leaves_nothing_and_can_be_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = _Case(tmp, "dflash").write()
            listing = sorted(path.name for path in case.root.iterdir())
            with (
                patch(
                    "specforge.checkpoint.draft_dir.save_file",
                    side_effect=OSError("disk full"),
                ),
                self.assertRaisesRegex(OSError, "disk full"),
            ):
                case.convert()
            self.assertEqual(sorted(path.name for path in case.root.iterdir()), listing)
            case.convert()
            self.assert_converted(case)

    def test_command_line_and_standalone_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = _Case(tmp, "peagle").write()
            argv = ["convert_legacy_checkpoint.py", "--checkpoint", str(case.source)]
            argv += ["--output-dir", str(case.output), "--target-model"]
            argv += [str(case.target), "--mask-token-id", "63"]
            with patch.object(sys, "argv", argv):
                main()
            self.assert_converted(case)
            case.output = case.root / "settings"
            argv[4] = str(case.output)
            argv += ["--norm-before-residual", "--capture-layers", "6", "2", "4"]
            with patch.object(sys, "argv", argv):
                main()
            spec = read_draft(case.output).config.spec_config
            self.assertIs(spec.norm_before_residual, True)
            self.assertEqual(spec.target_features.target_layer_ids, (2, 4, 6))
        # `python scripts/convert_legacy_checkpoint.py` has scripts/, not the
        # repository root, on sys.path
        hidden = {
            name: None for name in sys.modules if name.startswith("scripts.legacy")
        } | {"scripts.legacy_checkpoint": None}
        with (
            patch.dict(sys.modules, hidden),
            patch.object(sys, "path", [str(SCRIPT.parent), *sys.path]),
        ):
            standalone = runpy.run_path(str(SCRIPT))
            self.assertEqual(
                standalone["read_legacy"].__module__, "legacy_checkpoint.reader"
            )

    def test_package_reader_names_the_script_for_historical_checkpoints(self):
        script = "scripts/convert_legacy_checkpoint.py"
        with tempfile.TemporaryDirectory() as tmp:
            case = _Case(tmp, "dflash").write()
            config = decode_config(case.expected["config"])
            for read in (
                lambda: read_draft(case.source, target=case.target),
                lambda: load_draft(case.source, target=case.target),
                lambda: read_export(case.source, target=case.target),
                lambda: initialize_draft(
                    config, target=case.target, weights_from=case.source
                ),
            ):
                with self.assertRaisesRegex(ValueError, f"qwen3.*{script}"):
                    read()
            (case.source / "config.json").unlink()
            with self.assertRaisesRegex(ValueError, f"{script}.*--draft-config"):
                read_draft(case.source, target=case.target)
            with self.assertRaisesRegex(ValueError, f"{script}.*--draft-config"):
                read_draft(case.source / "model.safetensors", target=case.target)
            # a directory without weights is a wrong path, not a historical checkpoint
            (case.source / "model.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "has no config.json") as raised:
                read_draft(case.source)
            self.assertNotIn(script, str(raised.exception))
            case.write().convert()
            self.assertEqual(read_draft(case.output, target=case.target).config, config)


if __name__ == "__main__":
    unittest.main()
