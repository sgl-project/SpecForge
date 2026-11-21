"""Old canonical records convert without changing weights, optimizer state or shards."""

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.convert_canonical_checkpoint import convert_checkpoint, convert_record
from specforge.algorithms.objectives.registry import objective_record, resolve_objective
from tests.test_runtime import test_checkpoint_resume as resume_tests

# (architecture, objective, old objective options, old flat loss as the old code
# recorded it: always resolved, never "default", expected named terms)
CASES = (
    ("DFlashDraftModel", "dflash", {}, {"kind": "ce", "options": {}}, {}),
    (
        "DFlash2DraftModel",
        "dflash2",
        {"selector_loss_alpha": 0.4},
        {"kind": "ce", "options": {}},
        {"selector": {"coefficient": 0.4}},
    ),
    (
        "DSparkDraftModel",
        "dspark",
        {"ce_loss_alpha": 0.2, "l1_loss_alpha": 0.8, "confidence_loss_alpha": 0.3},
        {"kind": "ce_l1", "options": {"ce_weight": 0.2, "l1_weight": 0.8}},
        {
            "token": {"options": {"ce_weight": 0.2, "l1_weight": 0.8}},
            "confidence": {"coefficient": 0.3},
        },
    ),
    (
        "DominoDraftModel",
        "domino",
        {},
        {"kind": "kl", "options": {}},
        {"base": {"kind": "kl"}, "final": {"kind": "kl"}},
    ),
    (
        "DSpineDraftModel",
        "dspine",
        {"ce_weight": 0.7, "alignment_weight": 0.25},
        {"kind": "ce_l1", "options": {"ce_weight": 0.7, "l1_weight": 1 - 0.7}},
        {
            "token": {"options": {"ce_weight": 0.7, "l1_weight": 1 - 0.7}},
            "refinement": {"options": {"ce_weight": 0.7, "l1_weight": 1 - 0.7}},
            "alignment": {"coefficient": 0.25},
        },
    ),
    # an explicit flat ce_l1 kept the formula's own weights; ce_weight reached refinement only
    (
        "DSpineDraftModel",
        "dspine",
        {"ce_weight": 0.7},
        {"kind": "ce_l1", "options": {}},
        {
            "token": {"options": {"ce_weight": 0.1, "l1_weight": 0.9}},
            "refinement": {"options": {"ce_weight": 0.7, "l1_weight": 1 - 0.7}},
        },
    ),
    ("MTPDraftModel", "mtp", {}, {"kind": "ce", "options": {}}, {}),
)


class ConvertCanonicalCheckpointTest(unittest.TestCase):
    def test_old_records_convert_and_resume(self):
        harness = resume_tests.TestTrainerResumeEntrypoint()
        if not torch.distributed.is_initialized():
            harness.setUpClass()
            self.addCleanup(harness.tearDownClass)
        for architecture, name, old_options, flat, terms in CASES:
            for has_execution in (False, True):
                with (
                    self.subTest(
                        architecture=architecture,
                        flat=flat,
                        has_execution=has_execution,
                    ),
                    tempfile.TemporaryDirectory() as temp,
                ):
                    root = Path(temp)
                    features = resume_tests._write_feature_files(
                        str(root / "features"), n=4
                    )
                    defaults = resolve_objective(architecture)
                    options = {**asdict(defaults.objective), **old_options}
                    old = {
                        "canonical_draft_config": {"architectures": [architecture]},
                        "canonical_objective": name,
                        "canonical_objective_options": options,
                        "canonical_loss": flat,
                    }
                    if has_execution:
                        old.update(
                            # the old default: reference formulas
                            canonical_loss_execution=asdict(defaults.execution)
                            | {"fused": False},
                            canonical_loss_implementations={},
                            canonical_use_fused_conv=False,
                        )
                    trainer, _, _ = harness._make_trainer(
                        str(root / "run"),
                        feat_dir=features,
                        max_steps=1,
                        checkpoint_extra=old,
                    )
                    trainer.fit()
                    source = (root / "run" / "rz-latest").resolve()
                    # These files must remain byte-for-byte unchanged.
                    (source / "config.json").write_text('{"fixture": true}')
                    before = {
                        p.name: p.read_bytes() for p in source.iterdir() if p.is_file()
                    }
                    output = root / "converted"
                    convert_checkpoint(source, output)
                    self.assertEqual(
                        before,
                        {
                            p.name: p.read_bytes()
                            for p in source.iterdir()
                            if p.is_file()
                        },
                    )
                    for filename in ("config.json", "training_state_rank0.pt"):
                        self.assertEqual(
                            (output / filename).read_bytes(), before[filename]
                        )
                    # An old run evaluated the reference; the resumed one says so.
                    expected = resolve_objective(
                        architecture, loss={"terms": terms, "fused": False}
                    )
                    saved = torch.load(output / "training_state.pt", weights_only=True)
                    self.assertEqual(saved["canonical_loss"], expected.losses.to_dict())
                    self.assertEqual(
                        saved["canonical_objective_options"],
                        objective_record(name, asdict(expected.objective)),
                    )
                    original = torch.load(
                        source / "training_state.pt", weights_only=True
                    )
                    rewritten = {
                        "canonical_loss",
                        "canonical_objective_options",
                        "canonical_loss_execution",
                        "canonical_loss_implementations",
                        "canonical_use_fused_conv",
                    }
                    from torch.utils._pytree import tree_flatten

                    for key in original.keys() - rewritten:
                        actual, actual_structure = tree_flatten(saved[key])
                        expected_values, structure = tree_flatten(original[key])
                        self.assertEqual(actual_structure, structure)
                        for value, expected_value in zip(actual, expected_values):
                            if isinstance(value, torch.Tensor):
                                torch.testing.assert_close(
                                    value, expected_value, rtol=0, atol=0
                                )
                            else:
                                self.assertEqual(value, expected_value)
                    contract = {
                        **old,
                        "canonical_loss": expected.losses.to_dict(),
                        "canonical_objective_options": objective_record(
                            name, asdict(expected.objective)
                        ),
                        "canonical_loss_execution": asdict(expected.execution),
                        "canonical_use_fused_conv": False,
                    }
                    # Old records fail clearly; converted records enter the actual Trainer.
                    with self.assertRaisesRegex(
                        ValueError, "convert_canonical_checkpoint"
                    ):
                        harness._make_trainer(
                            str(root / "run"),
                            feat_dir=features,
                            max_steps=2,
                            checkpoint_extra=contract,
                            resume_from=str(source),
                        )
                    from specforge.algorithms.objectives.implementations.registry import (
                        ExecutionSelection,
                    )

                    composite, _, _ = resume_tests._fake_seam()
                    model = composite()
                    model.loss_implementations = ExecutionSelection()
                    resumed, _, _ = harness._make_trainer(
                        str(root / "run"),
                        feat_dir=features,
                        max_steps=2,
                        checkpoint_extra=contract,
                        resume_from=str(output),
                        model=model,
                    )
                    resumed.fit()

    def test_staging_directory_must_not_contain_source(self):
        architecture, name, old_options, flat, _ = CASES[4]
        old = {
            "canonical_draft_config": {"architectures": [architecture]},
            "canonical_objective": name,
            "canonical_objective_options": {
                **asdict(resolve_objective(architecture).objective),
                **old_options,
            },
            "canonical_loss": flat,
        }
        for nested in (False, True):
            with self.subTest(nested=nested), tempfile.TemporaryDirectory() as temp:
                output = Path(temp) / "converted"
                staging = Path(temp) / "converted.converting"
                source = staging / "source" if nested else staging
                source.mkdir(parents=True)
                torch.save(old, source / "training_state.pt")
                (source / "config.json").write_text('{"fixture": true}')
                (source / "training_state_rank0.pt").write_bytes(b"shard")
                if nested:
                    (staging / "sibling").write_bytes(b"preserve me")
                before = {
                    p.relative_to(staging): p.read_bytes()
                    for p in staging.rglob("*")
                    if p.is_file()
                }
                with self.assertRaisesRegex(ValueError, "Staging directory"):
                    convert_checkpoint(source, output)
                self.assertFalse(output.exists())
                self.assertEqual(
                    before,
                    {
                        p.relative_to(staging): p.read_bytes()
                        for p in staging.rglob("*")
                        if p.is_file()
                    },
                )

    def test_fused_dspine_identities_and_invalid_destinations(self):
        resolved = resolve_objective("DSpineDraftModel")
        old = {
            "canonical_draft_config": {"architectures": ["DSpineDraftModel"]},
            "canonical_objective": "dspine",
            "canonical_objective_options": {
                **asdict(resolved.objective),
                "ce_weight": 0.1,
            },
            "canonical_loss": {
                "kind": "ce_l1",
                "options": {"ce_weight": 0.1, "l1_weight": 0.9},
            },
            "canonical_loss_execution": asdict(resolved.execution) | {"fused": True},
            "canonical_loss_implementations": {
                "0/compute/refinement:0": "token_ce",
                "0/compute/candidates:0": "token_ce",
            },
        }
        result = convert_record(old)
        self.assertEqual(
            result["canonical_loss_implementations"],
            {
                "0/compute/refinement:0": "token_ce_l1",
                "0/compute/alignment:0": "reference",
                "0/compute/candidates:0": "token_ce",
            },
        )
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "outside"):
                convert_checkpoint(temp, Path(temp) / "converted")
            with self.assertRaises(FileExistsError):
                convert_checkpoint(Path(temp) / "source", temp)
        # a legacy-wrapper checkpoint has no canonical record to convert
        with self.assertRaisesRegex(ValueError, "Not a canonical training checkpoint"):
            convert_record({"step": 1})
        # a failed conversion leaves nothing behind, so the retry is not blocked
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "source", Path(temp) / "converted"
            source.mkdir()
            torch.save(old, source / "training_state.pt")
            with (
                patch("torch.save", side_effect=OSError("disk full")),
                self.assertRaisesRegex(OSError, "disk full"),
            ):
                convert_checkpoint(source, output)
            self.assertEqual([p.name for p in Path(temp).iterdir()], ["source"])
            convert_checkpoint(source, output)
            saved = torch.load(output / "training_state.pt", weights_only=True)
            self.assertIn("terms", saved["canonical_loss"])
