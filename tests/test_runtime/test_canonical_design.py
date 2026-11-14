"""Canonical composition must not depend on legacy providers or HF defaults."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from specforge.algorithms.objectives import (
    TrainingRequirements,
    register_protocol,
    registry,
)
from specforge.algorithms.objectives.protocols.dflash import DFlashObjective
from specforge.application.composition import resolve_run
from specforge.checkpoint.resolve import read_target_config
from specforge.loading import initialize_draft
from specforge.training.assembly import build_model_bundle
from specforge.training.capture_contract import resolve_server_capture_contract
from specforge.training.model_loading import training_target_config
from tests.test_runtime.test_canonical_training import (
    draft_config,
    run_config,
    target_directory,
)


class CanonicalDesignTest(unittest.TestCase):
    def test_custom_objective_does_not_resolve_legacy_registry(self):
        def supervised_last_token(mask):
            return bool(mask[-1])

        with patch.dict(registry._PROTOCOLS), patch.dict(registry._DEFAULTS):
            register_protocol(
                "custom",
                architectures=("DFlashDraftModel",),
                inputs=("input_ids", "hidden_states", "loss_mask"),
                provides=("logits", "labels", "chain_positions"),
                trains=("backbone", "fusion"),
                training=TrainingRequirements(
                    capture_method="dflash",
                    attention_backends=frozenset({"sdpa"}),
                    loss_mask_filter=supervised_last_token,
                ),
                default=False,
            )(DFlashObjective)
            cfg = run_config(
                Path("target"),
                draft_config(),
                training={"attention_backend": "sdpa", "objective": "custom"},
            )
            with patch(
                "specforge.algorithms.builtin.builtin_algorithm_registry",
                side_effect=AssertionError("legacy registry used"),
            ):
                resolved = resolve_run(cfg)
            self.assertEqual(resolved.algorithm.name, "custom")
            self.assertEqual(
                resolved.algorithm.spec.capabilities.attention_backends,
                frozenset({"sdpa"}),
            )
            self.assertIs(
                resolved.algorithm.providers.model.loss_mask_filter,
                supervised_last_token,
            )
            self.assertTrue(resolved.algorithm.providers.model.loss_mask_filter([0, 1]))
            cfg.training.attention_backend = "eager"
            with self.assertRaisesRegex(ValueError, "attention_backend"):
                resolve_run(cfg)
            self.assertEqual(
                resolved.algorithm.providers.server_streaming_for(
                    "text"
                ).capture_method,
                "dflash",
            )

    def test_worker_defaults_and_legacy_identifiers(self):
        from specforge.training.assembly import _dataloader_num_workers
        from tests.test_algorithms.test_objectives import configurations

        for draft in configurations():
            with self.subTest(architecture=draft.architecture):
                cfg = run_config(Path("target"), draft)
                cfg.data.dataloader_num_workers = None
                run = resolve_run(cfg)
                self.assertEqual(_dataloader_num_workers(run.config, run.algorithm), 4)
                for workers in (0, 8):
                    run.config.data.dataloader_num_workers = workers
                    self.assertEqual(
                        _dataloader_num_workers(run.config, run.algorithm), workers
                    )
                if draft.architecture in ("DFlash2DraftModel", "DMTPDraftModel"):
                    self.assertEqual(run.algorithm.name, "dflash")
                    self.assertEqual(run.config.training.strategy, "dflash")

    def test_target_metadata_is_raw_and_shared_with_capture(self):
        with TemporaryDirectory() as temp:
            target = target_directory(Path(temp))
            metadata = json.loads((target / "config.json").read_text())
            metadata["model_type"] = "unregistered_target"
            (target / "config.json").write_text(json.dumps({"text_config": metadata}))
            cfg = run_config(target, draft_config())
            self.assertEqual(training_target_config(cfg), read_target_config(target))
            with patch(
                "transformers.AutoConfig.from_pretrained",
                side_effect=AssertionError("HF defaults used"),
            ):
                run = resolve_run(cfg, target_config=training_target_config(cfg))
                bundle = build_model_bundle(run.config, algorithm=run.algorithm)
                capture = resolve_server_capture_contract(
                    run.config, algorithm=run.algorithm
                )
            self.assertEqual(capture.target_hidden_size, bundle.target_hidden_size)
            self.assertEqual(capture.aux_layer_ids, tuple(bundle.capture_layers))

    def test_frozen_parameters_require_actual_sources(self):
        with TemporaryDirectory() as temp:
            target = target_directory(Path(temp))
            cfg = run_config(
                target,
                draft_config(),
                training={"attention_backend": "eager", "frozen": ["fusion"]},
            )
            run = resolve_run(cfg)
            with self.assertRaisesRegex(
                ValueError, "Frozen parameters require loaded weights"
            ):
                build_model_bundle(run.config, algorithm=run.algorithm)
            # Binding is a source even though it is absent from state_dict.
            initialize_draft(draft_config(), target=target, require_loaded=("lm_head",))
            original = initialize_draft(draft_config(), target=target)
            checkpoint = Path(temp) / "draft"
            original.save_pretrained(checkpoint)
            restored = initialize_draft(
                draft_config(),
                target=target,
                weights_from=checkpoint,
                require_loaded=("fusion",),
            )
            torch.testing.assert_close(
                restored.fusion.state_dict(), original.fusion.state_dict()
            )
            missing = tuple(
                key for key in original.state_dict() if key.startswith("fusion.")
            )
            with self.assertRaisesRegex(
                ValueError, "Frozen parameters require loaded weights"
            ):
                initialize_draft(
                    draft_config(),
                    target=target,
                    weights_from=checkpoint,
                    mapping={
                        key: key for key in original.state_dict() if key not in missing
                    },
                    allow_new=missing,
                    require_loaded=("fusion",),
                )

    def test_objective_frozen_parameters_are_checked(self):
        from tests.test_modeling.test_canonical_dflash2 import dflash2_config

        with TemporaryDirectory() as temp:
            target = target_directory(Path(temp))
            cfg = run_config(
                target,
                dflash2_config(),
                training={
                    "attention_backend": "eager",
                    "loss": {"terms": {"selector": {"coefficient": 0}}},
                },
            )
            run = resolve_run(cfg)
            with self.assertRaisesRegex(
                ValueError,
                "Frozen parameters require loaded weights.*candidate_selector",
            ):
                build_model_bundle(run.config, algorithm=run.algorithm)
