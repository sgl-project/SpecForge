# coding=utf-8
"""Disaggregated producers share the canonical draft-config resolver."""

import unittest
from types import SimpleNamespace
from unittest import mock

from specforge.application import resolve_run
from specforge.config import Config
from specforge.training.capture_contract import (
    ServerCaptureContract,
    resolve_server_capture_contract,
)
from specforge.training.disaggregated import _producer_capture_metadata


def _config(*, strategy="dflash", **model_overrides):
    return Config.model_validate(
        {
            "model": {
                "target_model_path": "target/model",
                "target_backend": "sglang",
                **model_overrides,
            },
            "data": {"train_data_path": "train.jsonl"},
            "training": {
                "strategy": strategy,
                "role": "producer",
                "total_steps": 2,
            },
            "deployment": {
                "mode": "disaggregated",
                "disaggregated": {
                    "control_dir": "outputs/test/control",
                    "backend": "mooncake",
                    "server_urls": ["http://capture"],
                },
            },
        }
    )


class DisaggregatedModelLoadingTest(unittest.TestCase):
    def test_capture_contract_uses_canonical_draft_config_sources(self):
        configs = (
            _config(),
            _config(draft_model_config="org/dflash-config"),
            _config(draft_checkpoint_path="org/dflash-weights"),
        )
        draft_payload = {
            "architectures": ["DFlashDraftModel"],
            "vocab_size": 128,
            "draft_vocab_size": 32,
            "dflash_config": {"target_layer_ids": [3, 7]},
        }
        target_config = SimpleNamespace(hidden_size=64, vocab_size=128)

        for cfg in configs:
            resolved = resolve_run(cfg)
            with (
                self.subTest(model=cfg.model.model_dump()),
                mock.patch(
                    "transformers.AutoConfig.from_pretrained",
                    return_value=target_config,
                ),
                mock.patch(
                    "specforge.training.model_loading.draft_config_dict",
                    return_value=draft_payload,
                ) as resolve,
            ):
                contract = resolve_server_capture_contract(
                    resolved.config,
                    algorithm=resolved.algorithm,
                )

            self.assertEqual(
                contract,
                ServerCaptureContract("dflash", (3, 7), 64, 128, 32),
            )
            resolve.assert_called_once_with(
                resolved.config,
                provider=resolved.algorithm.providers.model.draft_config,
            )

    def test_producer_metadata_delegates_to_the_shared_server_contract(self):
        resolved = resolve_run(_config())
        contract = ServerCaptureContract(
            method="dflash",
            aux_layer_ids=(3, 7),
            target_hidden_size=64,
            target_vocab_size=128,
            draft_vocab_size=32,
        )
        with mock.patch(
            "specforge.training.capture_contract.resolve_server_capture_contract",
            return_value=contract,
        ) as resolve:
            metadata = _producer_capture_metadata(
                resolved.config,
                resolved.algorithm,
            )
        self.assertEqual(metadata, ([3, 7], 64, 128, 32))
        resolve.assert_called_once_with(
            resolved.config,
            algorithm=resolved.algorithm,
        )

    def test_domino_uses_the_dflash_server_capture_method(self):
        resolved = resolve_run(
            _config(strategy="domino", draft_model_config="domino-draft")
        )
        draft_payload = {
            "architectures": ["DominoDraftModel"],
            "vocab_size": 128,
            "dflash_config": {
                "projector_type": "domino",
                "target_layer_ids": [3, 7],
            },
        }
        with (
            mock.patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=SimpleNamespace(hidden_size=64, vocab_size=128),
            ),
            mock.patch(
                "specforge.training.model_loading.draft_config_dict",
                return_value=draft_payload,
            ),
        ):
            contract = resolve_server_capture_contract(
                resolved.config,
                algorithm=resolved.algorithm,
            )

        self.assertEqual(contract.method, "dflash")
        self.assertEqual(contract.aux_layer_ids, (3, 7))


if __name__ == "__main__":
    unittest.main(verbosity=2)
