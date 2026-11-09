"""Capture contracts use resolved nodes and the same metadata on the producer."""

import unittest
from types import SimpleNamespace
from unittest import mock

from specforge.application import resolve_run
from specforge.config import Config
from specforge.config.draft.codec import encode_config
from specforge.training.capture_contract import (
    ServerCaptureContract,
    resolve_server_capture_contract,
)
from specforge.training.disaggregated import _producer_capture_metadata
from tests.test_config.draft_fixtures import draft_config
from tests.test_modeling.test_canonical_domino import model_config as domino_config
from tests.test_modeling.test_canonical_dspark import model_config as dspark_config


def run_config(draft):
    return Config.model_validate(
        dict(
            draft=encode_config(draft),
            model=dict(target_model_path="target/model", target_backend="sglang"),
            data=dict(train_data_path="train.jsonl"),
            training=dict(role="producer", total_steps=2),
            deployment=dict(
                mode="disaggregated",
                disaggregated=dict(
                    control_dir="outputs/test/control",
                    backend="mooncake",
                    server_urls=["http://capture"],
                ),
            ),
        )
    )


class CaptureContractTest(unittest.TestCase):
    def test_capture_and_producer_share_resolved_structure(self):
        for draft, method in (
            (draft_config(), "dflash"),
            (domino_config(), "dflash"),
            (dspark_config(), "dspark"),
        ):
            with self.subTest(architecture=draft.architecture):
                resolved = resolve_run(run_config(draft))
                with mock.patch(
                    "specforge.modeling.target.target_utils.load_target_config",
                    return_value=SimpleNamespace(hidden_size=32, vocab_size=64),
                ):
                    contract = resolve_server_capture_contract(
                        resolved.config, algorithm=resolved.algorithm
                    )
                expected = ServerCaptureContract(
                    method,
                    tuple(draft.spec_config.target_features.target_layer_ids),
                    32,
                    64,
                    draft.draft_vocab_size,
                )
                self.assertEqual(contract, expected)
                with mock.patch(
                    "specforge.training.capture_contract.resolve_server_capture_contract",
                    return_value=contract,
                ) as capture:
                    metadata = _producer_capture_metadata(
                        resolved.config, resolved.algorithm
                    )
                self.assertEqual(
                    metadata,
                    (list(expected.aux_layer_ids), 32, 64, draft.draft_vocab_size),
                )
                capture.assert_called_once_with(
                    resolved.config, algorithm=resolved.algorithm
                )


if __name__ == "__main__":
    unittest.main()
