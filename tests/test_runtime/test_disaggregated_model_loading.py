# coding=utf-8
"""Disaggregated producers share the canonical draft-config resolver."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import yaml

from specforge.application import resolve_run
from specforge.config import Config
from specforge.config.draft.codec import encode_config
from specforge.training.capture_contract import (
    ServerCaptureContract,
    resolve_server_capture_contract,
)
from specforge.training.disaggregated import _producer_capture_metadata
from tests.test_config.draft_fixtures import draft_config


def _config(*, strategy="dflash", **model_overrides):
    return Config.model_validate(
        {
            "draft": f"specforge/config/draft/presets/qwen3-8b-{strategy}.yaml",
            "model": {
                "target_model_path": "target/model",
                "target_backend": "sglang",
                **model_overrides,
            },
            "data": {"train_data_path": "train.jsonl"},
            "training": {
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
        payload = encode_config(draft_config())
        with TemporaryDirectory() as temp:
            sources = [payload]
            for suffix, serialize in (("json", json.dumps), ("yaml", yaml.safe_dump)):
                path = Path(temp) / f"draft.{suffix}"
                path.write_text(serialize(payload))
                sources.append(str(path))
            for source in sources:
                with self.subTest(source=type(source).__name__):
                    cfg = _config().model_copy(update={"draft": source})
                    resolved = resolve_run(cfg)
                    with mock.patch(
                        "specforge.training.model_loading.training_target_config",
                        return_value={
                            "hidden_size": 32,
                            "vocab_size": 64,
                            "num_hidden_layers": 8,
                        },
                    ):
                        contract = resolve_server_capture_contract(
                            resolved.config,
                            algorithm=resolved.algorithm,
                        )
                    self.assertEqual(contract.method, "dflash")
                    self.assertEqual(
                        contract.aux_layer_ids,
                        draft_config().spec_config.target_features.target_layer_ids,
                    )
                    self.assertEqual(contract.target_hidden_size, 32)

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

    def test_context_families_keep_their_capture_method(self):
        for family, method in (("domino", "dflash"), ("dspark", "dspark")):
            with self.subTest(family=family):
                resolved = resolve_run(_config(strategy=family))
                with mock.patch(
                    "specforge.training.model_loading.training_target_config",
                    return_value={
                        "hidden_size": 4096,
                        "vocab_size": 151936,
                        "num_hidden_layers": 36,
                    },
                ):
                    contract = resolve_server_capture_contract(
                        resolved.config,
                        algorithm=resolved.algorithm,
                    )
                self.assertEqual(contract.method, method)
                self.assertTrue(contract.aux_layer_ids)


if __name__ == "__main__":
    unittest.main()
