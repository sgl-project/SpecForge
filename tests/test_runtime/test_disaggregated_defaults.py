"""Disaggregated workers must retain defaults and explicit receive overrides."""

import unittest
from unittest import mock

from specforge.config import Config
from specforge.runtime.data_plane.mooncake_store import _InjectedReplicateConfig
from specforge.training.disaggregated import _mooncake_store
from tests.test_runtime.test_mooncake_store import _FakeMooncakeStore


class DisaggregatedDefaultsTest(unittest.TestCase):
    def test_worker_store_uses_config_unless_environment_overrides_it(self):
        for configured, override, expected in (
            (None, None, "pinned"),
            ("pageable", None, "pageable"),
            ("pageable", "pinned", "pinned"),
            (None, "pageable", "pageable"),
        ):
            with self.subTest(configured=configured, override=override):
                deployment = {
                    "control_dir": "/control",
                    "backend": "mooncake",
                    "server_urls": ["http://capture:30000"],
                }
                if configured is not None:
                    deployment["receive_buffers"] = configured
                cfg = Config.model_validate(
                    {
                        "model": {"target_model_path": "target"},
                        "data": {"prompts_path": "prompts.jsonl"},
                        "deployment": {
                            "mode": "disaggregated",
                            "disaggregated": deployment,
                        },
                    }
                )
                self._assert_worker_mode(cfg, override, expected)

    def test_legacy_environment_worker_uses_pinned_by_default(self):
        cfg = Config.model_validate(
            {
                "model": {"target_model_path": "target"},
                "data": {"hidden_states_path": "/features"},
            }
        )
        self._assert_worker_mode(cfg, None, "pinned")

    def _assert_worker_mode(self, cfg, override, expected):
        env = {
            "MOONCAKE_METADATA_SERVER": "http://metadata:8080/metadata",
            "MOONCAKE_MASTER_SERVER_ADDR": "master:50051",
        }
        if override is not None:
            env["DISAGG_RECEIVE_BUFFERS"] = override
        with (
            mock.patch.dict("os.environ", env, clear=True),
            mock.patch(
                "specforge.runtime.data_plane.mooncake_store._connect_store",
                return_value=(_FakeMooncakeStore(), _InjectedReplicateConfig),
            ),
        ):
            store = _mooncake_store(cfg)
        self.assertEqual(store.receive_buffers, expected)
        self.assertEqual(store._receive_pool is not None, expected != "pageable")
