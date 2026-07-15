from __future__ import annotations

import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError

from specforge.application import resolve_run
from specforge.config import Config

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG_DIR = REPO_ROOT / "examples" / "configs"


def _online_payload(
    *,
    backend: str = "sglang",
    modality: str = "text",
    topology: str = "disaggregated",
) -> dict:
    payload = {
        "model": {
            "target_model_path": "target",
            "draft_model_config": "draft.json",
            "vocab_mapping_path": "mapping.pt",
            "target_backend": backend,
            "input_modality": modality,
        },
        "data": {"train_data_path": "train.jsonl"},
        "training": {"strategy": "eagle3", "max_steps": 1},
        "deployment": {"mode": topology},
    }
    if topology == "disaggregated":
        payload["deployment"]["disaggregated"] = {
            "control_dir": "outputs/test/control",
            "backend": "mooncake",
            "server_urls": ["http://127.0.0.1:30000"],
            "mooncake_metadata_server": "http://127.0.0.1:35880/metadata",
            "mooncake_master_server_addr": "127.0.0.1:35551",
        }
    return payload


class ServerOnlyOnlineConfigTest(unittest.TestCase):
    def test_colocated_online_is_rejected_at_the_config_boundary(self):
        with self.assertRaisesRegex(ValidationError, "colocated online training"):
            Config.model_validate(_online_payload(topology="local_colocated"))

    def test_online_requires_the_sglang_server_backend(self):
        for backend in ("hf", "custom"):
            with (
                self.subTest(backend=backend),
                self.assertRaisesRegex(
                    ValidationError, "external SGLang capture server"
                ),
            ):
                Config.model_validate(_online_payload(backend=backend))

    def test_vlm_is_explicitly_unsupported(self):
        config = Config.model_validate(_online_payload(modality="qwen2_5_vl"))
        self.assertEqual(config.model.input_modality, "qwen2_5_vl")
        with self.assertRaisesRegex(
            ValueError,
            "no streaming feature contract and provider for modality " "'qwen2_5_vl'",
        ):
            resolve_run(config)

    def test_offline_feature_consumers_do_not_lose_backend_compatibility(self):
        for backend in ("sglang", "hf", "custom"):
            with self.subTest(backend=backend):
                config = Config.model_validate(
                    {
                        "model": {
                            "target_model_path": "target",
                            "draft_model_config": "draft.json",
                            "vocab_mapping_path": "mapping.pt",
                            "target_backend": backend,
                        },
                        "data": {"hidden_states_path": "features"},
                    }
                )
                self.assertEqual(config.mode, "offline")

    def test_every_online_recipe_uses_the_server_data_plane(self):
        online_recipes = []
        for path in sorted(EXAMPLE_CONFIG_DIR.glob("*.yaml")):
            payload = yaml.safe_load(path.read_text())
            data = payload["data"]
            if not (data.get("train_data_path") or data.get("prompts_path")):
                continue
            online_recipes.append(path.name)
            self.assertEqual(payload["deployment"]["mode"], "disaggregated")
            self.assertEqual(payload["model"]["target_backend"], "sglang")
            self.assertFalse(payload["model"].get("shard_target_output", False))
            self.assertEqual(
                payload["deployment"]["disaggregated"]["backend"], "mooncake"
            )
        self.assertTrue(online_recipes)
        self.assertFalse(
            (EXAMPLE_CONFIG_DIR / "qwen2.5-vl-7b-eagle3-online.yaml").exists()
        )
        self.assertFalse(
            (EXAMPLE_CONFIG_DIR / "qwen2.5-vl-32b-eagle3-online.yaml").exists()
        )

    def test_application_resolution_accepts_the_server_only_contract(self):
        resolved = resolve_run(Config.model_validate(_online_payload()))
        self.assertEqual(resolved.algorithm.name, "eagle3")
        self.assertEqual(resolved.config.deployment.mode, "disaggregated")


if __name__ == "__main__":
    unittest.main(verbosity=2)
