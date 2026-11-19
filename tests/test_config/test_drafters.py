# coding=utf-8
"""A ``drafters`` recipe: each drafter flattens to a plain single-draft config."""

import copy
import unittest

from specforge.config import Config, flatten_drafter

DFLASH = "specforge/config/draft/presets/qwen3-8b-dflash.yaml"
DSPARK = "specforge/config/draft/presets/qwen3-8b-dspark.yaml"


def _recipe(*, managed=True):
    disaggregated = {"control_dir": "/ctl", "backend": "mooncake"}
    if managed:
        disaggregated["managed_local"] = {
            "capture_servers": [{"port": 30000, "cuda_visible_devices": ["0"]}]
        }
    else:
        disaggregated["server_urls"] = ["http://capture:30000"]
    devices = (["1", "2"], ["3"]) if managed else (None, None)
    return {
        "run_id": "qwen3",
        "output_dir": "./out",
        "model": {"target_model_path": "Qwen/Qwen3-8B"},
        "data": {"train_data_path": "/train.jsonl"},
        "training": {"learning_rate": 1e-4, "max_steps": 10},
        "drafters": [
            {
                "name": "dflash",
                "draft": DFLASH,
                "training": {"learning_rate": 5e-4},
                "trainer": {"cuda_visible_devices": devices[0]},
            },
            {
                "name": "dspark",
                "draft": DSPARK,
                "trainer": {"cuda_visible_devices": devices[1]},
            },
        ],
        "deployment": {"mode": "disaggregated", "disaggregated": disaggregated},
    }


class TestDrafters(unittest.TestCase):
    def test_a_drafter_flattens_to_the_hand_written_single_draft_config(self):
        recipe = _recipe()
        del recipe["drafters"][1]
        written = copy.deepcopy(recipe)
        del written["drafters"]
        written.update(run_id="qwen3-dflash", output_dir="./out/dflash", draft=DFLASH)
        written["training"]["learning_rate"] = 5e-4
        written["deployment"]["trainer"] = {"nproc_per_node": 2}
        disaggregated = written["deployment"]["disaggregated"]
        disaggregated.update(control_dir="/ctl/dflash", store_id="qwen3")
        disaggregated["managed_local"]["trainer_cuda_visible_devices"] = ["1", "2"]

        flat = flatten_drafter(Config.model_validate(recipe), "dflash")

        self.assertEqual(flat, Config.model_validate(written))
        self.assertEqual(
            flat.model_dump(exclude_unset=True),
            Config.model_validate(written).model_dump(exclude_unset=True),
        )

    def test_drafters_share_the_store_and_keep_their_own_state(self):
        recipe = _recipe(managed=False)
        recipe["deployment"]["disaggregated"]["consumer_state_dir"] = "/state"
        recipe["drafters"][1]["tracking"] = {"report_to": "wandb"}
        config = Config.model_validate(recipe)

        dflash, dspark = (
            flatten_drafter(config, name) for name in ("dflash", "dspark")
        )

        for flat, name in ((dflash, "dflash"), (dspark, "dspark")):
            deployment = flat.deployment.disaggregated
            self.assertEqual(deployment.store_id, "qwen3")
            self.assertEqual(deployment.control_dir, f"/ctl/{name}")
            self.assertEqual(deployment.consumer_state_dir, f"/state/{name}")
        self.assertEqual(dspark.training.learning_rate, 1e-4)
        self.assertEqual(
            (dflash.tracking.report_to, dspark.tracking.report_to), ("none", "wandb")
        )

    def test_every_conflict_raises(self):
        def drafter(index, **fields):
            return lambda raw: raw["drafters"][index].update(fields)

        def top(**fields):
            return lambda raw: raw.update(fields)

        def deployment(**fields):
            return lambda raw: raw["deployment"]["disaggregated"].update(fields)

        def managed(**fields):
            return lambda raw: raw["deployment"]["disaggregated"][
                "managed_local"
            ].update(fields)

        def multi_node(raw):
            for item in raw["drafters"]:
                item["trainer"] = {"nnodes": 2, "master_addr": "node0"}
            raw["deployment"]["disaggregated"]["consumer_state_dir"] = "/state"

        cases = {
            # What every drafter must share or keep apart.
            "top-level draft": (True, top(draft=DFLASH), "on each drafter"),
            "top-level initialization": (
                True,
                top(initialization={}),
                "on each drafter",
            ),
            "top-level trainer": (
                True,
                lambda raw: raw["deployment"].update(trainer={"nproc_per_node": 2}),
                "on each drafter",
            ),
            "top-level trainer at its defaults": (
                True,
                lambda raw: raw["deployment"].update(trainer={"nproc_per_node": 1}),
                "on each drafter",
            ),
            "top-level trainer devices": (
                True,
                managed(trainer_cuda_visible_devices=["4"]),
                "on each drafter",
            ),
            "resume": (True, top(resume="/ckpt"), "do not support"),
            "nvlink": (False, deployment(mooncake_protocol="nvlink"), "nvlink"),
            "inbox relay": (
                False,
                deployment(inbox_server_url="http://node0:8000"),
                "do not support",
            ),
            "offline": (
                False,
                top(data={"hidden_states_path": "/features"}),
                "online disaggregated",
            ),
            "same name": (True, drafter(1, name="DFlash"), "unique"),
            "reserved name": (True, drafter(1, name="logs"), "not be 'logs'"),
            "unsafe name": (True, drafter(1, name="a/b"), "letters, digits"),
            "own epochs": (
                True,
                drafter(1, training={"num_epochs": 2}),
                "num_epochs",
            ),
            "own prompt seed": (
                True,
                drafter(1, training={"prompt_seed": 1}),
                "prompt_seed",
            ),
            "own resume": (
                True,
                drafter(1, training={"resume_from": "/ckpt"}),
                "resume_from",
            ),
            "own seed orders prompts": (
                True,
                drafter(1, training={"seed": 1}),
                "training.prompt_seed",
            ),
            "shared device": (
                True,
                drafter(1, trainer={"cuda_visible_devices": ["2"]}),
                "must not share",
            ),
            "shared rendezvous": (False, multi_node, "master_port"),
            # What the flattened single-draft config rejects.
            "capture server device": (
                True,
                drafter(1, trainer={"cuda_visible_devices": ["0"]}),
                "(?s)drafter 'dspark'.*must be disjoint",
            ),
            "missing devices": (
                True,
                drafter(1, trainer={}),
                "(?s)drafter 'dspark'.*is required",
            ),
            "devices without a managed stack": (
                False,
                drafter(1, trainer={"cuda_visible_devices": ["3"]}),
                "(?s)drafter 'dspark'.*needs managed_local",
            ),
            "rank count against devices": (
                True,
                drafter(
                    1, trainer={"cuda_visible_devices": ["3"], "nproc_per_node": 2}
                ),
                "(?s)drafter 'dspark'.*count must equal",
            ),
            "no draft": (
                True,
                drafter(1, draft=None),
                "(?s)drafter 'dspark'.*requires",
            ),
        }
        for label, (managed_stack, change, message) in cases.items():
            with self.subTest(conflict=label):
                raw = _recipe(managed=managed_stack)
                change(raw)
                with self.assertRaisesRegex(ValueError, message):
                    config = Config.model_validate(raw)
                    for item in config.drafters:
                        flatten_drafter(config, item.name)

    def test_a_shared_default_is_checked_through_each_drafter(self):
        recipe = _recipe()
        # On its own the shared batch of 16 would not fit the watermarks.
        recipe["training"]["batch_size"] = 16
        recipe["runtime"] = {
            "consumer_dispatch": "two_window_bucketed",
            "in_flight_high_watermark": 8,
            "in_flight_low_watermark": 8,
        }
        for drafter in recipe["drafters"]:
            drafter["training"] = {"batch_size": 2}
        config = Config.model_validate(recipe)

        self.assertEqual(flatten_drafter(config, "dflash").training.batch_size, 2)
        recipe["drafters"][0]["training"] = {"batch_size": 4}
        with self.assertRaisesRegex(ValueError, "(?s)drafter 'dflash'.*two global"):
            flatten_drafter(Config.model_validate(recipe), "dflash")

    def test_seed_per_drafter_needs_only_a_shared_prompt_seed(self):
        recipe = _recipe()
        recipe["training"]["prompt_seed"] = 7
        recipe["drafters"][1]["training"] = {"seed": 1}
        config = Config.model_validate(recipe)

        self.assertEqual(flatten_drafter(config, "dspark").training.seed, 1)
        with self.assertRaisesRegex(ValueError, "unknown drafter 'eagle'"):
            flatten_drafter(config, "eagle")


if __name__ == "__main__":
    unittest.main()
