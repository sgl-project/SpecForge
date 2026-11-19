# coding=utf-8
"""A ``drafters`` recipe: each drafter flattens to a plain single-draft config."""

import copy
import unittest
from unittest import mock

from specforge.config import Config, flatten_drafters

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


def _flat(raw):
    return dict(flatten_drafters(Config.model_validate(raw)))


def _drafter(index, **fields):
    return lambda raw: raw["drafters"][index].update(fields)


def _top(**fields):
    return lambda raw: raw.update(fields)


def _deployment(**fields):
    return lambda raw: raw["deployment"]["disaggregated"].update(fields)


def _managed(**fields):
    return lambda raw: raw["deployment"]["disaggregated"]["managed_local"].update(
        fields
    )


def _multi_node(raw):
    for item in raw["drafters"]:
        item["trainer"] = {"nnodes": 2, "master_addr": "node0"}
    raw["deployment"]["disaggregated"]["consumer_state_dir"] = "/state"


class TestDrafters(unittest.TestCase):
    def test_a_config_without_drafters_is_one_unnamed_run(self):
        recipe = _recipe()
        del recipe["drafters"]
        recipe["draft"] = DFLASH
        recipe["deployment"]["trainer"] = {"nproc_per_node": 2}
        managed = recipe["deployment"]["disaggregated"]["managed_local"]
        managed["trainer_cuda_visible_devices"] = ["1", "2"]
        config = Config.model_validate(recipe)

        ((name, run),) = flatten_drafters(config)

        self.assertIsNone(name)
        self.assertIs(run, config)

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

        ((name, flat),) = flatten_drafters(Config.model_validate(recipe))

        self.assertEqual(name, "dflash")
        self.assertEqual(flat, Config.model_validate(written))
        self.assertEqual(
            flat.model_dump(exclude_unset=True),
            Config.model_validate(written).model_dump(exclude_unset=True),
        )

    def test_drafters_share_the_store_and_keep_their_own_state(self):
        recipe = _recipe(managed=False)
        recipe["deployment"]["disaggregated"]["consumer_state_dir"] = "/state"
        recipe["drafters"][1]["tracking"] = {"report_to": "wandb"}

        runs = _flat(recipe)

        self.assertEqual(list(runs), ["dflash", "dspark"])
        for name, flat in runs.items():
            deployment = flat.deployment.disaggregated
            self.assertEqual(deployment.store_id, "qwen3")
            self.assertEqual(deployment.control_dir, f"/ctl/{name}")
            self.assertEqual(deployment.consumer_state_dir, f"/state/{name}")
        self.assertEqual(runs["dspark"].training.learning_rate, 1e-4)
        self.assertEqual(
            (runs["dflash"].tracking.report_to, runs["dspark"].tracking.report_to),
            ("none", "wandb"),
        )

    def test_a_recipe_conflict_raises_at_load(self):
        cases = {
            "top-level draft": (True, _top(draft=DFLASH), "on each drafter"),
            "top-level initialization": (
                True,
                _top(initialization={}),
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
                _managed(trainer_cuda_visible_devices=["4"]),
                "on each drafter",
            ),
            "resume": (True, _top(resume="/ckpt"), "do not support"),
            "nvlink": (False, _deployment(mooncake_protocol="nvlink"), "nvlink"),
            "inbox relay": (
                False,
                _deployment(inbox_server_url="http://node0:8000"),
                "do not support",
            ),
            "prompts other than text": (
                True,
                lambda raw: raw["model"].update(input_modality="qwen2_5_vl"),
                "input_modality other than 'text'",
            ),
            "offline": (
                False,
                _top(data={"hidden_states_path": "/features"}),
                "online disaggregated",
            ),
            "same name": (True, _drafter(1, name="DFlash"), "unique"),
            "reserved name": (True, _drafter(1, name="logs"), "not be 'logs'"),
            "unsafe name": (True, _drafter(1, name="a/b"), "letters, digits"),
            "own epochs": (
                True,
                _drafter(1, training={"num_epochs": 2}),
                "num_epochs",
            ),
            "own prompt seed": (
                True,
                _drafter(1, training={"prompt_seed": 1}),
                "prompt_seed",
            ),
            "own resume": (
                True,
                _drafter(1, training={"resume_from": "/ckpt"}),
                "resume_from",
            ),
            "own seed orders prompts": (
                True,
                _drafter(1, training={"seed": 1}),
                "training.prompt_seed",
            ),
            "shared device": (
                True,
                _drafter(1, trainer={"cuda_visible_devices": ["2"]}),
                "must not share",
            ),
            "devices without a managed stack": (
                False,
                _drafter(1, trainer={"cuda_visible_devices": ["3"]}),
                "drafter 'dspark'.*needs managed_local",
            ),
            "shared rendezvous": (False, _multi_node, "master_port"),
        }
        for label, (managed_stack, change, message) in cases.items():
            with self.subTest(conflict=label):
                raw = _recipe(managed=managed_stack)
                change(raw)
                with self.assertRaisesRegex(ValueError, message):
                    Config.model_validate(raw)

    def test_a_drafter_conflict_raises_when_flattened(self):
        cases = {
            "capture server device": (
                _drafter(1, trainer={"cuda_visible_devices": ["0"]}),
                "must be disjoint",
            ),
            "missing devices": (_drafter(1, trainer={}), "is required"),
            "rank count against devices": (
                _drafter(
                    1, trainer={"cuda_visible_devices": ["3"], "nproc_per_node": 2}
                ),
                "count must equal",
            ),
            "no draft": (_drafter(1, draft=None), "requires"),
        }
        for label, (change, message) in cases.items():
            with self.subTest(conflict=label):
                raw = _recipe()
                change(raw)
                config = Config.model_validate(raw)
                with self.assertRaisesRegex(
                    ValueError, f"(?s)drafter 'dspark'.*{message}"
                ):
                    flatten_drafters(config)

    def test_a_shared_default_is_checked_through_each_drafter(self):
        recipe = _recipe()
        recipe["training"]["batch_size"] = 16
        recipe["runtime"] = {
            "consumer_dispatch": "two_window_bucketed",
            "in_flight_high_watermark": 8,
            "in_flight_low_watermark": 8,
        }
        for drafter in recipe["drafters"]:
            drafter["training"] = {"batch_size": 2}

        self.assertEqual(_flat(recipe)["dflash"].training.batch_size, 2)
        recipe["drafters"][0]["training"] = {"batch_size": 4}
        with self.assertRaisesRegex(ValueError, "(?s)drafter 'dflash'.*two global"):
            _flat(recipe)

    def test_seed_per_drafter_needs_only_a_shared_prompt_seed(self):
        recipe = _recipe()
        recipe["training"]["prompt_seed"] = 7
        recipe["drafters"][1]["training"] = {"seed": 1}

        self.assertEqual(_flat(recipe)["dspark"].training.seed, 1)


class TestResolveRun(unittest.TestCase):
    def test_a_recipe_with_one_drafter_resolves_as_its_flat_config(self):
        from specforge.application import resolve_run

        recipe = _recipe()
        del recipe["drafters"][1]
        with (
            mock.patch(
                "specforge.training.canonical.resolve_run",
                side_effect=RuntimeError("resolved"),
            ) as resolve,
            self.assertRaisesRegex(RuntimeError, "resolved"),
        ):
            resolve_run(Config.model_validate(recipe))

        self.assertEqual(resolve.call_args.args[0], _flat(recipe)["dflash"])

    def test_several_drafters_are_refused_before_resolving(self):
        from specforge.application import resolve_run

        with (
            mock.patch("specforge.training.canonical.resolve_run") as resolve,
            self.assertRaisesRegex(ValueError, "several drafters"),
        ):
            resolve_run(Config.model_validate(_recipe()))

        resolve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
