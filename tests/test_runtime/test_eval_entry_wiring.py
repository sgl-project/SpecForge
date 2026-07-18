# coding=utf-8
"""CPU wiring gates for configured eval on the one ``specforge train`` path."""

import types
import unittest
from unittest import mock

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.config import Config
from specforge.training.assembly import (
    ModelBundle,
    _prepare_prompts,
    build_training_run,
)

ALGORITHM = builtin_algorithm_registry().resolve("eagle3")


class TestEvalAssembly(unittest.TestCase):
    def test_prompt_preparation_honors_eval_path_and_cache_key_overrides(self):
        cfg = Config.model_validate(
            {
                "model": {
                    "target_model_path": "target",
                    "draft_model_config": "draft.json",
                },
                "data": {
                    "train_data_path": "/train.jsonl",
                    "cache_key": "train-cache",
                },
                "training": {"max_steps": 1},
                "deployment": {
                    "mode": "disaggregated",
                    "disaggregated": {
                        "control_dir": "/tmp/eval-prompt-control",
                        "backend": "mooncake",
                        "server_urls": ["http://capture.invalid:30000"],
                        "mooncake_metadata_server": (
                            "http://metadata.invalid:35880/metadata"
                        ),
                        "mooncake_master_server_addr": "master.invalid:35551",
                        "mooncake_protocol": "tcp",
                    },
                },
            }
        )
        tokenizer = object()
        with mock.patch(
            "specforge.data.prompt_builder.prepare_prompt_tasks",
            return_value=[{"task_id": "eval"}],
        ) as prepare:
            result = _prepare_prompts(
                cfg,
                tokenizer,
                algorithm=ALGORITHM,
                draft_config=object(),
                path="/eval.jsonl",
                cache_key="eval-cache",
            )

        self.assertEqual(result, [{"task_id": "eval"}])
        self.assertEqual(prepare.call_args.args[:2], ("/eval.jsonl", tokenizer))
        self.assertEqual(prepare.call_args.kwargs["cache_key"], "eval-cache")

    def test_offline_config_wires_eval_into_the_unified_runtime(self):
        cfg = Config.model_validate(
            {
                "model": {
                    "target_model_path": "target",
                    "draft_model_config": "draft.json",
                    "vocab_mapping_path": "mapping.pt",
                },
                "data": {
                    "hidden_states_path": "/train-features",
                    "eval_hidden_states_path": "/eval-features",
                },
                "training": {"eval_interval": 7, "seed": 19},
            }
        )
        bundle = ModelBundle(
            model=object(),
            draft_model=object(),
            draft_config=object(),
            target_head=object(),
            strategy_kwargs={},
        )
        trainer = object()
        with (
            mock.patch(
                "specforge.training.assembly.build_model_bundle",
                return_value=bundle,
            ),
            mock.patch(
                "specforge.launch.build_offline_runtime", return_value=trainer
            ) as build,
        ):
            run = build_training_run(cfg, algorithm=ALGORITHM)

        self.assertIs(run.trainer, trainer)
        self.assertEqual(build.call_args.kwargs["eval_interval"], 7)
        self.assertEqual(build.call_args.kwargs["seed"], 19)
        self.assertEqual(
            build.call_args.kwargs["eval_hidden_states_path"], "/eval-features"
        )


class TestEvalLaunch(unittest.TestCase):
    @staticmethod
    def _offline_algorithm(reader):
        provider = types.SimpleNamespace(build_reader=reader)
        return types.SimpleNamespace(
            name="eagle3",
            providers=types.SimpleNamespace(
                offline_for=lambda _modality: provider,
                step=types.SimpleNamespace(uses_external_target_head=True),
            ),
        )

    def test_offline_eval_factory_reuses_io_contract_and_keeps_partial_batch(self):
        from specforge.launch import _make_offline_eval_data_factory

        refs = [object(), object(), object()]
        reader_calls = []

        def reader(path, **kwargs):
            reader_calls.append((path, kwargs))
            return types.SimpleNamespace(read=lambda: refs)

        collate = mock.Mock(name="collate")
        transform = mock.Mock(name="transform")
        algorithm = self._offline_algorithm(reader)
        store = object()
        with (
            mock.patch("specforge.launch.LocalFeatureStore", return_value=store),
            mock.patch(
                "specforge.launch._offline_io", return_value=(collate, transform)
            ),
        ):
            factory = _make_offline_eval_data_factory(
                algorithm=algorithm,
                modality="text",
                hidden_states_path="/eval-features",
                run_id="run",
                batch_size=2,
                max_len=128,
                ttt_length=5,
                use_usp_preprocess=False,
                dataloader_num_workers=3,
            )

        first = factory()
        second = factory()
        self.assertIsNot(first, second)
        self.assertIs(first.store, store)
        self.assertEqual(first._refs, refs)
        self.assertIs(first.collate_fn, collate)
        self.assertIs(first.per_sample_transform, transform)
        self.assertFalse(first.drop_last)
        self.assertEqual(first.num_workers, 3)
        self.assertEqual(
            reader_calls,
            [
                (
                    "/eval-features",
                    {"run_id": "run-eval", "ttt_length": 5, "max_len": 128},
                )
            ],
        )

    def test_offline_builder_passes_eval_factory_to_the_one_trainer(self):
        from specforge.launch import build_offline_runtime

        reader = lambda *_args, **_kwargs: types.SimpleNamespace(read=lambda: [])
        collate = mock.Mock(name="collate")
        transform = mock.Mock(name="transform")
        algorithm = self._offline_algorithm(reader)
        eval_factory = mock.Mock(name="eval_factory")
        trainer = object()
        with (
            mock.patch("specforge.launch.DataFlowController"),
            mock.patch("specforge.launch.LocalFeatureStore"),
            mock.patch(
                "specforge.launch._offline_io", return_value=(collate, transform)
            ),
            mock.patch(
                "specforge.launch._make_offline_eval_data_factory",
                return_value=eval_factory,
            ),
            mock.patch(
                "specforge.launch._assemble_trainer", return_value=trainer
            ) as assemble,
        ):
            result = build_offline_runtime(
                algorithm=algorithm,
                hidden_states_path="/train",
                eval_hidden_states_path="/eval",
                eval_interval=3,
                draft_model=object(),
                target_head=object(),
                optimizer_factory=object(),
                run_id="run",
                output_dir="/out",
                seed=31,
            )

        self.assertIs(result, trainer)
        self.assertEqual(assemble.call_args.kwargs["eval_interval"], 3)
        self.assertIs(assemble.call_args.kwargs["eval_data_factory"], eval_factory)
        self.assertTrue(
            callable(assemble.call_args.kwargs["ref_source"]["refs_for_epoch"])
        )
        self.assertEqual(
            assemble.call_args.kwargs["checkpoint_extra"]["sampler_seed"], 31
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
