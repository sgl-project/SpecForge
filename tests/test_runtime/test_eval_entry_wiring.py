# coding=utf-8
"""CPU wiring gates for configured eval on the one ``specforge train`` path."""

import types
import unittest
from unittest import mock

from specforge.config import Config
from specforge.training.assembly import (
    ModelBundle,
    _prepare_prompts,
    build_training_run,
)


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
                    "eval_data_path": "/eval.jsonl",
                    "cache_key": "train-cache",
                },
                "training": {"eval_interval": 2},
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
                path=cfg.data.eval_data_path,
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
            run = build_training_run(cfg)

        self.assertIs(run.trainer, trainer)
        self.assertEqual(build.call_args.kwargs["eval_interval"], 7)
        self.assertEqual(build.call_args.kwargs["seed"], 19)
        self.assertEqual(
            build.call_args.kwargs["eval_hidden_states_path"], "/eval-features"
        )

    def test_online_config_builds_a_distinct_eval_prompt_source(self):
        cfg = Config.model_validate(
            {
                "model": {
                    "target_model_path": "target",
                    "draft_model_config": "draft.json",
                },
                "data": {
                    "train_data_path": "/train.jsonl",
                    "eval_data_path": "/eval.jsonl",
                },
                "training": {"eval_interval": 4, "max_steps": 2},
            }
        )
        bundle = ModelBundle(
            model=object(),
            draft_model=object(),
            tokenizer=object(),
            target_engine=object(),
            target_hidden_size=8,
            target_vocab_size=16,
            draft_vocab_size=16,
            strategy_kwargs={},
        )
        train_prompts = [{"task_id": "train"}]
        eval_prompts = [{"task_id": "eval"}]
        trainer = object()
        with (
            mock.patch(
                "specforge.training.assembly.build_model_bundle",
                return_value=bundle,
            ),
            mock.patch(
                "specforge.training.assembly._prepare_prompts",
                side_effect=[train_prompts, eval_prompts],
            ) as prepare,
            mock.patch("specforge.training.assembly._device", return_value="npu:3"),
            mock.patch(
                "specforge.launch.build_online_runtime", return_value=trainer
            ) as build,
        ):
            run = build_training_run(cfg)

        self.assertIs(run.trainer, trainer)
        self.assertEqual(prepare.call_count, 2)
        self.assertEqual(prepare.call_args_list[1].kwargs["path"], "/eval.jsonl")
        self.assertTrue(
            prepare.call_args_list[1].kwargs["cache_key"].startswith("eval-")
        )
        planned_eval = build.call_args.kwargs["eval_prompts"]
        self.assertEqual(len(planned_eval), 1)
        self.assertEqual(planned_eval[0]["metadata"]["base_task_id"], "eval")
        self.assertEqual(planned_eval[0]["metadata"]["prompt_epoch"], 0)
        self.assertEqual(build.call_args.kwargs["eval_interval"], 4)
        self.assertEqual(build.call_args.kwargs["device"], "npu:3")

    def test_local_online_resume_prepositions_the_deterministic_prompt_plan(self):
        from specforge.launch import _plan_online_prompt_stream

        cfg = Config.model_validate(
            {
                "model": {
                    "target_model_path": "target",
                    "draft_model_config": "draft.json",
                },
                "data": {"train_data_path": "/train.jsonl"},
                "training": {
                    "num_epochs": 2,
                    "seed": 17,
                    "resume_from": "/checkpoints/run-latest",
                },
            }
        )
        bundle = ModelBundle(
            model=object(),
            draft_model=object(),
            tokenizer=object(),
            target_engine=object(),
            target_hidden_size=8,
            target_vocab_size=16,
            draft_vocab_size=16,
            strategy_kwargs={},
        )
        source = [
            {"task_id": f"p{i}", "payload": {"input_ids": [i]}}
            for i in range(3)
        ]
        state = {"epoch": 0, "epoch_samples": 2}
        trainer = object()
        with (
            mock.patch(
                "specforge.training.assembly.build_model_bundle", return_value=bundle
            ),
            mock.patch(
                "specforge.training.assembly._prepare_prompts", return_value=source
            ),
            mock.patch(
                "specforge.training.checkpoint.CheckpointManager.read_resume_state",
                return_value=state,
            ),
            mock.patch(
                "specforge.launch.build_online_runtime", return_value=trainer
            ) as build,
        ):
            run = build_training_run(cfg)

        expected = _plan_online_prompt_stream(
            source,
            num_epochs=2,
            seed=17,
            tp_size=1,
            batch_size=1,
            dp_rank=0,
            dp_size=1,
        )
        self.assertIs(run.trainer, trainer)
        self.assertEqual(build.call_args.kwargs["prompts"], expected[2:])
        self.assertEqual(build.call_args.kwargs["dataset_size"], 6)
        self.assertIs(build.call_args.kwargs["resume_state"], state)
        self.assertEqual(build.call_args.kwargs["prompt_epochs"], 1)
        self.assertEqual(
            build.call_args.kwargs["checkpoint_extra"]["prompt_epochs"], 2
        )


class TestEvalLaunch(unittest.TestCase):
    @staticmethod
    def _offline_spec(reader, collate, transform):
        return types.SimpleNamespace(
            name="eagle3",
            uses_target_head=True,
            make_offline_reader=lambda *args, **kwargs: reader(*args, **kwargs),
            make_offline_collate=lambda: collate,
            make_offline_transform=lambda *args, **kwargs: transform,
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
        spec = self._offline_spec(reader, collate, transform)
        store = object()
        with mock.patch("specforge.launch.LocalFeatureStore", return_value=store):
            factory = _make_offline_eval_data_factory(
                spec=spec,
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
        spec = self._offline_spec(reader, mock.Mock(), mock.Mock())
        eval_factory = mock.Mock(name="eval_factory")
        trainer = object()
        with (
            mock.patch("specforge.launch.resolve_strategy", return_value=spec),
            mock.patch("specforge.launch.DataFlowController"),
            mock.patch("specforge.launch.LocalFeatureStore"),
            mock.patch(
                "specforge.launch._make_offline_eval_data_factory",
                return_value=eval_factory,
            ),
            mock.patch(
                "specforge.launch._assemble_trainer", return_value=trainer
            ) as assemble,
        ):
            result = build_offline_runtime(
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

    def test_online_eval_factory_builds_and_closes_a_private_stream_each_time(self):
        from specforge.launch import _make_online_eval_data_factory

        spec = types.SimpleNamespace(name="eagle3")
        controllers = [
            mock.MagicMock(name="controller-1"),
            mock.MagicMock(name="controller-2"),
        ]
        stores = [mock.MagicMock(name="store-1"), mock.MagicMock(name="store-2")]
        streams = [
            mock.MagicMock(name="stream-1"),
            mock.MagicMock(name="stream-2"),
        ]
        loaders = [mock.MagicMock(name="loader-1"), mock.MagicMock(name="loader-2")]
        worker_kwargs = {
            "target_model": object(),
            "num_rollout_workers": 1,
            "batch_partition": types.SimpleNamespace(size=1),
        }
        with (
            mock.patch(
                "specforge.launch.DataFlowController", side_effect=controllers
            ) as controller_cls,
            mock.patch("specforge.launch.LocalFeatureStore", side_effect=stores),
            mock.patch(
                "specforge.launch._assemble_rollout_workers",
                side_effect=[[object()], [object()]],
            ),
            mock.patch("specforge.launch.LocalRolloutStream", side_effect=streams),
            mock.patch(
                "specforge.launch.FeatureDataLoader", side_effect=loaders
            ) as loader_cls,
        ):
            factory = _make_online_eval_data_factory(
                spec=spec,
                prompts=[{"task_id": "eval", "payload": {"input_ids": [1]}}],
                run_id="run",
                batch_size=2,
                collate_fn="collate",
                rollout_worker_kwargs=worker_kwargs,
                dataloader_num_workers=3,
            )
            first = factory()
            second = factory()
            with first as entered:
                self.assertIs(entered, loaders[0])
            with second as entered:
                self.assertIs(entered, loaders[1])

        self.assertEqual(
            [call.args[0] for call in controller_cls.call_args_list],
            ["run-eval-000001", "run-eval-000002"],
        )
        first_prompts = controllers[0].ingest_prompts.call_args.args[0]
        second_prompts = controllers[1].ingest_prompts.call_args.args[0]
        self.assertEqual(first_prompts, second_prompts)
        self.assertIsNot(first_prompts, second_prompts)
        for stream in streams:
            stream.close.assert_called_once_with(reason="evaluation_finished")
        for loader in loaders:
            loader.close.assert_called_once_with()
        for call, stream in zip(loader_cls.call_args_list, streams):
            self.assertIs(call.kwargs["queue"], stream)
            self.assertFalse(call.kwargs["drop_last"])
            self.assertEqual(call.kwargs["num_workers"], 3)

    def test_online_builder_passes_the_eval_factory_to_the_one_trainer(self):
        from specforge.launch import build_online_runtime

        spec = types.SimpleNamespace(
            name="eagle3", make_online_collate=lambda: "collate"
        )
        controller = mock.MagicMock()
        stream = mock.MagicMock()
        trainer = mock.MagicMock()
        eval_factory = mock.Mock(name="eval_factory")
        with (
            mock.patch("specforge.launch.resolve_strategy", return_value=spec),
            mock.patch("specforge.launch.DataFlowController", return_value=controller),
            mock.patch("specforge.launch.LocalFeatureStore"),
            mock.patch(
                "specforge.launch._assemble_rollout_workers", return_value=[object()]
            ),
            mock.patch("specforge.launch.LocalRolloutStream", return_value=stream),
            mock.patch(
                "specforge.launch._make_online_eval_data_factory",
                return_value=eval_factory,
            ) as make_eval,
            mock.patch(
                "specforge.launch._assemble_trainer", return_value=trainer
            ) as assemble,
        ):
            result = build_online_runtime(
                target_model=object(),
                prompts=[{"task_id": "train"}],
                eval_prompts=[{"task_id": "eval"}],
                eval_interval=5,
                draft_model=object(),
                optimizer_factory=object(),
                run_id="run",
                output_dir="/out",
                target_hidden_size=8,
            )

        self.assertIs(result, trainer)
        self.assertEqual(make_eval.call_args.kwargs["prompts"], [{"task_id": "eval"}])
        self.assertEqual(assemble.call_args.kwargs["eval_interval"], 5)
        self.assertIs(assemble.call_args.kwargs["eval_data_factory"], eval_factory)


if __name__ == "__main__":
    unittest.main(verbosity=2)
