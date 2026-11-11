"""Checkpoint round trips and preflight failures, using tiny model fixtures."""

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from specforge.checkpoint import (
    TensorStore,
    apply_weight_plan,
    make_weight_plan,
    save_draft,
)
from specforge.checkpoint.draft_dir import read_artifact_metadata
from specforge.modeling import build_draft_model
from tests.test_config.draft_fixtures import draft_config


def _sharded_load(rank, directory):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Partial, Shard, distribute_tensor
    from torch.distributed.tensor.placement_types import _StridedShard

    dist.init_process_group(
        "gloo",
        init_method="file://" + directory + "/rendezvous",
        rank=rank,
        world_size=2,
    )
    try:
        model = build_draft_model(draft_config())
        mesh = init_device_mesh("cpu", (2,))
        for name in model.config.draft.target_bindings:
            module = model.get_submodule(name)
            module.weight = torch.nn.Parameter(
                distribute_tensor(module.weight.detach(), mesh, [Shard(0)]),
                requires_grad=False,
            )
        store = TensorStore(Path(directory) / "draft")
        target = TensorStore(Path(directory) / "target.safetensors")
        plan = make_weight_plan(model, store.refs(), bindings=target.refs())
        apply_weight_plan(model, plan)
        expected = target.read("embed_tokens")[rank * 32 : (rank + 1) * 32]
        torch.testing.assert_close(
            model.embed_tokens.weight.to_local(), expected, rtol=0, atol=0
        )
        for placement in (Partial(), _StridedShard(0, split_factor=2)):
            model.embed_tokens.weight = torch.nn.Parameter(
                DTensor.from_local(
                    torch.empty(32, 32),
                    mesh,
                    [placement],
                    shape=(64, 32),
                    stride=(32, 1),
                    run_check=False,
                ),
                requires_grad=False,
            )
            with patch.object(
                TensorStore, "read", side_effect=AssertionError("preflight read")
            ):
                with unittest.TestCase().assertRaisesRegex(
                    ValueError, type(placement).__name__
                ):
                    apply_weight_plan(model, plan)
    finally:
        dist.destroy_process_group()


class DraftWeightsTest(unittest.TestCase):
    def test_sharded_round_trip_ties_bindings_and_slices(self):
        for tied, bound in ((False, False), (True, False), (False, True)):
            with self.subTest(tied=tied, bound=bound), TemporaryDirectory() as tmp:
                config = draft_config(
                    **(
                        {}
                        if bound
                        else {"target_bindings": (), "target_logit_transform": None}
                    ),
                    tie_word_embeddings=tied,
                )
                original = build_draft_model(config)
                path = Path(tmp) / "draft"
                save_draft(original, path, max_shard_size=1024)
                store = TensorStore(path)
                payload = json.loads((path / "config.json").read_text())
                self.assertEqual(read_artifact_metadata(payload)["format"], "specforge")
                self.assertFalse((path / "artifact.json").exists())
                self.assertTrue((path / "model.safetensors.index.json").exists())
                restored = build_draft_model(config)
                bindings = {}
                if bound:
                    target = Path(tmp) / "target.safetensors"
                    save_file(
                        {n: torch.randn(64, 32) for n in config.target_bindings}, target
                    )
                    bindings = TensorStore(target).refs()
                    self.assertNotIn("lm_head.weight", store.metadata)
                plan = make_weight_plan(restored, store.refs(), bindings=bindings)
                apply_weight_plan(restored, plan)
                torch.testing.assert_close(
                    restored.state_dict(), original.state_dict(), rtol=0, atol=0
                )
                key = next(k for k, v in store.metadata.items() if len(v.shape) == 2)
                torch.testing.assert_close(
                    store.read(key, (slice(0, 1), slice(None))), store.read(key)[:1]
                )
                if tied:
                    self.assertIs(restored.embed_tokens.weight, restored.lm_head.weight)
                if bound:
                    self.assertFalse(restored.lm_head.weight.requires_grad)
                    for name, ref in bindings.items():
                        torch.testing.assert_close(
                            restored.get_submodule(name).weight,
                            ref.read(),
                            rtol=0,
                            atol=0,
                        )
                    for invalid in (
                        {},
                        bindings | {"unknown": next(iter(bindings.values()))},
                    ):
                        with self.assertRaisesRegex(ValueError, "target bindings"):
                            make_weight_plan(restored, store.refs(), bindings=invalid)
                with self.assertRaises(FileExistsError):
                    save_draft(original, path)

    def test_two_rank_target_binding(self):
        with TemporaryDirectory() as tmp:
            model = build_draft_model(draft_config())
            save_draft(model, Path(tmp) / "draft")
            save_file(
                {
                    name: torch.randn(64, 32)
                    for name in model.config.draft.target_bindings
                },
                Path(tmp) / "target.safetensors",
            )
            torch.multiprocessing.spawn(_sharded_load, args=(tmp,), nprocs=2, join=True)

    def test_preflight_rejects_without_reading_or_mutating(self):
        model = build_draft_model(
            draft_config(target_bindings=(), target_logit_transform=None)
        )
        with TemporaryDirectory() as tmp:
            save_draft(model, Path(tmp) / "draft")
            refs = TensorStore(Path(tmp) / "draft").refs()
            key = next(iter(refs))
            with patch.object(
                TensorStore, "read", side_effect=AssertionError("preflight read")
            ):
                for weights in (
                    {k: v for k, v in refs.items() if k != key},
                    refs | {"unknown": refs[key]},
                ):
                    with self.assertRaises(ValueError):
                        make_weight_plan(model, weights)
                wrong = build_draft_model(
                    draft_config(target_bindings=(), target_logit_transform=None),
                    dtype=torch.bfloat16,
                )
                with self.assertRaisesRegex(ValueError, "precision"):
                    make_weight_plan(wrong, refs, strict_copies=refs)
            plan = make_weight_plan(
                model, {k: v for k, v in refs.items() if k != key}, allow_new=[key]
            )
            before = model.state_dict()[key].clone()
            apply_weight_plan(model, plan)
            torch.testing.assert_close(model.state_dict()[key], before, rtol=0, atol=0)

    def test_shape_ties_revalidation_and_meta_fail_before_reading(self):
        config = draft_config(target_bindings=(), target_logit_transform=None)
        model = build_draft_model(config)
        with TemporaryDirectory() as tmp:
            save_draft(model, Path(tmp) / "draft")
            store = TensorStore(Path(tmp) / "draft")
            refs = store.refs()
            plan = make_weight_plan(model, refs)
            key = "embed_tokens.weight"
            info = store.metadata[key]
            before = {name: value.clone() for name, value in model.state_dict().items()}
            with patch.object(
                TensorStore, "read", side_effect=AssertionError("preflight read")
            ):
                store.metadata[key] = replace(info, shape=(1,))
                with self.assertRaisesRegex(ValueError, "shape mismatch"):
                    make_weight_plan(model, refs)
                with self.assertRaisesRegex(ValueError, "shape mismatch"):
                    apply_weight_plan(model, plan)
                store.metadata[key] = info
                tied = build_draft_model(replace(config, tie_word_embeddings=True))
                with self.assertRaisesRegex(ValueError, "conflicting assignments"):
                    make_weight_plan(tied, refs)
                meta = build_draft_model(config, device="meta")
                with self.assertRaisesRegex(ValueError, "materialize"):
                    apply_weight_plan(meta, plan)
                with self.assertRaisesRegex(ValueError, "gathered"):
                    save_draft(meta, Path(tmp) / "meta")
            torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)

    def test_legacy_tensor_payload_index_validation_and_format_guard(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            torch.save(
                {"draft_state_dict": {"weight": torch.ones(2, 3)}, "step": 7},
                path / "training_state.pt",
            )
            store = TensorStore(path)
            self.assertEqual(store.metadata["weight"].shape, (2, 3))
            torch.testing.assert_close(store.read("weight"), torch.ones(2, 3))
            index = path / "model.safetensors.index.json"
            index.write_text(
                json.dumps({"weight_map": {"weight": "../outside.safetensors"}})
            )
            with self.assertRaisesRegex(ValueError, "leaves"):
                TensorStore(path)
            with self.assertRaisesRegex(ValueError, "serving-only"):
                read_artifact_metadata(
                    {"specforge_artifact": {"format": "sglang", "version": 1}}
                )
            self.assertEqual(read_artifact_metadata({}), {})
            for metadata in (None, [], {}, {"format": "specforge", "version": 2}):
                with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                    read_artifact_metadata({"specforge_artifact": metadata})

    def test_vocabulary_is_finalized_after_direct_copies(self):
        from tests.test_modeling.test_canonical_eagle3 import (
            install_mapping,
            model_config,
        )

        model = build_draft_model(model_config())
        install_mapping(model)
        with TemporaryDirectory() as tmp:
            save_draft(model, Path(tmp) / "draft")
            refs = TensorStore(Path(tmp) / "draft").refs()
            restored = build_draft_model(model_config())
            apply_weight_plan(restored, make_weight_plan(restored, refs))
            ids = torch.arange(model.config.draft.draft_vocab_size)
            torch.testing.assert_close(
                restored.vocabulary.to_target(ids), model.vocabulary.to_target(ids)
            )
            # Invalid mappings must not become usable merely because copying succeeded.
            weights = {name: ref.read() for name, ref in refs.items()}
            weights["vocabulary.d2t"].zero_()
            save_file(weights, Path(tmp) / "bad.safetensors")
            bad = TensorStore(Path(tmp) / "bad.safetensors").refs()
            with self.assertRaisesRegex(ValueError, "ascending target IDs"):
                apply_weight_plan(restored, make_weight_plan(restored, bad))
            self.assertFalse(restored.vocabulary.ready)

    def test_failed_publish_leaves_no_artifact_and_nonzero_rank_is_rejected(self):
        model = build_draft_model(draft_config())
        with TemporaryDirectory() as tmp:
            destination = Path(tmp) / "draft"
            with patch(
                "specforge.checkpoint.draft_dir.os.replace",
                side_effect=RuntimeError("write failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "write failed"):
                    save_draft(model, destination)
            self.assertEqual(list(Path(tmp).iterdir()), [])
            with (
                patch("torch.distributed.is_initialized", return_value=True),
                patch("torch.distributed.get_rank", return_value=1),
            ):
                with self.assertRaisesRegex(ValueError, "global rank 0"):
                    save_draft(model, destination)
            self.assertFalse(destination.exists())
