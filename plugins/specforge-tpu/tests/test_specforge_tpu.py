# coding=utf-8
"""CPU-only contracts of the specforge_tpu plugin (no torch_tpu required)."""

from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from specforge.launch_plan import CommandSpec, LaunchPlan
from specforge_tpu import patches, producer
from specforge_tpu import sglang_capture as capture
from specforge_tpu import shapes


class ActivationTest(unittest.TestCase):
    def test_registration_requires_explicit_tpu_device(self):
        import specforge_tpu

        with (
            mock.patch.dict(os.environ, clear=True),
            mock.patch("importlib.util.find_spec", return_value=object()),
            mock.patch.object(patches, "apply") as apply,
        ):
            specforge_tpu.register()
            os.environ["SPECFORGE_DEVICE"] = "cuda"
            specforge_tpu.register()
            apply.assert_not_called()

    def test_explicit_topology_skips_detection(self):
        with mock.patch.dict(
            os.environ, {"WORLD_SIZE": "8", "TORCH_TPU_TOPOLOGY": "custom"}
        ):
            patches._resolve_rank_topology()
            self.assertEqual(os.environ["TORCH_TPU_TOPOLOGY"], "custom")


class WireCodecTest(unittest.TestCase):
    def test_spec_rides_in_per_request_custom_params(self):
        specs = [{"sample_id": "r:0"}, {"sample_id": "r:1"}]
        body = producer.encode_capture_body(
            {
                "input_ids": [[1], [2]],
                "sampling_params": {"max_new_tokens": 0},
                "spec_capture": specs,
            }
        )
        self.assertNotIn("spec_capture", body)
        self.assertTrue(body["return_hidden_states"])
        self.assertEqual(
            [
                json.loads(p["custom_params"]["spec_capture"])
                for p in body["sampling_params"]
            ],
            specs,
        )
        self.assertEqual(body["sampling_params"][1]["max_new_tokens"], 0)

    def test_plain_requests_pass_through(self):
        body = {"input_ids": [[1]], "sampling_params": {"max_new_tokens": 1}}
        self.assertEqual(producer.encode_capture_body(dict(body)), body)


class ShapesTest(unittest.TestCase):
    def test_registration_uses_longest_sequence_by_default(self):
        self.check_registered_shapes(buckets=None, expected_length=5)

    def test_registration_uses_configured_buckets(self):
        self.check_registered_shapes(buckets="4,8", expected_length=8)

    def check_registered_shapes(self, buckets, expected_length):
        import specforge_tpu
        from specforge import distributed, launch_plan, utils
        from specforge.algorithms.common import batch_shapes
        from specforge.algorithms.objectives.inputs import (
            prepare_anchor_capacity,
            sample_anchors,
        )
        from specforge.training.canonical_data import build_collator

        for owner, name in (
            (specforge_tpu, "_registered"),
            (utils, "get_local_device"),
            (distributed, "_bind_local_device"),
            (launch_plan, "_trainer_command"),
            (batch_shapes, "padded_sequence_length"),
            (batch_shapes, "anchor_capacity"),
        ):
            self.addCleanup(setattr, owner, name, getattr(owner, name))
        with (
            mock.patch.dict(
                os.environ,
                {"SPECFORGE_DEVICE": "tpu"},
            ),
            mock.patch.dict(distributed._DISTRIBUTED_BACKENDS),
            mock.patch.object(patches, "apply_host"),
            mock.patch.object(
                patches, "import_module", side_effect=[ImportError("torch_tpu"), None]
            ) as compiler,
        ):
            os.environ.pop("WORLD_SIZE", None)
            if buckets is None:
                os.environ.pop("SPECFORGE_TPU_SEQ_BUCKETS", None)
            else:
                os.environ["SPECFORGE_TPU_SEQ_BUCKETS"] = buckets
            with self.assertRaises(ImportError):
                specforge_tpu.register()
            specforge_tpu.register()
            specforge_tpu.register()
            self.assertEqual(compiler.call_count, 2)
            self.assertEqual(distributed._distributed_backend("tpu"), "tpu_dist")

            ids = torch.ones(1, 5, dtype=torch.long)
            feature = {"input_ids": ids, "loss_mask": ids, "lengths": torch.tensor([5])}
            batch = build_collator(keys=("input_ids", "loss_mask"))([feature])
            self.assertEqual(tuple(batch["input_ids"].shape), (1, expected_length))
            self.assertEqual(batch["attention_mask"].sum().item(), 5)
            self.assertIs(batch["lengths"], feature["lengths"])
            for valid_count in (4, 0):
                if not valid_count:
                    batch["loss_mask"].zero_()
                capacity = prepare_anchor_capacity(batch, None)[
                    "anchor_capacity"
                ].numel()
                valid = batch["loss_mask"].bool()
                _, keep = sample_anchors(valid[:, :-1] & valid[:, 1:], 8, capacity)
                self.assertEqual(keep.numel(), expected_length - 1)
                self.assertEqual(keep.sum().item(), valid_count)

    def test_length_past_the_largest_bucket_raises(self):
        with self.assertRaises(ValueError):
            shapes.bucket_for(9, (4, 8))

    def test_offline_recipe_selects_dspark(self):
        from specforge.application.composition import resolve_run
        from specforge.config import load_config

        cfg = load_config(
            "examples/configs/offline/colocated/gpt-oss-120b-dspark-tpu.yaml"
        )
        self.assertEqual(resolve_run(cfg).algorithm.spec.name, "dspark")


class TrainerCommandTest(unittest.TestCase):
    def test_producer_child_stays_off_the_chips(self):
        plan = LaunchPlan(
            "supervisor",
            "both",
            commands=(
                CommandSpec("producer", ("p",), {"A": "1"}),
                CommandSpec("consumer", ("c",), {}),
            ),
        )
        out = patches._build_launch_plan(lambda: plan)
        self.assertEqual(out.commands[0].env["TORCH_DEVICE_BACKEND_AUTOLOAD"], "0")
        self.assertEqual(out.commands[0].env["A"], "1")
        self.assertEqual(out.commands[1].env, {})

    def test_trainer_ranks_share_slicebuilder_ports(self):
        cfg = SimpleNamespace(
            deployment=SimpleNamespace(
                trainer=SimpleNamespace(nnodes=1, nproc_per_node=4)
            )
        )
        command = patches._trainer_command(
            lambda cfg, **_: CommandSpec("consumer", ("torchrun",), {}), cfg
        )
        self.assertEqual(
            len(command.env["TORCH_TPU_SLICEBUILDER_ADDRESSES"].split(",")), 4
        )
        self.assertEqual(command.env["TORCH_TPU_DEVICES_PER_HOST"], "4")


class _Req:
    def __init__(self, spec, prompt_len):
        self.sampling_params = SimpleNamespace(
            custom_params={"spec_capture": json.dumps(spec)}
        )
        self.origin_input_ids = list(range(prompt_len))
        self.inflight_middle_chunks = 0
        self.is_retracted = False
        self.rid = spec["sample_id"]

    def finished(self):
        return False


class CaptureHookTest(unittest.TestCase):
    def setUp(self):
        self.sink = mock.Mock()
        self.sink.put_samples.side_effect = lambda samples: [
            {"sample_id": s["sample_id"], "aux": tuple(a.shape), "last": tuple(l.shape)}
            for s, a, l in samples
        ]
        patcher = mock.patch.multiple(capture, _aux_layer_ids=[1, 2], _sink=self.sink)
        patcher.start()
        self.addCleanup(patcher.stop)
        writer = mock.patch.object(capture, "_is_writer", return_value=True)
        writer.start()
        self.addCleanup(writer.stop)

    def _append(self, req, rows, offset, length, store=True):
        return capture.append_prefill_hidden_states(
            None,
            None,
            req=req,
            logits_output=SimpleNamespace(hidden_states=rows),
            hidden_state_offset=offset,
            capture_hidden_mode=SimpleNamespace(is_full=lambda: True),
            extend_input_len=length,
            store=store,
        )

    def test_chunked_prefill_rows_are_joined_and_split(self):
        req = _Req({"sample_id": "s"}, prompt_len=5)
        rows = torch.arange(10 * 6, dtype=torch.float32).reshape(10, 6)
        self.assertEqual(self._append(req, rows, 0, 3, store=False), 3)
        self.sink.put_samples.assert_not_called()
        self.assertEqual(self._append(req, rows, 4, 2), 6)
        result = req._specforge_result
        self.assertEqual((result["aux"], result["last"]), ((5, 4), (5, 2)))

    def test_result_is_streamed_once_per_request(self):
        req = _Req({"sample_id": "s"}, prompt_len=1)
        req._specforge_result = {"sample_id": "s"}
        acc = SimpleNamespace(rids=["other", "s"], customized_info={})
        capture.stream_capture_result(None, acc, req=req)
        self.assertEqual(
            acc.customized_info["spec_capture"], [[], [{"sample_id": "s"}]]
        )


if __name__ == "__main__":
    unittest.main()
