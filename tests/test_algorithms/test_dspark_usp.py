import os
import socket
import tempfile
import unittest
from unittest import mock

import torch

from specforge.algorithms.common.dflash_family_data import (
    build_dspark_collator,
    build_dspark_offline_normalizer,
    normalize_dspark_usp_offline_sample,
)
from specforge.algorithms.common.dflash_family_model import gather_sequence_shards


def _raw_sequence(length=5):
    return {
        "input_ids": torch.arange(length, dtype=torch.long),
        "loss_mask": torch.ones(length, dtype=torch.long),
        "hidden_states": torch.arange(length * 4, dtype=torch.float32).reshape(
            1, length, 4
        ),
        "target_last_hidden_states": torch.arange(
            length * 2, dtype=torch.float32
        ).reshape(1, length, 2),
    }


def _can_bind_loopback() -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
        return True
    except OSError:
        return False


def _gather_worker(rank: int, init_file: str, output_dir: str) -> None:
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        local = torch.tensor([[[float(rank + 1)]]], requires_grad=True)
        gathered = gather_sequence_shards(
            local,
            group=dist.group.WORLD,
            global_sequence_length=2,
        )
        gathered.square().sum().backward()
        torch.save(
            {"gathered": gathered.detach(), "grad": local.grad.detach()},
            os.path.join(output_dir, f"rank{rank}.pt"),
        )
    finally:
        dist.destroy_process_group()


class TestDSparkUSPData(unittest.TestCase):
    def test_shards_both_target_hidden_tensors_with_static_padding(self):
        raw = _raw_sequence()
        left = normalize_dspark_usp_offline_sample(raw, 5, sp_rank=0, sp_size=2)
        right = normalize_dspark_usp_offline_sample(raw, 5, sp_rank=1, sp_size=2)
        self.assertEqual(
            {
                "input_ids",
                "loss_mask",
                "hidden_states",
                "target_last_hidden_states",
                "attention_mask",
            },
            set(left),
        )
        self.assertEqual([0, 1, 2], left["input_ids"][0].tolist())
        self.assertEqual([3, 4, 0], right["input_ids"][0].tolist())
        self.assertEqual([1, 1, 1], left["attention_mask"][0].tolist())
        self.assertEqual([1, 1, 0], right["attention_mask"][0].tolist())
        self.assertTrue(
            torch.equal(raw["hidden_states"][0, 3:5], right["hidden_states"][0, :2])
        )
        self.assertTrue(
            torch.equal(
                raw["target_last_hidden_states"][0, 3:5],
                right["target_last_hidden_states"][0, :2],
            )
        )
        self.assertTrue(torch.all(right["hidden_states"][0, 2] == 0))
        self.assertTrue(torch.all(right["target_last_hidden_states"][0, 2] == 0))

    def test_usp_collator_keeps_attention_mask(self):
        raw = _raw_sequence()
        features = [
            normalize_dspark_usp_offline_sample(raw, 5, sp_rank=rank, sp_size=2)
            for rank in range(2)
        ]
        batch = build_dspark_collator()(features)
        self.assertEqual((2, 3), tuple(batch["attention_mask"].shape))
        self.assertEqual([1, 1, 0], batch["attention_mask"][1].tolist())
        self.assertEqual((2, 3, 2), tuple(batch["target_last_hidden_states"].shape))

    def test_provider_normalizer_resolves_rank_local_usp_shard(self):
        group = object()
        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_rank", return_value=1),
            mock.patch("torch.distributed.get_world_size", return_value=2),
            mock.patch("specforge.distributed.get_draft_sp_group", return_value=group),
        ):
            normalizer = build_dspark_offline_normalizer(5, use_usp_preprocess=True)
            result = normalizer(_raw_sequence())
        self.assertEqual([3, 4, 0], result["input_ids"][0].tolist())

    def test_single_rank_gather_is_identity_with_gradient(self):
        local = torch.randn(1, 3, 4, requires_grad=True)
        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=1),
        ):
            gathered = gather_sequence_shards(
                local, group=object(), global_sequence_length=2
            )
        gathered.sum().backward()
        self.assertTrue(torch.equal(gathered, local[:, :2]))
        self.assertTrue(
            torch.equal(local.grad[:, :2], torch.ones_like(local.grad[:, :2]))
        )
        self.assertTrue(
            torch.equal(local.grad[:, 2:], torch.zeros_like(local.grad[:, 2:]))
        )

    @unittest.skipUnless(
        _can_bind_loopback(),
        "distributed collective test requires local loopback socket permission",
    )
    def test_two_rank_collective_reconstructs_sequence_and_routes_gradients(self):
        with tempfile.TemporaryDirectory() as tmp:
            init_file = os.path.join(tmp, "init")
            with mock.patch.dict(os.environ, {"GLOO_SOCKET_IFNAME": "lo"}):
                torch.multiprocessing.spawn(
                    _gather_worker,
                    args=(init_file, tmp),
                    nprocs=2,
                    join=True,
                )
            rank0 = torch.load(os.path.join(tmp, "rank0.pt"), weights_only=True)
            rank1 = torch.load(os.path.join(tmp, "rank1.pt"), weights_only=True)
        expected = torch.tensor([[[1.0], [2.0]]])
        self.assertTrue(torch.equal(rank0["gathered"], expected))
        self.assertTrue(torch.equal(rank1["gathered"], expected))
        self.assertTrue(torch.equal(rank0["grad"], torch.tensor([[[4.0]]])))
        self.assertTrue(torch.equal(rank1["grad"], torch.tensor([[[8.0]]])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
