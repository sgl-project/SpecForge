"""Two-rank DSpark Ulysses parity against full-sequence flex attention."""

from __future__ import annotations

import os
import tempfile
import types
import unittest

import torch

from tests.test_runtime import _fixtures as fx

WORLD_SIZE = 2
CUDA = torch.cuda.is_available()
NGPU = torch.cuda.device_count() if CUDA else 0


def _has_ulysses() -> bool:
    if NGPU < WORLD_SIZE:
        return False
    try:
        from yunchang.comm import SeqAllToAll4D  # noqa: F401
    except Exception:
        return False
    return True


def _bind_manual_anchors(model) -> None:
    def sample(self, seq_len, loss_mask, device):
        del seq_len, loss_mask
        return (
            torch.tensor([[2, 8]], dtype=torch.long, device=device),
            torch.ones((1, 2), dtype=torch.bool, device=device),
        )

    model._sample_anchor_positions = types.MethodType(sample, model)


def _worker(rank: int, world_size: int, port: int, workdir: str) -> None:
    fx.init_rank_distributed(
        rank,
        world_size,
        tp_size=1,
        sp_ulysses_size=world_size,
        sp_ring_size=1,
        port=str(port),
    )
    try:
        import torch.distributed as dist

        rank_dir = os.path.join(workdir, f"rank-{rank}")
        os.makedirs(rank_dir)
        reference, captured_width = fx.build_dspark(
            os.path.join(rank_dir, "reference"),
            hidden=64,
            block_size=4,
            num_anchors=2,
            attention_backend="flex_attention",
        )
        ulysses, usp_width = fx.build_dspark(
            os.path.join(rank_dir, "ulysses"),
            hidden=64,
            block_size=4,
            num_anchors=2,
            attention_backend="usp",
        )
        if captured_width != usp_width:
            raise AssertionError((captured_width, usp_width))
        ulysses.load_state_dict(reference.state_dict())
        _bind_manual_anchors(reference)
        _bind_manual_anchors(ulysses)

        torch.manual_seed(2026)
        sequence_length = 12
        input_ids = torch.arange(
            1, sequence_length + 1, device="cuda", dtype=torch.long
        ).unsqueeze(0)
        loss_mask = torch.ones_like(input_ids)
        hidden_states = torch.randn(
            1,
            sequence_length,
            captured_width,
            device="cuda",
            dtype=torch.bfloat16,
        )
        target_last_hidden = torch.randn(
            1,
            sequence_length,
            64,
            device="cuda",
            dtype=torch.bfloat16,
        )

        reference_loss, _, _ = reference(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
            target_last_hidden_states=target_last_hidden,
        )

        local_length = sequence_length // world_size
        start = rank * local_length
        stop = start + local_length
        ulysses_loss, _, _ = ulysses(
            input_ids=input_ids[:, start:stop],
            hidden_states=hidden_states[:, start:stop],
            loss_mask=loss_mask[:, start:stop],
            target_last_hidden_states=target_last_hidden[:, start:stop],
            position_ids=torch.arange(
                start, stop, device="cuda", dtype=torch.long
            ).unsqueeze(0),
            attention_mask=torch.ones(
                (1, local_length), device="cuda", dtype=torch.long
            ),
        )

        global_ulysses_loss = ulysses_loss.detach().clone()
        dist.all_reduce(global_ulysses_loss, op=dist.ReduceOp.SUM)
        global_ulysses_loss /= world_size
        torch.testing.assert_close(
            global_ulysses_loss,
            reference_loss.detach(),
            rtol=3e-2,
            atol=3e-2,
        )

        reference_loss.backward()
        ulysses_loss.backward()
        reference_parameters = dict(reference.draft_model.named_parameters())
        for name, parameter in ulysses.draft_model.named_parameters():
            reference_grad = reference_parameters[name].grad
            if parameter.grad is None or reference_grad is None:
                if parameter.grad is not None or reference_grad is not None:
                    raise AssertionError(f"gradient presence mismatch for {name}")
                continue
            averaged_grad = parameter.grad.detach().clone()
            dist.all_reduce(averaged_grad, op=dist.ReduceOp.SUM)
            averaged_grad /= world_size
            torch.testing.assert_close(
                averaged_grad,
                reference_grad,
                rtol=6e-2,
                atol=6e-2,
                msg=lambda message, name=name: f"{name}: {message}",
            )
    finally:
        from specforge.distributed import destroy_distributed

        destroy_distributed()


@unittest.skipUnless(
    CUDA and NGPU >= WORLD_SIZE and _has_ulysses(),
    "requires two CUDA devices and Yunchang Ulysses",
)
class TestDSparkUSPParity(unittest.TestCase):
    def test_two_rank_loss_and_gradients_match_full_sequence(self):
        import torch.multiprocessing as mp

        from tests.utils import get_available_port

        with tempfile.TemporaryDirectory(prefix="dspark-usp-parity-") as workdir:
            mp.spawn(
                _worker,
                nprocs=WORLD_SIZE,
                args=(WORLD_SIZE, get_available_port(), workdir),
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
