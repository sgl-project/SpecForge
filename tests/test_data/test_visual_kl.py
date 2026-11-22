from types import SimpleNamespace

import pytest
import torch

from specforge.data.visual_kl import visual_kl_per_token


class Target:
    def __init__(self):
        self.calls = []
        self.model = SimpleNamespace(
            language_model=SimpleNamespace(
                layers=torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
            )
        )

    def __call__(
        self,
        *,
        input_ids,
        logits_to_keep,
        use_cache,
        output_hidden_states=False,
        **inputs,
    ):
        self.calls.append((input_ids.tolist(), inputs))
        state = input_ids.float().unsqueeze(-1)
        # Each target distribution depends on both the previous token and the pixels.
        pixel = inputs.get("pixel_values", torch.tensor(0)).sum()
        first, last = self.model.language_model.layers
        visible, pre_norm = first(state + pixel), last(state + 5)
        logits = torch.cat([state * 0, visible, -state], dim=-1)
        return SimpleNamespace(
            logits=logits[:, -logits_to_keep:],
            hidden_states=(
                (state, visible, torch.nn.functional.normalize(pre_norm, dim=-1))
                if output_hidden_states
                else None
            ),
        )


def test_aligns_predecessors_and_captures_image_conditioned_postblock_states():
    target = Target()
    seen = dict(
        input_ids=torch.tensor([[8, 9, 1, 2, 3]]),
        pixel_values=torch.tensor([2]),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
    )
    blind = dict(input_ids=torch.tensor([[1, 2, 3]]))
    kl, entropy, hidden = visual_kl_per_token(
        target, seen, blind, [0, 0, 0, 1, 1], target_layer_ids=[0, 1]
    )
    p = torch.tensor([[0.0, 3.0, -1.0], [0.0, 4.0, -2.0]]).log_softmax(-1)
    q = torch.tensor([[0.0, 1.0, -1.0], [0.0, 2.0, -2.0]]).log_softmax(-1)
    torch.testing.assert_close(kl, (p.exp() * (p - q)).sum(-1))
    torch.testing.assert_close(entropy, -(p.exp() * p).sum(-1))
    torch.testing.assert_close(
        hidden,
        torch.stack(
            (seen["input_ids"][0] + 2, seen["input_ids"][0] + 5), dim=-1
        ).float(),
    )
    assert len(target.calls) == 2
    assert (
        "pixel_values" in target.calls[0][1]
        and "pixel_values" not in target.calls[1][1]
    )


def test_refuses_mismatched_answer_or_blind_seen_branch():
    target = Target()
    seen = dict(
        input_ids=torch.tensor([[8, 1, 2, 3]]),
        pixel_values=torch.tensor([2]),
        image_grid_thw=torch.tensor([[1, 1, 1]]),
    )
    with pytest.raises(ValueError, match="align"):
        visual_kl_per_token(
            target, seen, {"input_ids": torch.tensor([[1, 0, 3]])}, [0, 0, 1, 1]
        )
    with pytest.raises(ValueError, match="pixel"):
        visual_kl_per_token(
            target,
            {"input_ids": seen["input_ids"]},
            {"input_ids": torch.tensor([[1, 2, 3]])},
            [0, 0, 1, 1],
        )
    assert not target.calls
