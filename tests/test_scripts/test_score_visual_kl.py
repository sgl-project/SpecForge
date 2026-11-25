import json

import pytest
import torch

from specforge.offline_capture.mmflash import prepare_features
from tests.test_data.test_visual_kl import Target


def test_prepares_same_rows_for_both_trainers_and_publishes_atomically(
    tmp_path, monkeypatch
):
    inputs = tmp_path / "input.jsonl"
    outputs = tmp_path / "features"
    scores = tmp_path / "scores.jsonl"
    inputs.write_text('{"id":"image"}\n{"id":"text"}\n{"id":"short"}\n')

    def encode(record, *args, **kwargs):
        # Image rows share a common blind response suffix. The short row
        # contributes to the authors' global KL quantiles but cannot train.
        image = record["id"] != "text"
        ids = [1, 4, 4, 4, 4, 4] if record["id"] != "short" else [1, 4]
        blind = {"input_ids": torch.tensor([ids])}
        seen = {
            "input_ids": torch.tensor([[8, 9] + ids]),
            "pixel_values": torch.ones(1),
            "image_grid_thw": torch.ones(1, 3),
        }
        mask = [0] + [1] * (len(ids) - 1)
        return (seen, blind, [0, 0] + mask) if image else (blind, None, mask)

    monkeypatch.setattr("specforge.offline_capture.mmflash.encode_record", encode)
    monkeypatch.setattr(
        "specforge.offline_capture.mmflash.assistant_delimiters", lambda _: ([1], 0)
    )
    kwargs = dict(
        target=Target(),
        processor=None,
        layer_ids=[0],
        block_size=2,
        max_length=20,
        last_turn_only=False,
        device="cpu",
    )
    prepare_features(inputs, scores, outputs, **kwargs)
    assert [json.loads(line)["id"] for line in scores.read_text().splitlines()] == [
        "image",
        "short",
    ]
    assert sorted(p.name for p in outputs.glob("*.ckpt")) == [
        "00000001.ckpt",
        "00000002.ckpt",
    ]
    image = torch.load(outputs / "00000001.ckpt", weights_only=True)
    text = torch.load(outputs / "00000002.ckpt", weights_only=True)
    assert image["hidden_states"].shape == (8, 1)
    assert image["visual_score"].dtype == torch.long
    assert all(image["visual_score"][image["loss_mask"] == 0] == 0)
    assert all(text["visual_score"] == -1)
    with pytest.raises(FileExistsError):
        prepare_features(inputs, scores, outputs, **kwargs)

    bad_scores, bad_outputs = tmp_path / "bad.jsonl", tmp_path / "bad"
    inputs.write_text('{"id":"image"}\n{"id":"image"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        prepare_features(inputs, bad_scores, bad_outputs, **kwargs)
    assert not bad_scores.exists() and not bad_outputs.exists()
