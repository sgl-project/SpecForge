import json

import numpy as np
import pytest

from specforge.data.visual_score import VisualScoreTable, fingerprint_input_ids


def test_gate_uses_global_quantiles_then_entropy_and_preserves_positions(tmp_path):
    ids = [4, 4, 9, 10]
    mask = [0, 0, 1, 1]

    def row(key, kl, entropy):
        return dict(
            id=key,
            n_tokens=4,
            n_loss=2,
            fp=fingerprint_input_ids(ids),
            mask_fp=fingerprint_input_ids(mask),
            kl=kl,
            entropy=entropy,
        )

    rows = [row("a", [0, 1], [0, np.log(2)]), row("b", [2, 3], [0, 0])]
    path = tmp_path / "scores.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    table = VisualScoreTable.load(path)
    assert table.channel("a", ids, mask) == [0, 0, 10, 1670]
    assert table.channel("b", ids, mask) == [0, 0, 6670, 10000]
    with pytest.raises(ValueError, match="align"):
        table.channel("b", ids, [0, 1, 1, 0])
    with pytest.raises(ValueError, match="align"):
        table.channel("a", [4, 4, 8, 10], mask)
    with pytest.raises(ValueError, match="missing"):
        table.channel("missing", ids, mask)
