"""Generate a trained legacy P-EAGLE fixture on PR #1109's legacy implementation."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.config import AutoDraftConfig
from specforge.modeling.draft.llama3_eagle import LlamaRotaryEmbedding
from tests.test_config.test_draft_legacy_peagle import peagle_payload
from tests.test_modeling.test_draft_peagle_codec import eager_attention

torch.manual_seed(42)
path = Path(sys.argv[1])
path.mkdir(parents=True, exist_ok=True)
payload = peagle_payload() | dict(
    hidden_size=32,
    target_hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    vocab_size=64,
    draft_vocab_size=64,
    max_position_embeddings=64,
    pad_token_id=0,
    bos_token_id=None,
    eos_token_id=None,
)
payload["eagle_config"] = {"eagle_aux_hidden_state_layer_ids": [1, 2, 3]}
model = AutoDraftModel.from_config(AutoDraftConfig.from_dict(payload))
ids = torch.tensor([[0, 1, 2, 1]])
features = torch.randn(1, 4, 96)
positions = torch.arange(4).unsqueeze(0)
mask = torch.ones(1, 1, 4, 4, dtype=torch.bool).tril()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
before = model.embed_tokens.weight.detach().clone()
with (
    patch(
        "specforge.modeling.draft.peagle.compile_friendly_flex_attention",
        side_effect=eager_attention,
    ),
    patch.object(
        LlamaRotaryEmbedding, "forward", LlamaRotaryEmbedding.forward.__wrapped__
    ),
):
    hidden = model(
        features, model.embed_tokens(ids), attention_mask=mask, position_ids=positions
    )
    model.lm_head(hidden).square().sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    with torch.no_grad():
        hidden = model(
            features,
            model.embed_tokens(ids),
            attention_mask=mask,
            position_ids=positions,
        )
        logits = model.lm_head(hidden)
assert not torch.equal(before[1], model.embed_tokens.weight[1])
assert torch.equal(before[0], model.embed_tokens.weight[0])
(path / "config.json").write_text(json.dumps(payload, indent=2) + "\n")
save_file(
    {k: v.detach().contiguous() for k, v in model.state_dict().items()},
    path / "model.safetensors",
)
save_file(
    dict(input_ids=ids, target_features=features, hidden_states=hidden, logits=logits),
    path / "reference.safetensors",
)
print("Saved trained legacy P-EAGLE fixture:", path)
