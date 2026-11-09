"""Small composed DSpine fixtures for loss, replay and distributed tests."""

from dataclasses import replace

import torch
from torch import nn

from specforge.algorithms.objectives import build_objective_model
from specforge.modeling import build_draft_model
from tests.test_modeling.test_canonical_dspine import model_config


def tiny_config(head_dim=4):
    base = model_config()
    return replace(
        base,
        hidden_size=16,
        vocab_size=32,
        draft_vocab_size=32,
        layer_types=(base.layer_types[0],) * 3,
        attention_configs={
            "full": replace(
                base.attention_configs["full"],
                num_attention_heads=16 // head_dim,
                num_key_value_heads=min(2, 16 // head_dim),
                head_dim=head_dim,
            )
        },
        mlp_configs={"dense": replace(base.mlp_configs["dense"], intermediate_size=32)},
        spec_config=replace(
            base.spec_config,
            target_features=replace(
                base.spec_config.target_features,
                mask_token_id=31,
                target_layer_ids=(1, 4, 6),
            ),
        ),
    )


def training_model(chunk_size=0, head_dim=4, backend="sdpa", **options):
    torch.manual_seed(17)
    draft = build_draft_model(tiny_config(head_dim), backend=backend)
    head = nn.Linear(16, 32, bias=False)
    draft.initialize_transfer_space(head.weight, chunk_size=7)
    with torch.no_grad():
        for write in draft.injection.writes:
            write.weight.normal_(std=0.05)
    return build_objective_model(
        draft,
        target_head=head,
        options=dict(
            num_anchors=4, top_k=3, objective_chunk_blocks=chunk_size, **options
        ),
    )


def training_inputs():
    return dict(
        input_ids=torch.randint(0, 30, (2, 12)),
        hidden_states=torch.randn(2, 12, 48),
        target_last_hidden_states=torch.randn(2, 12, 16),
        loss_mask=torch.tensor(
            [[0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1]]
        ),
    )
