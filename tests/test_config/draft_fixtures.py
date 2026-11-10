"""Small effective draft configurations shared by configuration tests."""

from specforge.config.draft import DraftConfig
from specforge.config.draft.architecture import DFlashSpec
from specforge.config.draft.architecture.block import AnchoredBlockSpec
from specforge.config.draft.architecture.features import TargetFeatureSpec
from specforge.config.draft.attention import GQASpec
from specforge.config.draft.layer import LayerSpec
from specforge.config.draft.logits import NoLogitTransformSpec
from specforge.config.draft.mlp import DenseFFNSpec
from specforge.config.draft.norm import RMSNormSpec
from specforge.config.draft.residual import StandardResidualSpec
from specforge.config.draft.rope import DefaultRoPESpec


def draft_config(**overrides):
    # binds the target's tables, so the target width follows hidden_size
    hidden_size = overrides.get("hidden_size", 32)
    return DraftConfig(
        **{
            "architecture": "DFlashDraftModel",
            "hidden_size": hidden_size,
            "vocab_size": 64,
            "draft_vocab_size": 64,
            "max_position_embeddings": 256,
            "initializer_range": 0.02,
            "tie_word_embeddings": False,
            "attention_configs": {
                "full": GQASpec(
                    rope_parameters=None,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=8,
                    qk_norm=True,
                    attention_bias=False,
                    output_gate=False,
                    is_causal=False,
                    sliding_window=None,
                    attention_dropout=0.0,
                )
            },
            "mlp_configs": {
                "dense": DenseFFNSpec(intermediate_size=64, hidden_act="silu")
            },
            "conv_configs": {},
            "layer_configs": {"full_dense": LayerSpec(attention="full", mlp="dense")},
            "layer_types": ("full_dense", "full_dense"),
            "norm_config": RMSNormSpec(eps=1e-6, weight_offset=0),
            "residual_config": StandardResidualSpec(),
            "rope_parameters": DefaultRoPESpec(
                rope_theta=10000, partial_rotary_factor=1.0
            ),
            "spec_config": DFlashSpec(
                block=AnchoredBlockSpec(block_size=8),
                target_features=TargetFeatureSpec(
                    hidden_size=hidden_size, mask_token_id=63, target_layer_ids=(1, 3)
                ),
            ),
            "target_bindings": ("embed_tokens", "lm_head"),
            "target_logit_transform": NoLogitTransformSpec(),
            **overrides,
        }
    )
