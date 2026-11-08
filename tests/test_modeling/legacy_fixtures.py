"""Small independent legacy configuration for canonical model parity tests."""

from transformers import Qwen3Config


def legacy_dflash_config(**overrides):
    config = Qwen3Config(
        **(
            dict(
                hidden_size=32,
                intermediate_size=64,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                num_hidden_layers=2,
                vocab_size=64,
                num_target_layers=8,
                rms_norm_eps=1e-6,
                attention_bias=False,
                attention_dropout=0,
                layer_types=["full_attention"] * 2,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
            )
            | overrides
        )
    )
    config._attn_implementation = "eager"
    return config
