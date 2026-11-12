"""Pinned SGLang configuration for dense Qwen3.5 MTP compatibility."""

from specforge.config.draft.rope import DefaultRoPESpec, MultiAxisRoPESpec


def convert_config(config):
    from .converters import common, require

    require(
        config.target_bindings == ("embed_tokens", "lm_head"),
        "native MTP requires both target bindings",
    )
    require(
        config.num_hidden_layers == 1
        and config.spec_config.target_features.hidden_size == config.hidden_size,
        "native Qwen3.5 MTP requires one layer and equal feature width",
    )
    result, a, _ = common(config, native=True)
    require(
        config.norm_config.weight_offset == 1
        and not a.attention_bias
        and a.sliding_window is None,
        "native Qwen3.5 requires offset RMSNorm and full attention without biases",
    )
    require(
        type(a.rope_parameters or config.rope_parameters)
        in (DefaultRoPESpec, MultiAxisRoPESpec),
        "native Qwen3.5 supports default or multi-axis RoPE only",
    )
    result.update(attn_output_gate=a.output_gate)
    return result
