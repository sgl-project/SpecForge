"""Convolution constraints that depend on the containing draft's dimensions."""

from .grouped_causal import GroupedCausalConvSpec


def validate_grouped_causal(
    config: GroupedCausalConvSpec, *, hidden_size: int, num_model_slots: int | None
) -> None:
    if hidden_size % config.group_size:
        raise ValueError("group_size must divide hidden_size")
    if num_model_slots is None:
        raise ValueError("grouped_causal requires a block protocol")
    if config.kernel_size > num_model_slots:
        raise ValueError("kernel_size exceeds the model slot count")
