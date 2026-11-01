"""Convolution consumes draft-block boundaries, not an attention backend."""

from .dflash2 import DFlashGroupedConv


def build_grouped(*, spec, config):
    return DFlashGroupedConv(
        hidden_size=config.hidden_size,
        block_size=config.block_size,
        taps=spec.kernel_size,
        group_size=spec.group_size,
    )
