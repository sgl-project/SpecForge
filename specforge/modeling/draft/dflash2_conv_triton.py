"""Compatibility imports for the legacy DFlash2 model."""

from .layers.conv.triton import dflash2_grouped_conv_fused, supports_group_size

__all__ = ["dflash2_grouped_conv_fused", "supports_group_size"]
