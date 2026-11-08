"""Strict configuration encoding and decoding, with no file I/O."""

from .legacy.dflash import decode_legacy_config
from .v1 import decode_config, encode_config, parse_draft

__all__ = ["decode_config", "decode_legacy_config", "encode_config", "parse_draft"]
