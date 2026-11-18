"""Strict configuration encoding and decoding, with no file I/O."""

from .v1 import decode_config, encode_config, parse_draft

__all__ = ["decode_config", "encode_config", "parse_draft"]
