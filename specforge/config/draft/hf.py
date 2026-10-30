"""Temporary HF save adapter. It never becomes a model's runtime config."""

from __future__ import annotations

import json

from transformers import PretrainedConfig

from .format import encode_format


class DraftSaveConfig(PretrainedConfig):
    """HF writes weights/metadata while config.json retains the canonical layout."""

    def __init__(self, config=None):
        adapter = config.to_hf_config(for_export=False) if config is not None else None
        super().__init__(**(adapter.to_dict() if adapter is not None else {}))
        self._canonical = encode_format(config) if config is not None else None

    def to_dict(self):
        values = super().to_dict()
        values.pop("_canonical", None)
        return values

    def to_json_string(self, use_diff=True):
        if self._canonical is None:
            return super().to_json_string(use_diff=use_diff)
        payload = dict(self._canonical)
        metadata = self.to_dict()
        for name in ("dtype", "architectures"):
            if name in metadata:
                payload[name] = metadata[name]
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"
