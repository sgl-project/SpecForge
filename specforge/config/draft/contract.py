"""Stable draft settings used to validate weights and resume training.

Keep architecture fields (including extensions) by default. Only serialization
metadata and execution-only HF options are excluded. This makes new structural
settings participate in validation without maintaining a second field list.
"""

from __future__ import annotations

import json
from typing import Any

# These do not describe the draft computation or its learned parameters.
_METADATA_FIELDS = frozenset(
    {
        "transformers_version",
        "auto_map",
        "dtype",
        "torch_dtype",
        "use_cache",
        "return_dict",
        "output_hidden_states",
        "output_attentions",
    }
)


def draft_config_contract(config) -> dict[str, Any]:
    """Return normalized, JSON-compatible settings without source provenance."""
    from .auto import AutoDraftConfig
    from .model import DraftModelConfig

    normalized = AutoDraftConfig.finalize(config)
    # Compare the same serialized representation for inline and legacy configs.
    if isinstance(normalized, DraftModelConfig):
        normalized = AutoDraftConfig.finalize(normalized.to_hf_config())
    payload = {
        key: value
        for key, value in normalized.to_dict().items()
        if not key.startswith("_") and key not in _METADATA_FIELDS
    }
    # Tuples in typed options and lists read from JSON must compare identically.
    return json.loads(json.dumps(payload))


def validate_draft_config_contract(current: dict, saved: dict, *, source: str) -> None:
    """Fail before loading weights; never repair the current configuration."""
    if not isinstance(saved, dict):
        raise ValueError(f"checkpoint {source!r} has invalid draft settings metadata")
    differences = []

    def compare(left, right, path):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(left.keys() | right.keys()):
                child = f"{path}.{key}" if path else key
                if key not in left or key not in right:
                    differences.append(
                        f"{child}: inline={left.get(key)!r}, saved={right.get(key)!r}"
                    )
                else:
                    compare(left[key], right[key], child)
        elif left != right:
            differences.append(f"{path}: inline={left!r}, saved={right!r}")

    compare(current, saved, "")
    if differences:
        raise ValueError(
            f"draft settings do not match checkpoint {source!r}: "
            + "; ".join(differences)
            + ". Align model.draft_model_config with the weights; saved settings are never applied."
        )
