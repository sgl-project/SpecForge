"""Migrate the effective single-layer EAGLE3 model, including fusion norms."""

from ...architecture import Eagle3Spec
from ...model import DraftConfig
from ..target import validate_target
from .auxiliary import _FIELDS, read_auxiliary
from .context import _check_fields
from .eagle_rope import resolve_eagle_rope


def decode_eagle3(payload, *, target_config, capture_layers=None):
    _check_fields(
        payload,
        _FIELDS
        | {"fc_norm", "norm_output", "target_model_type", "decoder_sparse_step"},
        "legacy EAGLE3",
    )
    # The old model creates one midlayer regardless of the HF depth field.
    fields, features = read_auxiliary(
        payload, target_config, num_layers=1, capture_layers=capture_layers
    )
    fields["target_bindings"] = ("embed_tokens",)
    config = DraftConfig(
        architecture="Eagle3DraftModel",
        rope_parameters=resolve_eagle_rope(payload),
        spec_config=Eagle3Spec(
            target_features=features,
            normalize_features=(
                False if payload.get("fc_norm") is None else payload["fc_norm"]
            ),
            norm_output=(
                False if payload.get("norm_output") is None else payload["norm_output"]
            ),
        ),
        **fields,
    )
    validate_target(config, target_config)
    return config
