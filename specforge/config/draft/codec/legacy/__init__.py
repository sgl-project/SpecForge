"""Explicit migrations of historical formats; no file or model imports."""

from collections.abc import Mapping

from .dflash import decode_dflash
from .domino import decode_domino
from .dspark import decode_dspark
from .dspine import decode_dspine
from .eagle3 import decode_eagle3
from .mtp import decode_mtp
from .peagle import decode_peagle

_DECODERS = {
    "DFlashDraftModel": decode_dflash,
    "DFlash2DraftModel": decode_dflash,
    "DSparkDraftModel": decode_dspark,
    "DSpineDraftModel": decode_dspine,
    "DominoDraftModel": decode_domino,
    "PEagleDraftModel": decode_peagle,
    "LlamaForCausalLMEagle3": decode_eagle3,
    "Qwen3_5MTPDraftModel": decode_mtp,
}


def decode_legacy_config(
    payload: Mapping,
    *,
    target_config: Mapping,
    capture_layers=None,
    mask_token_id=None,
    norm_before_residual=None,
):
    if not isinstance(payload, Mapping):
        raise ValueError("legacy draft must be a configuration object")
    architectures = payload.get("architectures")
    if (
        not isinstance(architectures, (list, tuple))
        or len(architectures) != 1
        or type(architectures[0]) is not str
    ):
        raise ValueError(
            f"unsupported legacy architectures; expected one of {sorted(_DECODERS)}"
        )
    architecture = architectures[0]
    if architecture == "DFlashDraftModel":
        method = payload.get("dflash_config")
        if isinstance(method, Mapping):
            projector = method.get("projector_type")
            if projector == "domino":
                architecture = "DominoDraftModel"
            elif projector == "dspark":
                architecture = "DSparkDraftModel"
    if architecture not in _DECODERS:
        raise ValueError(
            f"unsupported legacy architectures; expected one of {sorted(_DECODERS)}"
        )
    if architecture == "PEagleDraftModel":
        return decode_peagle(
            payload,
            target_config=target_config,
            capture_layers=capture_layers,
            mask_token_id=mask_token_id,
            norm_before_residual=norm_before_residual,
        )
    if mask_token_id is not None or norm_before_residual is not None:
        raise ValueError(
            "recorded mask/residual settings are only supported for legacy P-EAGLE"
        )
    if architecture == "LlamaForCausalLMEagle3":
        return _DECODERS[architecture](
            payload, target_config=target_config, capture_layers=capture_layers
        )
    if capture_layers is not None:
        raise ValueError("recorded capture_layers are only supported for legacy EAGLE")
    return _DECODERS[architecture](payload, target_config=target_config)
