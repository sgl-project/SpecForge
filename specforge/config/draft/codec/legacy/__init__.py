"""Explicit migrations of historical formats; no file or model imports."""

from collections.abc import Mapping

from .dflash import decode_dflash
from .domino import decode_domino
from .dspark import decode_dspark
from .dspine import decode_dspine
from .peagle import decode_peagle

_DECODERS = {
    "DFlashDraftModel": decode_dflash,
    "DFlash2DraftModel": decode_dflash,
    "DSparkDraftModel": decode_dspark,
    "DSpineDraftModel": decode_dspine,
    "DominoDraftModel": decode_domino,
    "PEagleDraftModel": decode_peagle,
}


def decode_legacy_config(payload: Mapping, *, target_config: Mapping):
    if not isinstance(payload, Mapping):
        raise ValueError("legacy draft must be a configuration object")
    architectures = payload.get("architectures")
    if (
        not isinstance(architectures, (list, tuple))
        or len(architectures) != 1
        or type(architectures[0]) is not str
        or architectures[0] not in _DECODERS
    ):
        raise ValueError(
            f"unsupported legacy architectures; expected one of {sorted(_DECODERS)}"
        )
    return _DECODERS[architectures[0]](payload, target_config=target_config)
