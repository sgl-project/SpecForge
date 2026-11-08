"""Migrate DFlash/DFlash2 structure without importing HF or allocating weights."""

from collections.abc import Mapping
from dataclasses import replace

from ...architecture import DFlash2Spec, DFlashSpec
from ...architecture.block import AnchoredBlockSpec
from ...conv import GroupedCausalConvSpec
from ...model import DraftConfig
from ..target import validate_target
from .context import _COMMON_FIELDS, _TOP_FIELDS, _check_fields, read_context

_DFLASH2_FIELDS = {
    "conv_kernel_size",
    "conv_group_size",
    "selector_rank",
    "selector_top_k",
    "output_multiplier",
    "final_logit_softcapping",
}


def decode_dflash(payload, *, target_config):
    architectures = payload["architectures"]
    _check_fields(payload, _TOP_FIELDS, "legacy draft")
    is_dflash2 = architectures[0] == "DFlash2DraftModel"
    method = payload.get("dflash_config")
    if method is None:
        method = {}
    if not isinstance(method, Mapping):
        raise ValueError("dflash_config must be an object or null")
    _check_fields(
        method,
        _COMMON_FIELDS | (_DFLASH2_FIELDS if is_dflash2 else set()),
        "dflash_config",
    )
    if method.get("projector_type") is not None:
        raise ValueError("DFlash/DFlash2 does not support a projector_type")
    fields, features, block_size = read_context(payload, method, target_config)
    block = AnchoredBlockSpec(block_size=block_size)
    convs = {}
    if is_dflash2:
        convs["default"] = GroupedCausalConvSpec(
            kernel_size=method.get("conv_kernel_size"),
            group_size=method.get("conv_group_size"),
        )
        spec = DFlash2Spec(
            block=block,
            target_features=features,
            selector_rank=method.get("selector_rank"),
            selector_top_k=method.get("selector_top_k"),
            output_multiplier=method.get("output_multiplier", 1.0),
            final_logit_softcapping=method.get("final_logit_softcapping"),
        )
    else:
        spec = DFlashSpec(block=block, target_features=features)
    fields["conv_configs"] = convs
    if convs:
        fields["layer_configs"] = {
            name: replace(layer, sublayer_conv="default")
            for name, layer in fields["layer_configs"].items()
        }
    config = DraftConfig(architecture=architectures[0], spec_config=spec, **fields)
    validate_target(config, target_config)
    return config
