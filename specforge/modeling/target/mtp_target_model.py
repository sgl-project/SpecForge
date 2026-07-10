# coding=utf-8
"""MTP target-model data generation, decoupled from Eagle3TargetModel.

All MTP-specific logic lives here, including VLM-aware language model and
transformer-layer discovery.  ``eagle3_target_model.py`` is left untouched
except for the ``capture_aux_hidden_states`` parameter on ``extend()``.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from specforge.modeling.target.eagle3_target_model import (
    CustomEagle3TargetModel,
    Eagle3TargetOutput,
    HFEagle3TargetModel,
    SGLangEagle3TargetModel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VLM-aware module discovery (standalone, not methods on Eagle3TargetModel)
# ---------------------------------------------------------------------------


def _get_language_model(target_model) -> nn.Module:
    """Return the inner transformer module (without the lm_head).

    Handles causal LMs (``model.model``), multimodal models whose text decoder
    is under ``model.language_model``, and falls back to a recursive search.
    """
    model = target_model.model

    # Fast paths for common layouts.
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model"):
            return lm.model
        if hasattr(lm, "layers"):
            return lm
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "layers"):
        return model

    # Recursive fallback: a module whose direct child is a decoder-layers
    # ModuleList. Handles VLMs whose text decoder is nested deeper.
    candidates = []
    for name, module in model.named_modules():
        for attr in ("layers", "h"):
            child = getattr(module, attr, None)
            if isinstance(child, nn.ModuleList) and len(child) > 0:
                candidates.append((name, module, len(child)))
                break
    if candidates:
        candidates.sort(key=lambda c: ("language_model" not in c[0], -c[2]))
        return candidates[0][1]
    return model


def _get_transformer_layers(target_model) -> nn.ModuleList:
    """Find the transformer layer ModuleList, with VLM support."""
    lm = _get_language_model(target_model)
    if isinstance(getattr(lm, "layers", None), nn.ModuleList):
        return lm.layers
    if isinstance(getattr(lm, "h", None), nn.ModuleList):
        return lm.h
    if hasattr(lm, "transformer") and isinstance(
        getattr(lm.transformer, "h", None), nn.ModuleList
    ):
        return lm.transformer.h
    raise ValueError(
        "Could not locate transformer layers. "
        f"model class={type(target_model.model).__name__}, "
        f"top-level children={[n for n, _ in target_model.model.named_children()]}"
    )


def generate_mtp_data(
    target_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    **kwargs,
) -> Eagle3TargetOutput:
    """Dispatch to the correct backend's MTP data generation.

    Unlike ``generate_eagle3_data``, MTP returns *raw* (un-shifted)
    ``input_ids`` and the target model's *last* hidden states.  The
    next-token shift is performed inside ``OnlineMTPModel``.
    """
    if isinstance(target_model, HFEagle3TargetModel):
        return _generate_mtp_data_hf(
            target_model, input_ids, attention_mask, loss_mask, **kwargs
        )
    elif isinstance(target_model, SGLangEagle3TargetModel):
        return _generate_mtp_data_sglang(
            target_model, input_ids, attention_mask, loss_mask, **kwargs
        )
    elif isinstance(target_model, CustomEagle3TargetModel):
        return _generate_mtp_data_custom(
            target_model, input_ids, attention_mask, loss_mask, **kwargs
        )
    raise NotImplementedError(
        f"MTP data generation not supported for {type(target_model).__name__}"
    )


@torch.no_grad()
def _generate_mtp_data_hf(
    target: HFEagle3TargetModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    **kwargs,
) -> Eagle3TargetOutput:
    """HF backend: capture last hidden state via forward hook on the last layer."""
    if kwargs:
        logger.debug(f"unused kwargs {list(kwargs.keys())}")

    layers = _get_transformer_layers(target)
    last_hidden_state = [None]

    def get_hook():
        def hook(module, inp, out):
            last_hidden_state[0] = out[0] if isinstance(out, tuple) else out

        return hook

    handle = layers[-1].register_forward_hook(get_hook())
    try:
        _get_language_model(target)(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
        )
    finally:
        handle.remove()

    last_hidden_states = last_hidden_state[0]
    loss_mask = loss_mask[..., None].to(last_hidden_states.device)

    return Eagle3TargetOutput(
        hidden_states=None,
        target=None,
        loss_mask=loss_mask,
        input_ids=input_ids,
        attention_mask=attention_mask,
        last_hidden_states=last_hidden_states,
    )


def _generate_mtp_data_sglang(
    target: SGLangEagle3TargetModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    pixel_values: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    is_vlm: bool = False,
    shard_returns: bool = False,
    **kwargs,
) -> Eagle3TargetOutput:
    """SGLang backend: capture last hidden state via extend()."""
    if kwargs:
        logger.debug(f"unused kwargs {list(kwargs.keys())}")

    if is_vlm:
        data_cache, _, _, last_hidden_states_list = target.extend_vlm(
            input_ids,
            attention_mask,
            loss_mask,
            return_last_hidden_states=True,
            return_logits=False,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
    else:
        data_cache, _, _, last_hidden_states_list = target.extend(
            input_ids,
            attention_mask,
            loss_mask,
            return_last_hidden_states=True,
            capture_aux_hidden_states=False,
            return_logits=False,
            shard_returns=shard_returns,
        )

    input_ids_out = []
    attention_mask_out = []
    loss_mask_out = []
    last_hidden_states_out = []

    for data, last_hidden_states in zip(data_cache, last_hidden_states_list):
        input_ids_out.append(data[0])
        attention_mask_out.append(data[1])
        loss_mask_out.append(data[2])
        if last_hidden_states is not None:
            last_hidden_states_out.append(last_hidden_states.unsqueeze(0))

    input_ids_out = torch.cat(input_ids_out, dim=0)
    attention_mask_out = torch.cat(attention_mask_out, dim=0)
    loss_mask_out = torch.cat(loss_mask_out, dim=0)
    last_hidden_states_out = torch.cat(last_hidden_states_out, dim=0)
    loss_mask_out = loss_mask_out[..., None]

    return Eagle3TargetOutput(
        hidden_states=None,
        target=None,
        loss_mask=loss_mask_out,
        input_ids=input_ids_out,
        attention_mask=attention_mask_out,
        last_hidden_states=last_hidden_states_out,
    )


@torch.no_grad()
def _generate_mtp_data_custom(
    target: CustomEagle3TargetModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    **kwargs,
) -> Eagle3TargetOutput:
    """Custom backend: full forward with output_hidden_states."""
    if kwargs:
        logger.debug(f"unused kwargs {list(kwargs.keys())}")

    outputs = target.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        layers_to_output_hidden_states=[-1],
        use_cache=False,
    )

    last_hidden_states = outputs.hidden_states[-1]
    loss_mask = loss_mask[..., None].to(last_hidden_states.device)

    return Eagle3TargetOutput(
        hidden_states=None,
        target=None,
        loss_mask=loss_mask,
        input_ids=input_ids,
        attention_mask=attention_mask,
        last_hidden_states=last_hidden_states,
    )
