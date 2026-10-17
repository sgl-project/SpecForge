# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Target-capture policies: the algorithm half of target extraction.

A :class:`TargetCapturePolicy` owns everything about how one draft algorithm
extracts a *batched target output* from a frozen target model: what to load,
which layers to select, how each backend runs the frozen forward, and which
typed output record is returned. It deliberately does not know about
``PromptTask``, target-to-draft vocab projection, per-sample feature dicts, or
``FeatureStore`` writes; those are runtime adapter responsibilities.

This file is the algorithm side of the former ``algorithm x backend`` engine
matrix. The per-backend engines (``hf.py`` / ``sglang.py`` / ``custom.py``) stay
algorithm-free, and adding an algorithm is a policy subclass plus
``register_target_capture_policy``, not a new engine class per backend.

The policy is a *code* object, not a data table, because the HF side is a real
code difference: EAGLE3 captures 3 aux layers via forward hooks and needs the
target logits; DFlash reads ``output_hidden_states`` and selects layers with no
target distribution. The SGLang side shares :class:`SGLangCaptureBackend` and
differs only in build flags, the extend call, and output shaping.

The target-capture bodies here are the single implementation behind the
``{HF,SGLang,Custom}Eagle3TargetEngine`` / ``{HF,SGLang}DFlashTargetEngine``
methods; those classes delegate to these policies, so the extraction stays
byte-identical (enforced by ``test_phase_b_gate`` and the backend-parity
tests).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from specforge.utils import padding

logger = logging.getLogger(__name__)


class TargetCaptureBatch:
    """Marker base for the typed batched outputs a capture policy returns.

    Every ``TargetCapturePolicy`` capture body returns a ``TargetCaptureBatch``
    subclass (``Eagle3TargetOutput`` / ``DFlashTargetOutput``): batched tensors
    straight off the target forward, before any per-sample slicing, vocab
    projection, or feature-dict shaping (those are ``PolicyFeatureAdapter``
    responsibilities). The runtime adapter accepts only this type, which is
    what keeps the target-engine layer free of feature-store schema.
    """


@dataclass
class Eagle3TargetOutput(TargetCaptureBatch):
    hidden_states: torch.Tensor
    target: torch.Tensor
    loss_mask: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    last_hidden_states: Optional[torch.Tensor] = None


@dataclass
class DFlashTargetOutput(TargetCaptureBatch):
    hidden_states: torch.Tensor  # [batch, seq_len, n_capture*hidden_size]
    input_ids: torch.Tensor  # [batch, seq_len]
    attention_mask: torch.Tensor  # [batch, seq_len]
    loss_mask: torch.Tensor  # [batch, seq_len]
    # Target model's FINAL hidden state [batch, seq_len, hidden_size]. Optional:
    # DFlash never reads it, but DSpark's L1 distribution-distillation and
    # confidence-head losses need it (the frozen target LM head is applied to it
    # to form the soft next-token distribution). None when the backend does not
    # surface it (then DSpark must run CE-only).
    last_hidden_states: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class TargetCaptureSpec:
    """The declarative half of a policy: name + backend build/validation flags."""

    name: str
    #: required number of capture layers (None = any); EAGLE3 needs exactly 3.
    num_capture_layers: Optional[int] = None
    #: kwargs for SGLangCaptureBackend.build (wrap_eagle3_logits etc.).
    sglang_build_kwargs: Dict[str, Any] = field(default_factory=dict)
    #: push capture layers to the sglang model strictly, or only if supported.
    sglang_strict_capture_layers: bool = True


class TargetCapturePolicy(ABC):
    """How one draft algorithm captures batched target outputs on each backend."""

    spec: TargetCaptureSpec

    def resolve_capture_layers(
        self, model_config, layer_ids: Optional[List[int]]
    ) -> Optional[List[int]]:
        """Validate/default the capture-layer selection for hf/custom backends."""
        return layer_ids

    @abstractmethod
    def hf_load(
        self,
        pretrained_model_name_or_path: str,
        torch_dtype: Optional[torch.dtype],
        device: Optional[str],
        cache_dir: Optional[str],
        **kwargs,
    ) -> nn.Module: ...

    @abstractmethod
    def hf_capture(
        self,
        model: nn.Module,
        capture_layers: Optional[List[int]],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        **kwargs,
    ) -> TargetCaptureBatch: ...

    @abstractmethod
    def sglang_capture(
        self,
        backend,  # sglang_backend.SGLangCaptureBackend
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        **kwargs,
    ) -> TargetCaptureBatch: ...

    def custom_capture(
        self,
        model: nn.Module,
        capture_layers: Optional[List[int]],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        **kwargs,
    ) -> TargetCaptureBatch:
        raise NotImplementedError(
            f"{self.spec.name} has no custom-backend capture implementation"
        )


def _get_transformer_layers(model: nn.Module):
    """Find the module list containing the transformer layers.

    Adapts to common architectures (Llama, Qwen, Mistral, OPT, etc.)
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    elif hasattr(model, "layers"):
        return model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    else:
        raise ValueError(
            "Could not locate transformer layers in the model architecture to register hooks."
        )


class Eagle3CapturePolicy(TargetCapturePolicy):
    """EAGLE3: 3 aux hidden-state layers + target logits."""

    spec = TargetCaptureSpec(
        name="eagle3",
        num_capture_layers=3,
        sglang_build_kwargs={"wrap_eagle3_logits": True, "return_full_logits": False},
        sglang_strict_capture_layers=True,
    )

    def resolve_capture_layers(self, model_config, layer_ids):
        if layer_ids is None:
            if hasattr(model_config, "num_hidden_layers"):
                num_layers = model_config.num_hidden_layers
            else:
                raise ValueError(
                    f"Failed to set aux hidden states layers as model config {model_config} does not have num_hidden_layers"
                )
            layer_ids = [
                1,
                num_layers // 2 - 1,
                num_layers - 4,
            ]
        assert (
            len(layer_ids) == 3
        ), "aux_hidden_states_layers is expected to be 3 layers for EAGLE3"
        return layer_ids

    def hf_load(
        self,
        pretrained_model_name_or_path,
        torch_dtype,
        device,
        cache_dir,
        **kwargs,
    ):
        from specforge.distributed import get_tp_device_mesh, get_tp_group

        tp_size = get_tp_group().size()

        if tp_size > 1:
            device_kwargs = {
                "tp_plan": "auto",
                "tp_size": tp_size,
                "device_mesh": get_tp_device_mesh(),
            }
        else:
            device_kwargs = {
                "device_map": device,
            }

        return AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            **device_kwargs,
            **kwargs,
        )

    @torch.no_grad()
    def hf_capture(
        self,
        model,
        capture_layers,
        input_ids,
        attention_mask,
        loss_mask,
        **kwargs,
    ) -> Eagle3TargetOutput:
        """Capture only the required layers via forward hooks (memory-light)."""
        if kwargs:
            logger.debug(f"unused kwargs {list(kwargs.keys())}")

        captured_states = {}
        handles = []

        def get_hook(layer_idx):
            def hook(module, input, output):
                # HF layer outputs are usually tuples (hidden_states, present_key_value, ...)
                hidden = output[0] if isinstance(output, tuple) else output
                captured_states[layer_idx] = hidden

            return hook

        layers = _get_transformer_layers(model)
        target_indices = capture_layers

        for idx in target_indices:
            if 0 <= idx < len(layers):
                handles.append(layers[idx].register_forward_hook(get_hook(idx)))
            else:
                raise ValueError(
                    f"Layer index {idx} out of bounds for model with {len(layers)} layers."
                )

        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                output_attentions=False,
                output_router_logits=False,
                use_cache=False,
            )
            target = outputs.logits
        finally:
            # Always remove hooks to prevent leaks or side effects on later calls
            for handle in handles:
                handle.remove()

        if len(captured_states) != 3:
            raise RuntimeError(
                f"Expected to capture 3 layers, but captured {len(captured_states)}"
            )

        hidden_states = torch.cat(
            (
                captured_states[target_indices[0]],
                captured_states[target_indices[1]],
                captured_states[target_indices[2]],
            ),
            dim=-1,
        )

        target = padding(outputs.logits, left=False)
        input_ids = padding(input_ids, left=False)
        loss_mask = loss_mask[..., None].to(target.device)

        return Eagle3TargetOutput(
            hidden_states=hidden_states,
            target=target,
            loss_mask=loss_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    @torch.no_grad()
    def sglang_capture(
        self,
        backend,
        input_ids,
        attention_mask,
        loss_mask,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        is_vlm: bool = False,
        shard_returns: bool = False,
        **kwargs,
    ) -> Eagle3TargetOutput:
        if kwargs:
            logger.debug(f"unused kwargs {list(kwargs.keys())}")

        if is_vlm:
            data_cache, logits_list, aux_hidden_states_list, last_hidden_states_list = (
                backend.extend_eagle3_vlm(
                    input_ids,
                    attention_mask,
                    loss_mask,
                    return_last_hidden_states=False,
                    return_logits=True,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )
            )
        else:
            data_cache, logits_list, aux_hidden_states_list, last_hidden_states_list = (
                backend.extend_eagle3(
                    input_ids,
                    attention_mask,
                    loss_mask,
                    return_last_hidden_states=False,
                    return_logits=True,
                    shard_returns=shard_returns,
                )
            )
        aux_hidden_states_out = []
        target_out = []
        loss_mask_out = []
        attention_mask_out = []
        input_ids_out = []
        last_hidden_states_out = []

        for data, logits, aux_hidden_states, last_hidden_states in zip(
            data_cache, logits_list, aux_hidden_states_list, last_hidden_states_list
        ):
            if aux_hidden_states is not None:
                aux_hidden_states_out.append(aux_hidden_states.unsqueeze(0))
                loss_mask_out.append(data[2])
                attention_mask_out.append(data[1])
                input_ids_out.append(data[0])

            # offline hidden-state dumps keep last_hidden_states and no logits;
            # online training keeps logits and no last_hidden_states.
            if logits is not None:
                target_out.append(logits.unsqueeze(0))

            if last_hidden_states is not None:
                last_hidden_states_out.append(last_hidden_states.unsqueeze(0))

        aux_hidden_states_out = torch.cat(aux_hidden_states_out, dim=0)

        loss_mask_out = torch.cat(loss_mask_out, dim=0)
        attention_mask_out = torch.cat(attention_mask_out, dim=0)
        input_ids_out = torch.cat(input_ids_out, dim=0)

        target_out = torch.cat(target_out, dim=0) if target_out else None
        last_hidden_states_out = (
            torch.cat(last_hidden_states_out, dim=0) if last_hidden_states_out else None
        )

        if target_out is not None:
            target_out = padding(target_out, left=False)
        input_ids_out = padding(input_ids_out, left=False)
        loss_mask_out = loss_mask_out[..., None]

        return Eagle3TargetOutput(
            hidden_states=aux_hidden_states_out,
            target=target_out,
            loss_mask=loss_mask_out,
            input_ids=input_ids_out,
            attention_mask=attention_mask_out,
            last_hidden_states=last_hidden_states_out,
        )

    @torch.no_grad()
    def custom_capture(
        self,
        model,
        capture_layers,
        input_ids,
        attention_mask,
        loss_mask,
        **kwargs,
    ) -> Eagle3TargetOutput:
        if kwargs:
            logger.debug(f"unused kwargs {list(kwargs.keys())}")

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            layers_to_output_hidden_states=capture_layers,
            use_cache=False,
        )

        # Custom model implementations are responsible for returning only the
        # requested layers in `outputs.hidden_states`.
        hidden_states = torch.cat(outputs.hidden_states, dim=-1)

        target = padding(outputs.logits, left=False)
        input_ids = padding(input_ids, left=False)
        loss_mask = loss_mask[..., None].to(target.device)

        return Eagle3TargetOutput(
            hidden_states=hidden_states,
            target=target,
            loss_mask=loss_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


class DFlashCapturePolicy(TargetCapturePolicy):
    """DFlash: concatenated hidden states of arbitrary layers, no logits."""

    spec = TargetCaptureSpec(
        name="dflash",
        num_capture_layers=None,
        # Wrap the sglang logits processor (as eagle3 does) so the post-norm final
        # hidden can be surfaced via return_last_hidden_states for DSpark's L1 /
        # confidence losses. DFlash itself needs only the aux concat and asks for no
        # vocab logits (return_logits=False in extend_dflash), so this adds no
        # eagle3 vocab dependency.
        sglang_build_kwargs={"wrap_eagle3_logits": True},
        sglang_strict_capture_layers=False,
    )

    def hf_load(
        self,
        pretrained_model_name_or_path,
        torch_dtype,
        device,
        cache_dir,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            output_hidden_states=True,
            trust_remote_code=trust_remote_code,
            **kwargs,
        ).eval()

        if device:
            model = model.to(device)
        return model

    @torch.no_grad()
    def hf_capture(
        self,
        model,
        capture_layers,
        input_ids,
        attention_mask,
        loss_mask,
        **kwargs,
    ) -> DFlashTargetOutput:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # hidden_states[0] = embedding output; hidden_states[i+1] = layer i output;
        # hidden_states[-1] = final (post-norm) hidden, i.e. the LM-head input.
        offset = 1
        if capture_layers is not None:
            selected = [outputs.hidden_states[idx + offset] for idx in capture_layers]
            hidden_states = torch.cat(selected, dim=-1)
        else:
            hidden_states = outputs.hidden_states[-1]

        # Final hidden state for DSpark's L1 / confidence losses (DFlash ignores it).
        last_hidden_states = outputs.hidden_states[-1]

        return DFlashTargetOutput(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            last_hidden_states=last_hidden_states,
        )

    @torch.no_grad()
    def sglang_capture(
        self,
        backend,
        input_ids,
        attention_mask,
        loss_mask,
        **kwargs,
    ) -> DFlashTargetOutput:
        # context = the captured (aux) mid-layer concat used by DFlash; final = the
        # post-norm last-layer hidden, surfaced for DSpark's L1 / confidence losses
        # (None when the runner only returned a single hidden stream).
        data_cache, hidden_states_list, last_hidden_states_list = backend.extend_dflash(
            input_ids, attention_mask, loss_mask, return_last_hidden_states=True
        )

        hidden_states = torch.cat([h.unsqueeze(0) for h in hidden_states_list], dim=0)
        last_hidden_states = None
        if all(h is not None for h in last_hidden_states_list):
            last_hidden_states = torch.cat(
                [h.unsqueeze(0) for h in last_hidden_states_list], dim=0
            )
        input_ids = torch.cat([d[0] for d in data_cache], dim=0)
        attention_mask = torch.cat([d[1] for d in data_cache], dim=0)
        loss_mask = torch.cat([d[2] for d in data_cache], dim=0)

        return DFlashTargetOutput(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            last_hidden_states=last_hidden_states,
        )


TARGET_CAPTURE_POLICIES: Dict[str, TargetCapturePolicy] = {}
# Back-compat name retained for existing imports/tests.
CAPTURE_POLICIES = TARGET_CAPTURE_POLICIES


def register_target_capture_policy(name: str, policy: TargetCapturePolicy) -> None:
    existing = TARGET_CAPTURE_POLICIES.get(name)
    if existing is not None and type(existing) is not type(policy):
        raise ValueError(
            f"target capture policy {name!r} already registered to "
            f"{type(existing).__name__}"
        )
    TARGET_CAPTURE_POLICIES[name] = policy


def resolve_target_capture_policy(name: str) -> TargetCapturePolicy:
    try:
        return TARGET_CAPTURE_POLICIES[name]
    except KeyError:
        raise KeyError(
            f"unknown target capture policy {name!r}; "
            f"registered: {sorted(TARGET_CAPTURE_POLICIES)}"
        ) from None


def register_capture_policy(name: str, policy: TargetCapturePolicy) -> None:
    """Back-compat alias for :func:`register_target_capture_policy`."""
    register_target_capture_policy(name, policy)


def resolve_capture_policy(name: str) -> TargetCapturePolicy:
    """Back-compat alias for :func:`resolve_target_capture_policy`."""
    return resolve_target_capture_policy(name)


register_target_capture_policy("eagle3", Eagle3CapturePolicy())
register_target_capture_policy("dflash", DFlashCapturePolicy())
# Domino trains on the same captured features as DFlash (same capture path).
register_target_capture_policy("domino", TARGET_CAPTURE_POLICIES["dflash"])


# Back-compat aliases for the original shorter names. New code should prefer
# TargetCaptureSpec/TargetCapturePolicy to avoid confusion with
# inference.capture.FeatureContract.
CaptureSpec = TargetCaptureSpec
CapturePolicy = TargetCapturePolicy


__all__ = [
    "TargetCaptureBatch",
    "TargetCaptureSpec",
    "TargetCapturePolicy",
    "CaptureSpec",
    "CapturePolicy",
    "Eagle3CapturePolicy",
    "DFlashCapturePolicy",
    "Eagle3TargetOutput",
    "DFlashTargetOutput",
    "TARGET_CAPTURE_POLICIES",
    "CAPTURE_POLICIES",
    "register_target_capture_policy",
    "resolve_target_capture_policy",
    "register_capture_policy",
    "resolve_capture_policy",
]
